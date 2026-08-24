// CUDA device interpreter for lowered sampling plans.
//
// Three execution tiers share one action semantics (mirroring the HIP
// vertical slice so conformance results transfer):
//   * thread-per-shot: one thread walks a whole shot; only sensible when the
//     coefficient state is tiny (peak active width <= 4).
//   * block-per-shot, shared: one block walks a whole shot with the
//     coefficients resident in opt-in shared memory. One HBM load and store
//     per shot instead of one sweep per action.
//   * block-per-shot, global: the same cooperative kernel with the state in a
//     per-block global-memory slab, for widths past the shared budget.
//
// Scalar control flow (RNG draws, branch selection, frame actions) is
// executed redundantly by every thread in the block from identical inputs, so
// no broadcast storage is needed; byte outputs are written by thread 0 only.

#include "clifft/sampling/cuda/device_program.h"
#include "clifft/util/numeric.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace clifft::sampling::cuda::detail {

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr double kLogHalf = -0.693147180559945309417232121458176568;

template <typename Value>
struct ComplexValue {
    Value real = 0;
    Value imag = 0;
};

struct Xoshiro256PlusPlus {
    uint64_t state[4]{};

    __device__ __forceinline__ static uint64_t rotate_left(uint64_t value, uint32_t shift) {
        return (value << shift) | (value >> (64U - shift));
    }

    __device__ __forceinline__ uint64_t next() {
        const uint64_t result = rotate_left(state[0] + state[3], 23) + state[0];
        const uint64_t temporary = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= temporary;
        state[3] = rotate_left(state[3], 45);
        return result;
    }

    __device__ __forceinline__ double next_double() {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }
};

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ Xoshiro256PlusPlus shot_rng(SeedRoot root, uint64_t shot) {
    Xoshiro256PlusPlus rng;
    constexpr uint64_t kSamplingExecutorDomain = 0x01;
    for (uint64_t word = 0; word < 4; ++word) {
        const uint64_t stream = ((kSamplingExecutorDomain << 2) | word) * 0xBF58476D1CE4E5B9ULL;
        rng.state[word] = mix64(root.words[word] ^ (shot * 0x9E3779B97F4A7C15ULL) ^ stream);
    }
    return rng;
}

__device__ __forceinline__ bool parity(uint64_t value) {
    return (__popcll(value) & 1U) != 0;
}

template <typename Value>
__device__ __forceinline__ ComplexValue<Value> phase_at(const Action& action, uint64_t basis) {
    const Value sign = parity(basis & action.z) ? Value{-1} : Value{1};
    return {sign * static_cast<Value>(action.phase_real),
            sign * static_cast<Value>(action.phase_imag)};
}

template <typename Coefficient>
__device__ __forceinline__ ComplexValue<Coefficient> load(const Coefficient* real,
                                                          const Coefficient* imag,
                                                          uint64_t index) {
    return {real[index], imag[index]};
}

template <typename Coefficient>
__device__ __forceinline__ void store(Coefficient* real, Coefficient* imag, uint64_t index,
                                      ComplexValue<Coefficient> value) {
    real[index] = value.real;
    imag[index] = value.imag;
}

template <typename Coefficient>
__device__ __forceinline__ double norm(ComplexValue<Coefficient> value) {
    const double real = static_cast<double>(value.real);
    const double imag = static_cast<double>(value.imag);
    return real * real + imag * imag;
}

template <typename Value>
__device__ __forceinline__ ComplexValue<Value> multiply(ComplexValue<Value> left,
                                                        ComplexValue<Value> right) {
    return {left.real * right.real - left.imag * right.imag,
            left.real * right.imag + left.imag * right.real};
}

template <typename Coefficient>
__device__ __forceinline__ ComplexValue<Coefficient> normalize(ComplexValue<Coefficient> value,
                                                               double inverse_norm) {
    return {static_cast<Coefficient>(inverse_norm * static_cast<double>(value.real)),
            static_cast<Coefficient>(inverse_norm * static_cast<double>(value.imag))};
}

__device__ __forceinline__ uint64_t insert_zero_bit(uint64_t value, uint32_t bit) {
    const uint64_t low_mask = (uint64_t{1} << bit) - 1;
    return (value & low_mask) | ((value & ~low_mask) << 1);
}

__device__ __forceinline__ bool evaluate(const ProgramView& program, const uint8_t* symbols,
                                         uint32_t expression_index) {
    const Expression expression = program.expressions[expression_index];
    bool result = expression.constant != 0;
    for (uint32_t offset = 0; offset < expression.term_count; ++offset) {
        result ^= symbols[program.expression_terms[expression.term_begin + offset]] != 0;
    }
    return result;
}

template <typename Coefficient>
__device__ __forceinline__ ComplexValue<Coefficient> compact_nondiagonal(
    const Action& action, const Coefficient* real, const Coefficient* imag, uint64_t packed,
    bool branch) {
    const uint64_t source0 = insert_zero_bit(packed, action.index2);
    const uint64_t source1 = source0 ^ action.x;
    const ComplexValue<Coefficient> left = load(real, imag, source0);
    const ComplexValue<Coefficient> right = load(real, imag, source1);
    ComplexValue<Coefficient> conjugate_phase = phase_at<Coefficient>(action, source0);
    conjugate_phase.imag = -conjugate_phase.imag;
    ComplexValue<Coefficient> transformed = multiply(conjugate_phase, right);
    const Coefficient eigenvalue = branch ? Coefficient{-1} : Coefficient{1};
    transformed.real *= eigenvalue;
    transformed.imag *= eigenvalue;
    const Coefficient scale = static_cast<Coefficient>(kInvSqrt2);
    return {scale * (left.real + transformed.real), scale * (left.imag + transformed.imag)};
}

__device__ __forceinline__ void sample_noise(const ProgramView& program, Xoshiro256PlusPlus& rng,
                                             uint8_t* symbols, bool is_writer) {
    for (uint32_t site_index = 0; site_index < program.noise_site_count; ++site_index) {
        const NoiseSite site = program.noise_sites[site_index];
        if (site.outcome_count == 0 || site.execution_probability <= 0.0) {
            continue;
        }
        const double draw = rng.next_double();
        if (draw >= site.execution_probability) {
            continue;
        }
        uint32_t outcome = site.outcome_begin;
        const uint32_t end = site.outcome_begin + site.outcome_count;
        while (outcome + 1 < end &&
               draw >= program.noise_outcomes[outcome].cumulative_probability) {
            ++outcome;
        }
        if (is_writer) {
            symbols[program.noise_outcomes[outcome].symbol] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Cooperative block primitives. lane/lanes describe this thread's slice of
// the shot; for the thread-per-shot tier they are 0/1 and the loops below
// degenerate to the sequential HIP-slice bodies, so both tiers share one
// implementation of every action.
// ---------------------------------------------------------------------------

struct Lane {
    uint32_t lane = 0;
    uint32_t lanes = 1;
    // Reduction scratch with room for 2 * lanes doubles; unused when lanes==1.
    double* reduce = nullptr;

    __device__ __forceinline__ bool is_writer() const { return lane == 0; }

    __device__ __forceinline__ void sync() const {
        if (lanes > 1) {
            __syncthreads();
        }
    }

    // Tree-reduces the pair (a, b) across the block; every lane returns the
    // same totals. Requires lanes to be a power of two.
    __device__ __forceinline__ void reduce_pair(double& a, double& b) const {
        if (lanes == 1) {
            return;
        }
        reduce[lane] = a;
        reduce[lanes + lane] = b;
        __syncthreads();
        for (uint32_t half = lanes >> 1; half != 0; half >>= 1) {
            if (lane < half) {
                reduce[lane] += reduce[lane + half];
                reduce[lanes + lane] += reduce[lanes + lane + half];
            }
            __syncthreads();
        }
        a = reduce[0];
        b = reduce[lanes];
        __syncthreads();
    }
};

template <typename Coefficient>
__device__ void apply_rotation(const Action& action, bool sign, Coefficient* real,
                               Coefficient* imag, Lane lane) {
    const uint64_t size = uint64_t{1} << action.active_before;
    const Coefficient cosine = static_cast<Coefficient>(action.value0);
    const Coefficient sine =
        sign ? -static_cast<Coefficient>(action.value1) : static_cast<Coefficient>(action.value1);
    if (action.x == 0) {
        for (uint64_t basis = lane.lane; basis < size; basis += lane.lanes) {
            const Coefficient eigenvalue =
                parity(basis & action.z) ? Coefficient{-1} : Coefficient{1};
            const Coefficient signed_sine = sine * eigenvalue;
            const Coefficient input_real = real[basis];
            const Coefficient input_imag = imag[basis];
            real[basis] = cosine * input_real + signed_sine * input_imag;
            imag[basis] = cosine * input_imag - signed_sine * input_real;
        }
        lane.sync();
        return;
    }

    const bool real_phase = action.phase_real != 0;
    const Coefficient base_phase =
        static_cast<Coefficient>(real_phase ? action.phase_real : action.phase_imag);
    const Coefficient even_left_sine = sine * base_phase;
    const uint64_t pair_stride = action.pair_stride_or_z_without_pivot;
    const uint64_t low_mask = pair_stride - 1;
    const uint64_t pairs = size >> 1;
    for (uint64_t pair = lane.lane; pair < pairs; pair += lane.lanes) {
        const uint64_t left = ((pair & ~low_mask) << 1) | (pair & low_mask);
        const uint64_t right = left ^ action.x;
        const Coefficient left_real = real[left];
        const Coefficient left_imag = imag[left];
        const Coefficient right_real = real[right];
        const Coefficient right_imag = imag[right];
        const Coefficient left_sine = parity(left & action.z) ? -even_left_sine : even_left_sine;
        const Coefficient right_sine = real_phase ? left_sine : -left_sine;
        if (real_phase) {
            real[left] = cosine * left_real + right_sine * right_imag;
            imag[left] = cosine * left_imag - right_sine * right_real;
            real[right] = cosine * right_real + left_sine * left_imag;
            imag[right] = cosine * right_imag - left_sine * left_real;
        } else {
            real[left] = cosine * left_real + right_sine * right_real;
            imag[left] = cosine * left_imag + right_sine * right_imag;
            real[right] = cosine * right_real + left_sine * left_real;
            imag[right] = cosine * right_imag + left_sine * left_imag;
        }
    }
    lane.sync();
}

template <typename Coefficient>
__device__ void apply_promotion(const Action& action, bool sign, Coefficient* real,
                                Coefficient* imag, Lane lane) {
    const uint64_t old_size = uint64_t{1} << action.active_before;
    const Coefficient cosine = static_cast<Coefficient>(action.value0);
    const Coefficient sine =
        sign ? -static_cast<Coefficient>(action.value1) : static_cast<Coefficient>(action.value1);
    for (uint64_t basis = lane.lane; basis < old_size; basis += lane.lanes) {
        const Coefficient input_real = real[basis];
        const Coefficient input_imag = imag[basis];
        real[basis] = cosine * input_real;
        imag[basis] = cosine * input_imag;
        real[old_size + basis] = sine * input_imag;
        imag[old_size + basis] = -sine * input_real;
    }
    lane.sync();
}

template <typename Coefficient, bool Replay>
__device__ bool measure_active(const ProgramView& program, const Action& action,
                               Xoshiro256PlusPlus& rng, const uint8_t* forced_records,
                               double* log_probability, uint8_t* symbols, uint8_t* records,
                               Coefficient* real, Coefficient* imag, Coefficient* scratch_real,
                               Coefficient* scratch_imag, Lane lane) {
    const uint64_t size = uint64_t{1} << action.active_before;
    const uint64_t output_size = size >> 1;
    double probability_zero = 0.0;
    double probability_one = 0.0;
    if (action.x == 0) {
        for (uint64_t basis = lane.lane; basis < size; basis += lane.lanes) {
            const double probability = norm(load(real, imag, basis));
            if (parity(basis & action.z)) {
                probability_one += probability;
            } else {
                probability_zero += probability;
            }
        }
    } else {
        for (uint64_t packed = lane.lane; packed < output_size; packed += lane.lanes) {
            probability_zero += norm(compact_nondiagonal(action, real, imag, packed, false));
            probability_one += norm(compact_nondiagonal(action, real, imag, packed, true));
        }
    }
    lane.reduce_pair(probability_zero, probability_one);

    const double total = probability_zero + probability_one;
    const double epsilon = clifft::kMeasurementDustEpsilon * total;
    bool branch;
    if constexpr (Replay) {
        const bool correction = evaluate(program, symbols, action.expression);
        branch = (forced_records[action.index1] != 0) != correction;
        if ((probability_one <= epsilon && branch) || (probability_zero <= epsilon && !branch)) {
            return false;
        }
        if (probability_zero > epsilon && probability_one > epsilon) {
            const double selected_probability = branch ? probability_one : probability_zero;
            *log_probability += log(selected_probability / total);
        }
    } else {
        if (probability_one <= epsilon) {
            branch = false;
        } else if (probability_zero <= epsilon) {
            branch = true;
        } else {
            branch = rng.next_double() * total >= probability_zero;
        }
    }
    const double selected_probability = branch ? probability_one : probability_zero;
    const double inverse_norm = 1.0 / sqrt(selected_probability);
    if (lane.is_writer()) {
        symbols[action.index0] = static_cast<uint8_t>(branch);
    }
    lane.sync();
    if (lane.is_writer()) {
        records[action.index1] = static_cast<uint8_t>(evaluate(program, symbols, action.expression));
    }

    // Both collapse shapes stage into scratch: lanes write packed outputs
    // while other lanes still read overlapping source indices, so in-place
    // compaction is only safe for the sequential tier.
    if (action.x == 0) {
        for (uint64_t packed = lane.lane; packed < output_size; packed += lane.lanes) {
            const uint64_t without_pivot = insert_zero_bit(packed, action.index2);
            const bool other_parity =
                parity(without_pivot & action.pair_stride_or_z_without_pivot);
            const bool pivot_value = branch != other_parity;
            const uint64_t source =
                without_pivot | (static_cast<uint64_t>(pivot_value) << action.index2);
            const ComplexValue<Coefficient> value = load(real, imag, source);
            store(scratch_real, scratch_imag, packed, normalize(value, inverse_norm));
        }
    } else {
        for (uint64_t packed = lane.lane; packed < output_size; packed += lane.lanes) {
            const ComplexValue<Coefficient> value =
                compact_nondiagonal(action, real, imag, packed, branch);
            store(scratch_real, scratch_imag, packed, normalize(value, inverse_norm));
        }
    }
    lane.sync();
    for (uint64_t packed = lane.lane; packed < output_size; packed += lane.lanes) {
        real[packed] = scratch_real[packed];
        imag[packed] = scratch_imag[packed];
    }
    lane.sync();
    return true;
}

template <typename Coefficient>
__device__ double expectation_value(const Action& action, const Coefficient* real,
                                    const Coefficient* imag, Lane lane) {
    if ((action.flags & kAbsentActiveProjection) != 0) {
        return 0.0;
    }
    if (action.x == 0 && action.z == 0) {
        return 1.0;
    }
    const uint64_t size = uint64_t{1} << action.active_before;
    double result = 0.0;
    double unused = 0.0;
    if (action.x == 0) {
        for (uint64_t basis = lane.lane; basis < size; basis += lane.lanes) {
            const double eigenvalue = parity(basis & action.z) ? -1.0 : 1.0;
            result += eigenvalue * norm(load(real, imag, basis));
        }
        lane.reduce_pair(result, unused);
        return result;
    }
    for (uint64_t basis = lane.lane; basis < size; basis += lane.lanes) {
        const ComplexValue<Coefficient> paired_coefficient = load(real, imag, basis ^ action.x);
        const ComplexValue<Coefficient> basis_coefficient = load(real, imag, basis);
        const ComplexValue<double> paired{static_cast<double>(paired_coefficient.real),
                                          static_cast<double>(paired_coefficient.imag)};
        const ComplexValue<double> basis_value{static_cast<double>(basis_coefficient.real),
                                               static_cast<double>(basis_coefficient.imag)};
        const ComplexValue<double> phased = multiply(phase_at<double>(action, basis), basis_value);
        result += paired.real * phased.real + paired.imag * phased.imag;
    }
    lane.reduce_pair(result, unused);
    return result;
}

// ---------------------------------------------------------------------------
// One shot's interpreter loop, shared by every tier.
// ---------------------------------------------------------------------------

template <typename Coefficient, bool Replay>
__device__ void interpret_one_shot(const ProgramView& program, SeedRoot seed_root, uint32_t shot,
                                   Coefficient* real, Coefficient* imag,
                                   Coefficient* scratch_real, Coefficient* scratch_imag,
                                   uint8_t* symbols, uint8_t* records,
                                   const uint8_t* forced_records, uint8_t* detectors,
                                   uint8_t* observables, double* exp_vals,
                                   double* log_probability_storage, uint8_t* reachable_storage,
                                   uint8_t* survived, Lane lane) {
    for (uint64_t basis = lane.lane; basis < (uint64_t{1} << program.initial_active_width);
         basis += lane.lanes) {
        real[basis] = 0;
        imag[basis] = 0;
    }
    if (lane.is_writer()) {
        real[0] = static_cast<Coefficient>(1);
        for (uint32_t symbol = 0; symbol < program.num_symbols; ++symbol) {
            symbols[symbol] = 0;
        }
        for (uint32_t record = 0; record < program.num_records; ++record) {
            records[record] = 0;
        }
        for (uint32_t detector = 0; detector < program.num_detectors; ++detector) {
            detectors[detector] = 0;
        }
        for (uint32_t observable = 0; observable < program.num_observables; ++observable) {
            observables[observable] = 0;
        }
        for (uint32_t exp_val = 0; exp_val < program.num_exp_vals; ++exp_val) {
            exp_vals[exp_val] = 0.0;
        }
    }
    lane.sync();

    // Every lane advances an identical RNG from identical inputs, so scalar
    // decisions are uniform across the block without broadcasts.
    Xoshiro256PlusPlus rng = shot_rng(seed_root, shot);
    if constexpr (!Replay) {
        sample_noise(program, rng, symbols, lane.is_writer());
        lane.sync();
    }
    double log_probability = 0.0;
    bool reachable = true;
    bool discarded = false;
    for (uint32_t action_index = 0; action_index < program.action_count; ++action_index) {
        const Action action = program.actions[action_index];
        switch (action.tag) {
            case ActionTag::RotateActivePauli:
                apply_rotation(action, evaluate(program, symbols, action.expression), real, imag,
                               lane);
                break;
            case ActionTag::PromoteDormantRotation:
                apply_promotion(action, evaluate(program, symbols, action.expression), real, imag,
                                lane);
                break;
            case ActionTag::MeasureActivePauli:
                reachable = measure_active<Coefficient, Replay>(
                    program, action, rng, forced_records, &log_probability, symbols, records, real,
                    imag, scratch_real, scratch_imag, lane);
                break;
            case ActionTag::MeasureDormantRandom: {
                const bool correction = evaluate(program, symbols, action.expression);
                const bool branch = Replay ? (forced_records[action.index1] != 0) != correction
                                           : rng.next_double() >= 0.5;
                if constexpr (Replay) {
                    log_probability += kLogHalf;
                }
                if (lane.is_writer()) {
                    symbols[action.index0] = static_cast<uint8_t>(branch);
                }
                lane.sync();
                if (lane.is_writer()) {
                    records[action.index1] =
                        static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                }
                lane.sync();
                break;
            }
            case ActionTag::RecordClassical: {
                const bool outcome = evaluate(program, symbols, action.expression);
                if (lane.is_writer()) {
                    records[action.index0] = static_cast<uint8_t>(outcome);
                }
                if constexpr (Replay) {
                    reachable = static_cast<uint8_t>(outcome) == forced_records[action.index0];
                }
                lane.sync();
                break;
            }
            case ActionTag::DefineSymbol:
                if (lane.is_writer()) {
                    symbols[action.index0] =
                        static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                }
                lane.sync();
                break;
            case ActionTag::ApplyReadoutNoise: {
                if constexpr (Replay) {
                    // A post-readout record does not identify the sampled flip,
                    // so record-only replay cannot reconstruct this path.
                    reachable = false;
                } else {
                    const bool source = evaluate(program, symbols, action.expression);
                    const double probability = source ? action.value1 : action.value0;
                    const bool flip = probability >= 1.0 ||
                                      (probability > 0.0 && rng.next_double() < probability);
                    if (lane.is_writer()) {
                        symbols[action.index0] = static_cast<uint8_t>(flip);
                        records[action.index1] ^= static_cast<uint8_t>(flip);
                    }
                    lane.sync();
                }
                break;
            }
            case ActionTag::WriteDetector: {
                const bool outcome = evaluate(program, symbols, action.expression);
                if (lane.is_writer()) {
                    detectors[action.index0] = static_cast<uint8_t>(outcome);
                }
                if ((action.flags & kPostselected) != 0 && outcome) {
                    discarded = true;
                }
                lane.sync();
                break;
            }
            case ActionTag::WriteObservable:
                if (lane.is_writer()) {
                    observables[action.index0] =
                        static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                }
                lane.sync();
                break;
            case ActionTag::WriteExpectationValue: {
                const double value = expectation_value(action, real, imag, lane);
                if (lane.is_writer()) {
                    exp_vals[action.index0] =
                        evaluate(program, symbols, action.expression) ? -value : value;
                }
                lane.sync();
                break;
            }
        }
        if (!reachable || discarded) {
            break;
        }
    }
    if (lane.is_writer()) {
        if constexpr (Replay) {
            log_probability_storage[shot] = log_probability;
            reachable_storage[shot] = static_cast<uint8_t>(reachable);
        }
        survived[shot] = static_cast<uint8_t>(reachable && !discarded);
    }
    lane.sync();
}

// ---------------------------------------------------------------------------
// Tier kernels.
// ---------------------------------------------------------------------------

template <typename Coefficient, bool Replay>
__global__ void interpret_shots_thread(ProgramView program, SeedRoot seed_root, uint32_t shots,
                                       Coefficient* coefficient_storage, uint8_t* symbol_storage,
                                       uint8_t* record_storage,
                                       const uint8_t* forced_record_storage,
                                       uint8_t* detector_storage, uint8_t* observable_storage,
                                       double* exp_val_storage, double* log_probability_storage,
                                       uint8_t* reachable_storage, uint8_t* survived) {
    const uint32_t shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) {
        return;
    }
    const uint64_t capacity = uint64_t{1} << program.peak_active_width;
    const uint64_t scratch_capacity = capacity > 1 ? capacity >> 1 : 1;
    const uint64_t coefficient_stride = 2 * capacity + 2 * scratch_capacity;
    Coefficient* real = coefficient_storage + static_cast<uint64_t>(shot) * coefficient_stride;
    Coefficient* imag = real + capacity;
    Coefficient* scratch_real = imag + capacity;
    Coefficient* scratch_imag = scratch_real + scratch_capacity;
    uint8_t* symbols = program.num_symbols == 0
                           ? nullptr
                           : symbol_storage + static_cast<uint64_t>(shot) * program.num_symbols;
    uint8_t* records = program.num_records == 0
                           ? nullptr
                           : record_storage + static_cast<uint64_t>(shot) * program.num_records;
    uint8_t* detectors =
        program.num_detectors == 0
            ? nullptr
            : detector_storage + static_cast<uint64_t>(shot) * program.num_detectors;
    uint8_t* observables =
        program.num_observables == 0
            ? nullptr
            : observable_storage + static_cast<uint64_t>(shot) * program.num_observables;
    double* exp_vals = program.num_exp_vals == 0
                           ? nullptr
                           : exp_val_storage + static_cast<uint64_t>(shot) * program.num_exp_vals;
    const uint8_t* forced_records = nullptr;
    if constexpr (Replay) {
        if (program.num_records != 0) {
            forced_records =
                forced_record_storage + static_cast<uint64_t>(shot) * program.num_records;
        }
    }
    interpret_one_shot<Coefficient, Replay>(program, seed_root, shot, real, imag, scratch_real,
                                            scratch_imag, symbols, records, forced_records,
                                            detectors, observables, exp_vals,
                                            log_probability_storage, reachable_storage, survived,
                                            Lane{});
}

// Block-per-shot kernel. UseShared selects the coefficient residence: extern
// shared memory (carved past the reduction scratch) or a per-block global
// slab. Each block walks shots strided by the grid so global slabs bound
// memory by grid size rather than shot count.
template <typename Coefficient, bool Replay, bool UseShared>
__global__ void interpret_shots_block(ProgramView program, SeedRoot seed_root, uint32_t shots,
                                      Coefficient* slab_storage, uint8_t* symbol_storage,
                                      uint8_t* record_storage,
                                      const uint8_t* forced_record_storage,
                                      uint8_t* detector_storage, uint8_t* observable_storage,
                                      double* exp_val_storage, double* log_probability_storage,
                                      uint8_t* reachable_storage, uint8_t* survived) {
    extern __shared__ unsigned char dynamic_shared[];
    double* reduce = reinterpret_cast<double*>(dynamic_shared);

    const uint64_t capacity = uint64_t{1} << program.peak_active_width;
    const uint64_t scratch_capacity = capacity > 1 ? capacity >> 1 : 1;
    const uint64_t coefficient_stride = 2 * capacity + 2 * scratch_capacity;

    Coefficient* real;
    if constexpr (UseShared) {
        real = reinterpret_cast<Coefficient*>(dynamic_shared +
                                              2 * blockDim.x * sizeof(double));
    } else {
        real = slab_storage + static_cast<uint64_t>(blockIdx.x) * coefficient_stride;
    }
    Coefficient* imag = real + capacity;
    Coefficient* scratch_real = imag + capacity;
    Coefficient* scratch_imag = scratch_real + scratch_capacity;

    Lane lane{threadIdx.x, blockDim.x, reduce};

    for (uint32_t shot = blockIdx.x; shot < shots; shot += gridDim.x) {
        uint8_t* symbols =
            program.num_symbols == 0
                ? nullptr
                : symbol_storage + static_cast<uint64_t>(shot) * program.num_symbols;
        uint8_t* records =
            program.num_records == 0
                ? nullptr
                : record_storage + static_cast<uint64_t>(shot) * program.num_records;
        uint8_t* detectors =
            program.num_detectors == 0
                ? nullptr
                : detector_storage + static_cast<uint64_t>(shot) * program.num_detectors;
        uint8_t* observables =
            program.num_observables == 0
                ? nullptr
                : observable_storage + static_cast<uint64_t>(shot) * program.num_observables;
        double* exp_vals =
            program.num_exp_vals == 0
                ? nullptr
                : exp_val_storage + static_cast<uint64_t>(shot) * program.num_exp_vals;
        const uint8_t* forced_records = nullptr;
        if constexpr (Replay) {
            if (program.num_records != 0) {
                forced_records =
                    forced_record_storage + static_cast<uint64_t>(shot) * program.num_records;
            }
        }
        interpret_one_shot<Coefficient, Replay>(program, seed_root, shot, real, imag, scratch_real,
                                                scratch_imag, symbols, records, forced_records,
                                                detectors, observables, exp_vals,
                                                log_probability_storage, reachable_storage,
                                                survived, lane);
    }
}

}  // namespace

}  // namespace clifft::sampling::cuda::detail

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

#include "clifft/sampling/cuda/executable.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft::sampling::cuda {

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + operation +
                                 " failed: " + cudaGetErrorString(error));
    }
}

size_t checked_elements(uint64_t rows, uint64_t stride, const char* storage) {
    if (stride != 0 && rows > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error(std::string("CUDA ") + storage + " allocation exceeds size_t");
    }
    return static_cast<size_t>(rows * stride);
}

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count != 0) {
            if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
                throw std::length_error("CUDA allocation byte size exceeds size_t");
            }
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                       "allocation");
        }
    }

    explicit DeviceBuffer(std::span<const T> source) : DeviceBuffer(source.size()) {
        if (!source.empty()) {
            check_cuda(
                cudaMemcpy(data_, source.data(), source.size_bytes(), cudaMemcpyHostToDevice),
                "program upload");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                (void)cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }

    void download(std::span<T> destination) const {
        if (destination.size() != count_) {
            throw std::invalid_argument("CUDA download destination has the wrong size");
        }
        if (!destination.empty()) {
            check_cuda(cudaMemcpy(destination.data(), data_, destination.size_bytes(),
                                  cudaMemcpyDeviceToHost),
                       "result download");
        }
    }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct UploadedProgram {
    explicit UploadedProgram(const Executable& executable)
        : actions(executable.actions()),
          expressions(executable.expressions()),
          expression_terms(executable.expression_terms()),
          noise_sites(executable.noise_sites()),
          noise_outcomes(executable.noise_outcomes()) {
        view.actions = actions.data();
        view.expressions = expressions.data();
        view.expression_terms = expression_terms.data();
        view.noise_sites = noise_sites.data();
        view.noise_outcomes = noise_outcomes.data();
        view.action_count = static_cast<uint32_t>(executable.actions().size());
        view.initial_active_width = executable.initial_active_width();
        view.peak_active_width = executable.peak_active_width();
        view.num_symbols = executable.num_symbols();
        view.num_records = executable.num_records();
        view.num_visible_records = executable.num_visible_records();
        view.num_detectors = executable.num_detectors();
        view.num_observables = executable.num_observables();
        view.num_exp_vals = executable.num_exp_vals();
        view.noise_site_count = static_cast<uint32_t>(executable.noise_sites().size());
    }

    DeviceBuffer<detail::Action> actions;
    DeviceBuffer<detail::Expression> expressions;
    DeviceBuffer<uint32_t> expression_terms;
    DeviceBuffer<detail::NoiseSite> noise_sites;
    DeviceBuffer<detail::NoiseOutcome> noise_outcomes;
    detail::ProgramView view;
};

struct Rows {
    std::vector<uint8_t> records;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
    std::vector<double> log_probabilities;
    std::vector<uint8_t> reachable;
    std::vector<uint8_t> survived;
};

struct DeviceLimits {
    int device = 0;
    int multiprocessors = 0;
    size_t shared_optin = 0;
    size_t free_memory = 0;
};

DeviceLimits query_device() {
    DeviceLimits limits;
    check_cuda(cudaGetDevice(&limits.device), "device query");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, limits.device), "device properties");
    limits.multiprocessors = properties.multiProcessorCount;
    limits.shared_optin = properties.sharedMemPerBlockOptin;
    size_t total = 0;
    check_cuda(cudaMemGetInfo(&limits.free_memory, &total), "memory query");
    return limits;
}

uint64_t coefficient_stride_for(const Executable& executable) {
    const uint64_t capacity = uint64_t{1} << executable.peak_active_width();
    const uint64_t scratch_capacity = capacity > 1 ? capacity >> 1 : 1;
    return 2 * capacity + 2 * scratch_capacity;
}

size_t coefficient_bytes(const Executable& executable, CoefficientPrecision precision) {
    const size_t element =
        precision == CoefficientPrecision::FP32 ? sizeof(float) : sizeof(double);
    return static_cast<size_t>(coefficient_stride_for(executable)) * element;
}

size_t shared_bytes_needed(const Executable& executable, const SamplingOptions& options) {
    return 2 * static_cast<size_t>(options.block_size) * sizeof(double) +
           coefficient_bytes(executable, options.coefficient_precision);
}

ExecutionTier resolve_tier(const Executable& executable, const SamplingOptions& options,
                           const DeviceLimits& limits) {
    if (options.tier != ExecutionTier::Auto) {
        if (options.tier == ExecutionTier::BlockShared &&
            shared_bytes_needed(executable, options) > limits.shared_optin) {
            throw std::invalid_argument(
                "CUDA BlockShared tier does not fit this device's shared memory");
        }
        return options.tier;
    }
    if (executable.peak_active_width() <= kThreadPerShotMaxActiveWidth) {
        return ExecutionTier::ThreadPerShot;
    }
    if (shared_bytes_needed(executable, options) <= limits.shared_optin) {
        return ExecutionTier::BlockShared;
    }
    return ExecutionTier::BlockGlobal;
}

uint32_t resolve_concurrency(const Executable& executable, const SamplingOptions& options,
                             const DeviceLimits& limits, ExecutionTier tier, uint32_t shots) {
    if (options.max_concurrent_shots != 0) {
        return std::min(shots, options.max_concurrent_shots);
    }
    // Enough blocks to oversubscribe every SM regardless of occupancy; more
    // buys no parallelism and, for the global tier, multiplies slab memory.
    const uint64_t target =
        std::max<uint64_t>(static_cast<uint64_t>(limits.multiprocessors) * 32, 1);
    if (tier == ExecutionTier::BlockShared) {
        return static_cast<uint32_t>(std::min<uint64_t>(shots, target));
    }
    // Leave real headroom: huge slab allocations that fit free memory can
    // still starve the runtime's launch-time allocations.
    const size_t slab = coefficient_bytes(executable, options.coefficient_precision);
    const size_t reserve = std::max<size_t>(limits.free_memory / 10, size_t{2} << 30);
    const size_t budget = limits.free_memory > reserve ? limits.free_memory - reserve : 0;
    const uint64_t fit = slab == 0 ? shots : std::max<size_t>(budget / slab, 1);
    return static_cast<uint32_t>(std::min<uint64_t>(shots, std::min<uint64_t>(fit, target)));
}

template <typename Coefficient, bool Replay>
void launch_tier(ExecutionTier tier, uint32_t grid, uint32_t block, size_t dynamic_shared,
                 const detail::ProgramView& view, const SeedRoot& root, uint32_t shots,
                 Coefficient* slab, uint8_t* symbols, uint8_t* records,
                 const uint8_t* forced_records, uint8_t* detectors, uint8_t* observables,
                 double* exp_vals, double* log_probabilities, uint8_t* reachable,
                 uint8_t* survived) {
    const detail::SeedRoot device_root{{root.w[0], root.w[1], root.w[2], root.w[3]}};
    switch (tier) {
        case ExecutionTier::ThreadPerShot: {
            const uint32_t blocks = (shots + block - 1) / block;
            detail::interpret_shots_thread<Coefficient, Replay><<<blocks, block>>>(
                view, device_root, shots, slab, symbols, records, forced_records, detectors,
                observables, exp_vals, log_probabilities, reachable, survived);
            break;
        }
        case ExecutionTier::BlockShared: {
            auto* kernel = detail::interpret_shots_block<Coefficient, Replay, true>;
            check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(dynamic_shared)),
                       "shared memory opt-in");
            kernel<<<grid, block, dynamic_shared>>>(view, device_root, shots, slab, symbols,
                                                    records, forced_records, detectors,
                                                    observables, exp_vals, log_probabilities,
                                                    reachable, survived);
            break;
        }
        case ExecutionTier::BlockGlobal: {
            detail::interpret_shots_block<Coefficient, Replay, false>
                <<<grid, block, dynamic_shared>>>(view, device_root, shots, slab, symbols,
                                                  records, forced_records, detectors, observables,
                                                  exp_vals, log_probabilities, reachable,
                                                  survived);
            break;
        }
        case ExecutionTier::Auto:
            throw std::logic_error("CUDA tier must be resolved before launch");
    }
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");
}

template <typename Coefficient, bool Replay>
Rows run_device(const Executable& executable, uint32_t shots, const SamplingOptions& options,
                std::span<const uint8_t> forced_record_input = {}) {
    const DeviceLimits limits = query_device();
    const ExecutionTier tier = resolve_tier(executable, options, limits);
    UploadedProgram program(executable);

    const uint64_t coefficient_stride = coefficient_stride_for(executable);
    uint64_t slab_rows = 0;
    uint32_t grid = 1;
    uint32_t block = options.block_size;
    size_t dynamic_shared = 0;
    switch (tier) {
        case ExecutionTier::ThreadPerShot:
            slab_rows = shots;
            break;
        case ExecutionTier::BlockShared:
            grid = resolve_concurrency(executable, options, limits, tier, shots);
            dynamic_shared = shared_bytes_needed(executable, options);
            break;
        case ExecutionTier::BlockGlobal:
            grid = resolve_concurrency(executable, options, limits, tier, shots);
            slab_rows = grid;
            dynamic_shared = 2 * static_cast<size_t>(block) * sizeof(double);
            break;
        case ExecutionTier::Auto:
            throw std::logic_error("CUDA tier must be resolved before allocation");
    }

    DeviceBuffer<Coefficient> coefficients(
        checked_elements(slab_rows, coefficient_stride, "coefficient"));
    DeviceBuffer<uint8_t> symbols(checked_elements(shots, executable.num_symbols(), "symbol"));
    DeviceBuffer<uint8_t> records(checked_elements(shots, executable.num_records(), "record"));
    DeviceBuffer<uint8_t> forced_records(forced_record_input);
    DeviceBuffer<uint8_t> detectors(
        checked_elements(shots, executable.num_detectors(), "detector"));
    DeviceBuffer<uint8_t> observables(
        checked_elements(shots, executable.num_observables(), "observable"));
    DeviceBuffer<double> exp_vals(
        checked_elements(shots, executable.num_exp_vals(), "expectation"));
    DeviceBuffer<double> log_probabilities(Replay ? shots : 0);
    DeviceBuffer<uint8_t> reachable(Replay ? shots : 0);
    DeviceBuffer<uint8_t> survived(shots);

    const SeedRoot root = Replay ? SeedRoot{} : make_seed_root(shots, options.seed);
    launch_tier<Coefficient, Replay>(tier, grid, block, dynamic_shared, program.view, root, shots,
                                     coefficients.data(), symbols.data(), records.data(),
                                     forced_records.data(), detectors.data(), observables.data(),
                                     exp_vals.data(), log_probabilities.data(), reachable.data(),
                                     survived.data());

    Rows rows;
    rows.records.resize(checked_elements(shots, executable.num_records(), "record result"));
    rows.detectors.resize(checked_elements(shots, executable.num_detectors(), "detector result"));
    rows.observables.resize(
        checked_elements(shots, executable.num_observables(), "observable result"));
    rows.exp_vals.resize(checked_elements(shots, executable.num_exp_vals(), "expectation result"));
    if constexpr (Replay) {
        rows.log_probabilities.resize(shots);
        rows.reachable.resize(shots);
    }
    rows.survived.resize(shots);
    records.download(rows.records);
    detectors.download(rows.detectors);
    observables.download(rows.observables);
    exp_vals.download(rows.exp_vals);
    if constexpr (Replay) {
        log_probabilities.download(rows.log_probabilities);
        reachable.download(rows.reachable);
    }
    survived.download(rows.survived);
    return rows;
}

Rows execute(const Executable& executable, uint32_t shots, const SamplingOptions& options) {
    if (options.coefficient_precision == CoefficientPrecision::FP32) {
        return run_device<float, false>(executable, shots, options);
    }
    return run_device<double, false>(executable, shots, options);
}

Rows execute_replay(const Executable& executable, std::span<const uint8_t> forced_records,
                    CoefficientPrecision coefficient_precision) {
    SamplingOptions options;
    options.coefficient_precision = coefficient_precision;
    options.block_size = 64;
    if (coefficient_precision == CoefficientPrecision::FP32) {
        return run_device<float, true>(executable, 1, options, forced_records);
    }
    return run_device<double, true>(executable, 1, options, forced_records);
}

void validate_options(const SamplingOptions& options) {
    if (options.block_size == 0 || options.block_size > 1024 ||
        (options.block_size & (options.block_size - 1)) != 0) {
        throw std::invalid_argument("CUDA block size must be a power of two between 1 and 1024");
    }
}

template <typename T>
void compact_rows(std::vector<T>& values, std::span<const uint8_t> survived, size_t stride,
                  uint32_t passed_shots) {
    size_t destination = 0;
    for (size_t shot = 0; shot < survived.size(); ++shot) {
        if (survived[shot] == 0) {
            continue;
        }
        if (destination != shot && stride != 0) {
            std::copy_n(values.begin() + shot * stride, stride,
                        values.begin() + destination * stride);
        }
        ++destination;
    }
    values.resize(static_cast<size_t>(passed_shots) * stride);
}

}  // namespace

bool is_available() noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::string backend_info() {
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error != cudaSuccess) {
        return std::string("CUDA runtime unavailable: ") + cudaGetErrorString(count_error);
    }
    std::ostringstream output;
    output << "CUDA devices: " << device_count;
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
            output << "\n[" << device << "] " << properties.name << " sm_"
                   << properties.major << properties.minor << " smem_optin="
                   << properties.sharedMemPerBlockOptin;
        }
    }
    return output.str();
}

ExecutionTier selected_tier(const Executable& executable, const SamplingOptions& options) {
    return resolve_tier(executable, options, query_device());
}

SamplingResult sample(const Executable& executable, uint32_t shots,
                      const SamplingOptions& options) {
    validate_options(options);
    if (executable.has_postselection()) {
        throw std::invalid_argument(
            "CUDA fixed-row sampling does not support postselection; use sample_survivors");
    }
    SamplingResult result;
    if (shots == 0) {
        return result;
    }
    Rows rows = execute(executable, shots, options);
    result.measurements.resize(
        checked_elements(shots, executable.num_visible_records(), "measurement result"));
    for (uint32_t shot = 0; shot < shots; ++shot) {
        std::copy_n(rows.records.begin() + static_cast<size_t>(shot) * executable.num_records(),
                    executable.num_visible_records(),
                    result.measurements.begin() +
                        static_cast<size_t>(shot) * executable.num_visible_records());
    }
    result.detectors = std::move(rows.detectors);
    result.observables = std::move(rows.observables);
    result.exp_vals = std::move(rows.exp_vals);
    return result;
}

SamplingSurvivorResult sample_survivors(const Executable& executable, uint32_t shots,
                                        bool keep_records, const SamplingOptions& options) {
    validate_options(options);
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(executable.num_observables(), 0);
    Rows rows = execute(executable, shots, options);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (rows.survived[shot] == 0) {
            continue;
        }
        ++result.passed_shots;
        bool logical_error = false;
        for (uint32_t observable = 0; observable < executable.num_observables(); ++observable) {
            const bool value =
                rows.observables[static_cast<size_t>(shot) * executable.num_observables() +
                                 observable] != 0;
            result.observable_ones[observable] += static_cast<uint64_t>(value);
            logical_error |= value;
        }
        result.logical_errors += static_cast<uint32_t>(logical_error);
    }
    if (!keep_records) {
        return result;
    }

    result.measurements.resize(
        checked_elements(shots, executable.num_visible_records(), "measurement result"));
    for (uint32_t shot = 0; shot < shots; ++shot) {
        std::copy_n(rows.records.begin() + static_cast<size_t>(shot) * executable.num_records(),
                    executable.num_visible_records(),
                    result.measurements.begin() +
                        static_cast<size_t>(shot) * executable.num_visible_records());
    }
    compact_rows(result.measurements, rows.survived, executable.num_visible_records(),
                 result.passed_shots);
    compact_rows(rows.detectors, rows.survived, executable.num_detectors(), result.passed_shots);
    compact_rows(rows.observables, rows.survived, executable.num_observables(),
                 result.passed_shots);
    compact_rows(rows.exp_vals, rows.survived, executable.num_exp_vals(), result.passed_shots);
    result.detectors = std::move(rows.detectors);
    result.observables = std::move(rows.observables);
    result.exp_vals = std::move(rows.exp_vals);
    return result;
}

ReplayResult replay_shot(const Executable& executable, std::span<const uint8_t> forced_records,
                         CoefficientPrecision coefficient_precision) {
    if (forced_records.size() != executable.num_records()) {
        throw std::invalid_argument(
            "CUDA replay requires one forced value for every visible and hidden record");
    }
    if (!std::all_of(forced_records.begin(), forced_records.end(),
                     [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("CUDA replay record bytes must be Boolean");
    }
    if (!executable.noise_sites().empty()) {
        throw std::invalid_argument("CUDA replay does not support presampled noise");
    }
    Rows rows = execute_replay(executable, forced_records, coefficient_precision);
    ReplayResult result;
    result.reachable = rows.reachable[0] != 0;
    result.survived = rows.survived[0] != 0;
    result.log_probability = rows.log_probabilities[0];
    if (!result.reachable || !result.survived) {
        return result;
    }
    result.outputs.measurements.assign(rows.records.begin(),
                                       rows.records.begin() + executable.num_visible_records());
    result.outputs.detectors = std::move(rows.detectors);
    result.outputs.observables = std::move(rows.observables);
    result.outputs.exp_vals = std::move(rows.exp_vals);
    return result;
}

}  // namespace clifft::sampling::cuda
