// CUDA device interpreter for lowered sampling plans.
//
// Three execution tiers share one action semantics with the HIP backend, so
// conformance results transfer between them:
//   * thread-per-shot: one thread walks a whole shot; sensible only while the
//     coefficient state is tiny.
//   * block-per-shot, shared: one block walks a whole shot with the
//     coefficients resident in opt-in shared memory, so a shot costs one
//     global load and store instead of one sweep per action.
//   * block-per-shot, global: the same cooperative kernel with the state in a
//     per-block global-memory slab, for widths past the shared budget.
//
// Scalar control flow (RNG draws, branch selection, frame actions) runs
// redundantly on every thread of a block from identical inputs, so branch
// decisions need no broadcast storage; byte outputs are written by lane 0.

#include "clifft/sampling/cuda/device_program.h"
#include "clifft/util/numeric.h"
#include "clifft/util/shot_seed_domains.h"

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
    for (uint64_t word = 0; word < 4; ++word) {
        const uint64_t stream =
            ((clifft::kCudaSamplingExecutorDomain << 2) | word) * 0xBF58476D1CE4E5B9ULL;
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

__device__ __forceinline__ bool evaluate_output_value(const ProgramView& program,
                                                      const uint8_t* symbols,
                                                      const uint8_t* records,
                                                      const Action& action) {
    const uint8_t* values = (action.flags & kRecordParity) != 0 ? records : symbols;
    return evaluate(program, values, action.expression);
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

// Every lane draws the same noise so the RNG stays aligned across the block;
// only the writer lane records the sampled symbols.
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

// Describes this thread's slice of one shot. For the thread-per-shot tier the
// slice is the whole shot (lane 0 of 1), so the strided loops below degenerate
// to the sequential bodies and every tier shares one implementation of each
// action.
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
    // Pair p owns the basis index with a zero inserted at the pairing bit, so
    // lanes stride over pairs without ever touching the same element twice.
    const uint64_t low_mask = action.pair_stride - 1;
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
        records[action.index1] =
            static_cast<uint8_t>(evaluate(program, symbols, action.expression));
    }

    // Both collapse shapes stage through scratch: lanes write packed outputs
    // while other lanes still read overlapping source indices, so in-place
    // compaction is only safe for a single sequential lane.
    if (action.x == 0) {
        for (uint64_t packed = lane.lane; packed < output_size; packed += lane.lanes) {
            const uint64_t without_pivot = insert_zero_bit(packed, action.index2);
            // without_pivot already clears the measured pivot, so the full Z
            // mask has the same parity as its pivot-cleared form.
            const bool other_parity = parity(without_pivot & action.z);
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

// Per-shot byte and value rows. Every tier addresses them by the shot's local
// index within the launch; only the RNG sees the global shot index.
struct ShotRows {
    uint8_t* symbols = nullptr;
    uint8_t* records = nullptr;
    const uint8_t* forced_records = nullptr;
    uint8_t* detectors = nullptr;
    uint8_t* observables = nullptr;
    double* exp_vals = nullptr;
};

template <bool Replay>
__device__ __forceinline__ ShotRows shot_rows(const ProgramView& program, uint32_t shot,
                                              uint8_t* symbol_storage, uint8_t* record_storage,
                                              const uint8_t* forced_record_storage,
                                              uint8_t* detector_storage,
                                              uint8_t* observable_storage,
                                              double* exp_val_storage) {
    ShotRows rows;
    const uint64_t index = shot;
    rows.symbols = program.num_symbols == 0 ? nullptr : symbol_storage + index * program.num_symbols;
    rows.records = program.num_records == 0 ? nullptr : record_storage + index * program.num_records;
    rows.detectors =
        program.num_detectors == 0 ? nullptr : detector_storage + index * program.num_detectors;
    rows.observables = program.num_observables == 0
                           ? nullptr
                           : observable_storage + index * program.num_observables;
    rows.exp_vals =
        program.num_exp_vals == 0 ? nullptr : exp_val_storage + index * program.num_exp_vals;
    if constexpr (Replay) {
        if (program.num_records != 0) {
            rows.forced_records = forced_record_storage + index * program.num_records;
        }
    }
    return rows;
}

// One shot's interpreter loop, shared by every tier.
template <typename Coefficient, bool Replay>
__device__ void interpret_one_shot(const ProgramView& program, SeedRoot seed_root,
                                   uint64_t global_shot, uint32_t shot, Coefficient* real,
                                   Coefficient* imag, Coefficient* scratch_real,
                                   Coefficient* scratch_imag, ShotRows rows,
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
            rows.symbols[symbol] = 0;
        }
        for (uint32_t record = 0; record < program.num_records; ++record) {
            rows.records[record] = 0;
        }
        for (uint32_t detector = 0; detector < program.num_detectors; ++detector) {
            rows.detectors[detector] = 0;
        }
        for (uint32_t observable = 0; observable < program.num_observables; ++observable) {
            rows.observables[observable] = 0;
        }
        for (uint32_t exp_val = 0; exp_val < program.num_exp_vals; ++exp_val) {
            rows.exp_vals[exp_val] = 0.0;
        }
    }
    lane.sync();

    Xoshiro256PlusPlus rng = shot_rng(seed_root, global_shot);
    if constexpr (!Replay) {
        sample_noise(program, rng, rows.symbols, lane.is_writer());
        lane.sync();
    }
    double log_probability = 0.0;
    bool reachable = true;
    bool discarded = false;
    for (uint32_t action_index = 0; action_index < program.action_count; ++action_index) {
        const Action action = program.actions[action_index];
        switch (action.tag) {
            case ActionTag::RotateActivePauli:
                apply_rotation(action, evaluate(program, rows.symbols, action.expression), real,
                               imag, lane);
                break;
            case ActionTag::PromoteDormantRotation:
                apply_promotion(action, evaluate(program, rows.symbols, action.expression), real,
                                imag, lane);
                break;
            case ActionTag::MeasureActivePauli:
                reachable = measure_active<Coefficient, Replay>(
                    program, action, rng, rows.forced_records, &log_probability, rows.symbols,
                    rows.records, real, imag, scratch_real, scratch_imag, lane);
                break;
            case ActionTag::MeasureDormantRandom: {
                const bool correction = evaluate(program, rows.symbols, action.expression);
                const bool branch = Replay ? (rows.forced_records[action.index1] != 0) != correction
                                           : rng.next_double() >= 0.5;
                if constexpr (Replay) {
                    log_probability += kLogHalf;
                }
                if (lane.is_writer()) {
                    rows.symbols[action.index0] = static_cast<uint8_t>(branch);
                    rows.records[action.index1] =
                        static_cast<uint8_t>(evaluate(program, rows.symbols, action.expression));
                }
                lane.sync();
                break;
            }
            case ActionTag::RecordClassical: {
                const bool outcome = evaluate(program, rows.symbols, action.expression);
                if (lane.is_writer()) {
                    rows.records[action.index0] = static_cast<uint8_t>(outcome);
                }
                if constexpr (Replay) {
                    reachable = static_cast<uint8_t>(outcome) == rows.forced_records[action.index0];
                }
                lane.sync();
                break;
            }
            case ActionTag::DefineSymbol:
                if (lane.is_writer()) {
                    rows.symbols[action.index0] =
                        static_cast<uint8_t>(evaluate(program, rows.symbols, action.expression));
                }
                lane.sync();
                break;
            case ActionTag::ApplyReadoutNoise: {
                if constexpr (Replay) {
                    // A post-readout record does not identify the sampled flip,
                    // so record-only replay cannot reconstruct this path.
                    reachable = false;
                } else {
                    const bool source = evaluate(program, rows.symbols, action.expression);
                    const double probability = source ? action.value1 : action.value0;
                    const bool flip = probability >= 1.0 ||
                                      (probability > 0.0 && rng.next_double() < probability);
                    if (lane.is_writer()) {
                        rows.symbols[action.index0] = static_cast<uint8_t>(flip);
                        rows.records[action.index1] ^= static_cast<uint8_t>(flip);
                    }
                    lane.sync();
                }
                break;
            }
            case ActionTag::WriteDetector: {
                const bool outcome =
                    evaluate_output_value(program, rows.symbols, rows.records, action);
                if (lane.is_writer()) {
                    rows.detectors[action.index0] = static_cast<uint8_t>(outcome);
                }
                if ((action.flags & kPostselected) != 0 && outcome) {
                    discarded = true;
                }
                break;
            }
            case ActionTag::WriteObservable:
                if (lane.is_writer()) {
                    rows.observables[action.index0] = static_cast<uint8_t>(
                        evaluate_output_value(program, rows.symbols, rows.records, action));
                }
                break;
            case ActionTag::WriteExpectationValue: {
                if ((action.flags & kAbsentActiveProjection) != 0) {
                    if (lane.is_writer()) {
                        rows.exp_vals[action.index0] = 0.0;
                    }
                    break;
                }
                const double value = expectation_value(action, real, imag, lane);
                if (lane.is_writer()) {
                    rows.exp_vals[action.index0] =
                        evaluate(program, rows.symbols, action.expression) ? -value : value;
                }
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

template <typename Coefficient, bool Replay>
__global__ void interpret_shots_thread(ProgramView program, SeedRoot seed_root,
                                       uint64_t shot_offset, uint32_t shots,
                                       Coefficient* coefficient_storage, uint8_t* symbol_storage,
                                       uint8_t* record_storage,
                                       const uint8_t* forced_record_storage,
                                       uint8_t* detector_storage, uint8_t* observable_storage,
                                       double* exp_val_storage, double* log_probability_storage,
                                       uint8_t* reachable_storage, uint8_t* survived) {
    const uint64_t shot =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (shot >= shots) {
        return;
    }
    const uint64_t capacity = coefficient_state_capacity(program.peak_active_width);
    const uint64_t scratch_capacity = coefficient_scratch_capacity(program.peak_active_width);
    const uint64_t coefficient_stride = coefficient_elements_per_shot(program.peak_active_width);
    Coefficient* real = coefficient_storage + shot * coefficient_stride;
    Coefficient* imag = real + capacity;
    Coefficient* scratch_real = imag + capacity;
    Coefficient* scratch_imag = scratch_real + scratch_capacity;
    const ShotRows rows = shot_rows<Replay>(program, static_cast<uint32_t>(shot), symbol_storage,
                                            record_storage, forced_record_storage,
                                            detector_storage, observable_storage, exp_val_storage);
    interpret_one_shot<Coefficient, Replay>(program, seed_root, shot_offset + shot,
                                            static_cast<uint32_t>(shot), real, imag, scratch_real,
                                            scratch_imag, rows, log_probability_storage,
                                            reachable_storage, survived, Lane{});
}

// Block-per-shot kernel. UseShared selects where the coefficients live:
// dynamic shared memory or a per-block global slab. Each block walks shots
// strided by the grid, so global slabs are bounded by the grid rather than the
// shot count.
template <typename Coefficient, bool Replay, bool UseShared>
__global__ void interpret_shots_block(ProgramView program, SeedRoot seed_root,
                                      uint64_t shot_offset, uint32_t shots,
                                      Coefficient* slab_storage, uint8_t* symbol_storage,
                                      uint8_t* record_storage,
                                      const uint8_t* forced_record_storage,
                                      uint8_t* detector_storage, uint8_t* observable_storage,
                                      double* exp_val_storage, double* log_probability_storage,
                                      uint8_t* reachable_storage, uint8_t* survived) {
    __shared__ double reduce[2 * kMaxBlockSize];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];

    const uint64_t capacity = coefficient_state_capacity(program.peak_active_width);
    const uint64_t scratch_capacity = coefficient_scratch_capacity(program.peak_active_width);
    const uint64_t coefficient_stride = coefficient_elements_per_shot(program.peak_active_width);

    Coefficient* real;
    if constexpr (UseShared) {
        real = reinterpret_cast<Coefficient*>(dynamic_shared);
    } else {
        real = slab_storage + static_cast<uint64_t>(blockIdx.x) * coefficient_stride;
    }
    Coefficient* imag = real + capacity;
    Coefficient* scratch_real = imag + capacity;
    Coefficient* scratch_imag = scratch_real + scratch_capacity;

    const Lane lane{threadIdx.x, blockDim.x, reduce};
    for (uint32_t shot = blockIdx.x; shot < shots; shot += gridDim.x) {
        const ShotRows rows = shot_rows<Replay>(program, shot, symbol_storage, record_storage,
                                                forced_record_storage, detector_storage,
                                                observable_storage, exp_val_storage);
        interpret_one_shot<Coefficient, Replay>(program, seed_root, shot_offset + shot, shot, real,
                                                imag, scratch_real, scratch_imag, rows,
                                                log_probability_storage, reachable_storage,
                                                survived, lane);
    }
}

}  // namespace

}  // namespace clifft::sampling::cuda::detail

#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft::sampling::cuda {

namespace {

class BusyGuard {
  public:
    explicit BusyGuard(std::atomic_flag& busy) : busy_(busy) {
        if (busy_.test_and_set(std::memory_order_acquire)) {
            throw std::runtime_error("calls on one CUDA Sampler instance must not overlap");
        }
    }

    ~BusyGuard() { busy_.clear(std::memory_order_release); }

    BusyGuard(const BusyGuard&) = delete;
    BusyGuard& operator=(const BusyGuard&) = delete;

  private:
    std::atomic_flag& busy_;
};

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
    return static_cast<size_t>(rows) * static_cast<size_t>(stride);
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
    [[nodiscard]] size_t bytes() const { return count_ * sizeof(T); }

    void download(std::span<T> destination) const {
        if (destination.size() != count_) {
            throw std::invalid_argument("CUDA download destination must match the device buffer");
        }
        download_prefix(destination);
    }

    void download_prefix(std::span<T> destination) const {
        if (destination.size() > count_) {
            throw std::invalid_argument("CUDA download destination exceeds the device buffer");
        }
        if (!destination.empty()) {
            check_cuda(cudaMemcpy(destination.data(), data_, destination.size_bytes(),
                                  cudaMemcpyDeviceToHost),
                       "result download");
        }
    }

    void upload(std::span<const T> source) {
        if (source.size() > count_) {
            throw std::invalid_argument("CUDA upload source exceeds the device buffer");
        }
        if (!source.empty()) {
            check_cuda(
                cudaMemcpy(data_, source.data(), source.size_bytes(), cudaMemcpyHostToDevice),
                "input upload");
        }
    }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct UploadedProgram {
    explicit UploadedProgram(const ExecutablePlan& executable)
        : actions(executable.actions()),
          expressions(executable.expressions()),
          expression_terms(executable.expression_terms()),
          noise_sites(executable.noise_sites()),
          noise_outcomes(executable.noise_outcomes()),
          num_visible_records(executable.num_visible_records()),
          has_postselection(executable.has_postselection()) {
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
    uint32_t num_visible_records;
    bool has_postselection;

    [[nodiscard]] size_t bytes() const {
        return actions.bytes() + expressions.bytes() + expression_terms.bytes() +
               noise_sites.bytes() + noise_outcomes.bytes();
    }
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

size_t coefficient_bytes(const ExecutablePlan& executable, CoefficientPrecision precision) {
    const size_t element =
        precision == CoefficientPrecision::FP32 ? sizeof(float) : sizeof(double);
    return static_cast<size_t>(detail::coefficient_elements_per_shot(
               executable.peak_active_width())) *
           element;
}

bool fits_shared(const ExecutablePlan& executable, CoefficientPrecision precision,
                 const DeviceLimits& limits) {
    return coefficient_bytes(executable, precision) + detail::kReductionScratchBytes <=
           limits.shared_optin;
}

ExecutionTier resolve_tier(const ExecutablePlan& executable, CoefficientPrecision precision,
                           ExecutionTier requested, const DeviceLimits& limits) {
    if (requested != ExecutionTier::Auto) {
        if (requested == ExecutionTier::BlockShared &&
            !fits_shared(executable, precision, limits)) {
            throw std::invalid_argument(
                "CUDA BlockShared tier does not fit this device's shared memory");
        }
        return requested;
    }
    if (executable.peak_active_width() <= kThreadPerShotMaxActiveWidth) {
        return ExecutionTier::ThreadPerShot;
    }
    if (fits_shared(executable, precision, limits)) {
        return ExecutionTier::BlockShared;
    }
    return ExecutionTier::BlockGlobal;
}

// Number of shots the cooperative tiers keep resident per launch. Enough
// blocks to oversubscribe every multiprocessor regardless of occupancy; more
// buys no parallelism and, for the global tier, multiplies slab memory.
uint32_t resolve_concurrency(const ExecutablePlan& executable, CoefficientPrecision precision,
                             const DeviceLimits& limits, ExecutionTier tier,
                             uint32_t max_batch_shots, uint32_t requested) {
    if (tier == ExecutionTier::ThreadPerShot) {
        return max_batch_shots;
    }
    if (requested != 0) {
        return std::min(max_batch_shots, requested);
    }
    const uint64_t target =
        std::max<uint64_t>(static_cast<uint64_t>(limits.multiprocessors) * 32, 1);
    if (tier == ExecutionTier::BlockShared) {
        return static_cast<uint32_t>(std::min<uint64_t>(max_batch_shots, target));
    }
    // Leave real headroom: slab pools that merely fit free memory can still
    // starve the runtime's own launch-time allocations.
    const size_t slab = coefficient_bytes(executable, precision);
    const size_t reserve = std::max<size_t>(limits.free_memory / 10, size_t{2} << 30);
    const size_t budget = limits.free_memory > reserve ? limits.free_memory - reserve : 0;
    const uint64_t fit = std::max<uint64_t>(budget / slab, 1);
    return static_cast<uint32_t>(
        std::min<uint64_t>(max_batch_shots, std::min<uint64_t>(fit, target)));
}

void validate_block_size(uint32_t block_size) {
    if (block_size == 0 || block_size > detail::kMaxBlockSize ||
        (block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument("CUDA block size must be a power of two between 1 and 1024");
    }
}

void validate_max_batch_shots(uint32_t max_batch_shots) {
    if (max_batch_shots == 0) {
        throw std::invalid_argument("CUDA max_batch_shots must be positive");
    }
}

void validate_replay_input(uint32_t num_records, uint32_t noise_site_count,
                           std::span<const uint8_t> forced_records) {
    if (forced_records.size() != num_records) {
        throw std::invalid_argument(
            "CUDA replay requires one forced value for every visible and hidden record");
    }
    if (!std::all_of(forced_records.begin(), forced_records.end(),
                     [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("CUDA replay record bytes must be Boolean");
    }
    if (noise_site_count != 0) {
        throw std::invalid_argument("CUDA replay does not support presampled noise");
    }
}

void validate_replay_input(const ExecutablePlan& executable,
                           std::span<const uint8_t> forced_records) {
    validate_replay_input(executable.num_records(),
                          static_cast<uint32_t>(executable.noise_sites().size()), forced_records);
}

}  // namespace

class Sampler::Impl {
  public:
    enum class DownloadMode : uint8_t {
        FullRows,
        SurvivorCounts,
        Replay,
    };

    Impl(const ExecutablePlan& source, CoefficientPrecision selected_precision,
         uint32_t selected_max_batch_shots, ExecutionTier requested_tier,
         uint32_t requested_concurrency)
        : limits(query_device()),
          program(source),
          precision(selected_precision),
          max_batch(selected_max_batch_shots),
          tier(resolve_tier(source, selected_precision, requested_tier, limits)),
          concurrency(resolve_concurrency(source, selected_precision, limits, tier, max_batch,
                                          requested_concurrency)),
          coefficient_stride(detail::coefficient_elements_per_shot(source.peak_active_width())),
          slab_rows(tier == ExecutionTier::ThreadPerShot ? max_batch
                    : tier == ExecutionTier::BlockGlobal ? concurrency
                                                         : 0),
          dynamic_shared_bytes(tier == ExecutionTier::BlockShared
                                   ? coefficient_bytes(source, selected_precision)
                                   : 0),
          fp32_coefficients(selected_precision == CoefficientPrecision::FP32
                                ? checked_elements(slab_rows, coefficient_stride, "coefficient")
                                : 0),
          fp64_coefficients(selected_precision == CoefficientPrecision::FP64
                                ? checked_elements(slab_rows, coefficient_stride, "coefficient")
                                : 0),
          symbols(checked_elements(max_batch, source.num_symbols(), "symbol")),
          records(checked_elements(max_batch, source.num_records(), "record")),
          forced_records(source.num_records()),
          detectors(checked_elements(max_batch, source.num_detectors(), "detector")),
          observables(checked_elements(max_batch, source.num_observables(), "observable")),
          exp_vals(checked_elements(max_batch, source.num_exp_vals(), "expectation")),
          log_probabilities(1),
          reachable(1),
          survived(max_batch),
          host_records(checked_elements(max_batch, source.num_records(), "host record")),
          host_detectors(checked_elements(max_batch, source.num_detectors(), "host detector")),
          host_observables(
              checked_elements(max_batch, source.num_observables(), "host observable")),
          host_exp_vals(checked_elements(max_batch, source.num_exp_vals(), "host expectation")),
          host_survived(max_batch) {}

    void run_batch(detail::SeedRoot root, uint64_t shot_offset, uint32_t shots,
                   uint32_t block_size, DownloadMode download_mode) {
        if (shots > max_batch) {
            throw std::invalid_argument("CUDA batch exceeds retained workspace capacity");
        }
        if (precision == CoefficientPrecision::FP32) {
            launch<float, false>(root, shot_offset, shots, block_size, fp32_coefficients.data());
        } else {
            launch<double, false>(root, shot_offset, shots, block_size, fp64_coefficients.data());
        }
        download_rows(shots, download_mode);
    }

    void run_replay(std::span<const uint8_t> input) {
        forced_records.upload(input);
        const detail::SeedRoot empty_root{};
        const uint32_t block_size =
            tier == ExecutionTier::ThreadPerShot ? 1 : kDefaultBlockSize;
        if (precision == CoefficientPrecision::FP32) {
            launch<float, true>(empty_root, 0, 1, block_size, fp32_coefficients.data());
        } else {
            launch<double, true>(empty_root, 0, 1, block_size, fp64_coefficients.data());
        }
        download_rows(1, DownloadMode::Replay);
    }

    [[nodiscard]] size_t allocated_device_bytes() const {
        return program.bytes() + fp32_coefficients.bytes() + fp64_coefficients.bytes() +
               symbols.bytes() + records.bytes() + forced_records.bytes() + detectors.bytes() +
               observables.bytes() + exp_vals.bytes() + log_probabilities.bytes() +
               reachable.bytes() + survived.bytes();
    }

    [[nodiscard]] uint32_t num_visible_records() const { return program.num_visible_records; }
    [[nodiscard]] uint32_t num_records() const { return program.view.num_records; }
    [[nodiscard]] uint32_t num_detectors() const { return program.view.num_detectors; }
    [[nodiscard]] uint32_t num_observables() const { return program.view.num_observables; }
    [[nodiscard]] uint32_t num_exp_vals() const { return program.view.num_exp_vals; }

    DeviceLimits limits;
    UploadedProgram program;
    CoefficientPrecision precision;
    uint32_t max_batch;
    ExecutionTier tier;
    uint32_t concurrency;
    uint64_t coefficient_stride;
    uint64_t slab_rows;
    size_t dynamic_shared_bytes;
    DeviceBuffer<float> fp32_coefficients;
    DeviceBuffer<double> fp64_coefficients;
    DeviceBuffer<uint8_t> symbols;
    DeviceBuffer<uint8_t> records;
    DeviceBuffer<uint8_t> forced_records;
    DeviceBuffer<uint8_t> detectors;
    DeviceBuffer<uint8_t> observables;
    DeviceBuffer<double> exp_vals;
    DeviceBuffer<double> log_probabilities;
    DeviceBuffer<uint8_t> reachable;
    DeviceBuffer<uint8_t> survived;
    std::vector<uint8_t> host_records;
    std::vector<uint8_t> host_detectors;
    std::vector<uint8_t> host_observables;
    std::vector<double> host_exp_vals;
    std::vector<uint8_t> host_survived;
    double host_log_probability = 0.0;
    uint8_t host_reachable = 0;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;

  private:
    template <typename Coefficient, bool Replay>
    void launch(detail::SeedRoot root, uint64_t shot_offset, uint32_t shots, uint32_t block_size,
                Coefficient* coefficient_storage) {
        // A failed earlier launch leaves the runtime's last error set; consume
        // it so this launch reports its own status.
        (void)cudaGetLastError();
        switch (tier) {
            case ExecutionTier::ThreadPerShot: {
                const uint32_t blocks = static_cast<uint32_t>(
                    (static_cast<uint64_t>(shots) + block_size - 1) / block_size);
                detail::interpret_shots_thread<Coefficient, Replay><<<blocks, block_size>>>(
                    program.view, root, shot_offset, shots, coefficient_storage, symbols.data(),
                    records.data(), forced_records.data(), detectors.data(), observables.data(),
                    exp_vals.data(), log_probabilities.data(), reachable.data(),
                    survived.data());
                break;
            }
            case ExecutionTier::BlockShared: {
                auto* kernel = detail::interpret_shots_block<Coefficient, Replay, true>;
                check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(dynamic_shared_bytes)),
                           "shared memory opt-in");
                const uint32_t grid = std::min(concurrency, shots);
                kernel<<<grid, block_size, dynamic_shared_bytes>>>(
                    program.view, root, shot_offset, shots, coefficient_storage, symbols.data(),
                    records.data(), forced_records.data(), detectors.data(), observables.data(),
                    exp_vals.data(), log_probabilities.data(), reachable.data(),
                    survived.data());
                break;
            }
            case ExecutionTier::BlockGlobal: {
                const uint32_t grid = std::min(concurrency, shots);
                detail::interpret_shots_block<Coefficient, Replay, false><<<grid, block_size>>>(
                    program.view, root, shot_offset, shots, coefficient_storage, symbols.data(),
                    records.data(), forced_records.data(), detectors.data(), observables.data(),
                    exp_vals.data(), log_probabilities.data(), reachable.data(),
                    survived.data());
                break;
            }
            case ExecutionTier::Auto:
                throw std::logic_error("CUDA tier must be resolved before launch");
        }
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "kernel execution");
    }

    void download_rows(uint32_t shots, DownloadMode mode) {
        if (mode != DownloadMode::SurvivorCounts) {
            records.download_prefix(
                std::span(host_records)
                    .first(checked_elements(shots, program.view.num_records, "record result")));
            detectors.download_prefix(
                std::span(host_detectors)
                    .first(checked_elements(shots, program.view.num_detectors, "detector result")));
            exp_vals.download_prefix(
                std::span(host_exp_vals)
                    .first(checked_elements(shots, program.view.num_exp_vals, "expectation result")));
        }
        observables.download_prefix(
            std::span(host_observables)
                .first(checked_elements(shots, program.view.num_observables, "observable result")));
        survived.download_prefix(std::span(host_survived).first(shots));
        if (mode == DownloadMode::Replay) {
            log_probabilities.download(std::span(&host_log_probability, 1));
            reachable.download(std::span(&host_reachable, 1));
        }
    }
};

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
            output << "\n[" << device << "] " << properties.name << " sm_" << properties.major
                   << properties.minor
                   << " shared_optin_bytes=" << properties.sharedMemPerBlockOptin;
        }
    }
    return output.str();
}

ExecutionTier selected_tier(const ExecutablePlan& executable,
                            CoefficientPrecision coefficient_precision) {
    return resolve_tier(executable, coefficient_precision, ExecutionTier::Auto, query_device());
}

Sampler::Sampler(const ExecutablePlan& executable, CoefficientPrecision coefficient_precision,
                 uint32_t max_batch_shots, ExecutionTier tier, uint32_t max_concurrent_shots) {
    validate_max_batch_shots(max_batch_shots);
    impl_ = std::make_unique<Impl>(executable, coefficient_precision, max_batch_shots, tier,
                                   max_concurrent_shots);
}

Sampler::~Sampler() = default;
Sampler::Sampler(Sampler&&) noexcept = default;
Sampler& Sampler::operator=(Sampler&&) noexcept = default;

SamplingResult Sampler::sample(uint32_t shots, std::optional<uint64_t> seed, uint32_t block_size) {
    BusyGuard guard(impl_->busy);
    validate_block_size(block_size);
    if (impl_->program.has_postselection) {
        throw std::invalid_argument(
            "CUDA fixed-row sampling does not support postselection; use sample_survivors");
    }
    SamplingResult result;
    if (shots == 0) {
        return result;
    }
    result.measurements.resize(
        checked_elements(shots, impl_->num_visible_records(), "measurement result"));
    result.detectors.resize(checked_elements(shots, impl_->num_detectors(), "detector result"));
    result.observables.resize(
        checked_elements(shots, impl_->num_observables(), "observable result"));
    result.exp_vals.resize(checked_elements(shots, impl_->num_exp_vals(), "expectation result"));

    const SeedRoot root = make_seed_root(shots, seed);
    const detail::SeedRoot device_root{{root.w[0], root.w[1], root.w[2], root.w[3]}};
    for (uint32_t offset = 0; offset < shots;) {
        const uint32_t batch = std::min(impl_->max_batch, shots - offset);
        impl_->run_batch(device_root, offset, batch, block_size, Impl::DownloadMode::FullRows);
        for (uint32_t local_shot = 0; local_shot < batch; ++local_shot) {
            if (impl_->num_visible_records() != 0) {
                std::copy_n(impl_->host_records.begin() +
                                static_cast<size_t>(local_shot) * impl_->num_records(),
                            impl_->num_visible_records(),
                            result.measurements.begin() +
                                static_cast<size_t>(offset + local_shot) *
                                    impl_->num_visible_records());
            }
        }
        if (impl_->num_detectors() != 0) {
            std::copy_n(impl_->host_detectors.begin(),
                        checked_elements(batch, impl_->num_detectors(), "detector batch"),
                        result.detectors.begin() +
                            static_cast<size_t>(offset) * impl_->num_detectors());
        }
        if (impl_->num_observables() != 0) {
            std::copy_n(impl_->host_observables.begin(),
                        checked_elements(batch, impl_->num_observables(), "observable batch"),
                        result.observables.begin() +
                            static_cast<size_t>(offset) * impl_->num_observables());
        }
        if (impl_->num_exp_vals() != 0) {
            std::copy_n(impl_->host_exp_vals.begin(),
                        checked_elements(batch, impl_->num_exp_vals(), "expectation batch"),
                        result.exp_vals.begin() +
                            static_cast<size_t>(offset) * impl_->num_exp_vals());
        }
        offset += batch;
    }
    return result;
}

SamplingSurvivorResult Sampler::sample_survivors(uint32_t shots, bool keep_records,
                                                 std::optional<uint64_t> seed,
                                                 uint32_t block_size) {
    BusyGuard guard(impl_->busy);
    validate_block_size(block_size);
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(impl_->num_observables(), 0);
    if (keep_records) {
        result.measurements.resize(checked_elements(shots, impl_->num_visible_records(),
                                                    "survivor measurement result"));
        result.detectors.resize(
            checked_elements(shots, impl_->num_detectors(), "survivor detector result"));
        result.observables.resize(
            checked_elements(shots, impl_->num_observables(), "survivor observable result"));
        result.exp_vals.resize(
            checked_elements(shots, impl_->num_exp_vals(), "survivor expectation result"));
    }

    const SeedRoot root = make_seed_root(shots, seed);
    const detail::SeedRoot device_root{{root.w[0], root.w[1], root.w[2], root.w[3]}};
    for (uint32_t offset = 0; offset < shots;) {
        const uint32_t batch = std::min(impl_->max_batch, shots - offset);
        impl_->run_batch(device_root, offset, batch, block_size,
                         keep_records ? Impl::DownloadMode::FullRows
                                      : Impl::DownloadMode::SurvivorCounts);
        for (uint32_t local_shot = 0; local_shot < batch; ++local_shot) {
            if (impl_->host_survived[local_shot] == 0) {
                continue;
            }
            const uint32_t destination = result.passed_shots++;
            bool logical_error = false;
            for (uint32_t observable = 0; observable < impl_->num_observables(); ++observable) {
                const bool value = impl_->host_observables[static_cast<size_t>(local_shot) *
                                                               impl_->num_observables() +
                                                           observable] != 0;
                result.observable_ones[observable] += static_cast<uint64_t>(value);
                logical_error |= value;
            }
            result.logical_errors += static_cast<uint32_t>(logical_error);
            if (!keep_records) {
                continue;
            }
            if (impl_->num_visible_records() != 0) {
                std::copy_n(impl_->host_records.begin() +
                                static_cast<size_t>(local_shot) * impl_->num_records(),
                            impl_->num_visible_records(),
                            result.measurements.begin() + static_cast<size_t>(destination) *
                                                              impl_->num_visible_records());
            }
            if (impl_->num_detectors() != 0) {
                std::copy_n(impl_->host_detectors.begin() +
                                static_cast<size_t>(local_shot) * impl_->num_detectors(),
                            impl_->num_detectors(),
                            result.detectors.begin() +
                                static_cast<size_t>(destination) * impl_->num_detectors());
            }
            if (impl_->num_observables() != 0) {
                std::copy_n(impl_->host_observables.begin() +
                                static_cast<size_t>(local_shot) * impl_->num_observables(),
                            impl_->num_observables(),
                            result.observables.begin() +
                                static_cast<size_t>(destination) * impl_->num_observables());
            }
            if (impl_->num_exp_vals() != 0) {
                std::copy_n(impl_->host_exp_vals.begin() +
                                static_cast<size_t>(local_shot) * impl_->num_exp_vals(),
                            impl_->num_exp_vals(),
                            result.exp_vals.begin() +
                                static_cast<size_t>(destination) * impl_->num_exp_vals());
            }
        }
        offset += batch;
    }
    if (keep_records) {
        result.measurements.resize(static_cast<size_t>(result.passed_shots) *
                                   impl_->num_visible_records());
        result.detectors.resize(static_cast<size_t>(result.passed_shots) *
                                impl_->num_detectors());
        result.observables.resize(static_cast<size_t>(result.passed_shots) *
                                  impl_->num_observables());
        result.exp_vals.resize(static_cast<size_t>(result.passed_shots) * impl_->num_exp_vals());
    }
    return result;
}

ReplayResult Sampler::replay_shot(std::span<const uint8_t> forced_records) {
    BusyGuard guard(impl_->busy);
    validate_replay_input(impl_->num_records(), impl_->program.view.noise_site_count,
                          forced_records);
    impl_->run_replay(forced_records);
    ReplayResult result;
    result.reachable = impl_->host_reachable != 0;
    result.survived = impl_->host_survived[0] != 0;
    result.log_probability = impl_->host_log_probability;
    if (!result.reachable || !result.survived) {
        return result;
    }
    result.outputs.measurements.assign(
        impl_->host_records.begin(), impl_->host_records.begin() + impl_->num_visible_records());
    result.outputs.detectors.assign(impl_->host_detectors.begin(),
                                    impl_->host_detectors.begin() + impl_->num_detectors());
    result.outputs.observables.assign(
        impl_->host_observables.begin(), impl_->host_observables.begin() + impl_->num_observables());
    result.outputs.exp_vals.assign(impl_->host_exp_vals.begin(),
                                   impl_->host_exp_vals.begin() + impl_->num_exp_vals());
    return result;
}

CoefficientPrecision Sampler::coefficient_precision() const {
    return impl_->precision;
}

ExecutionTier Sampler::execution_tier() const {
    return impl_->tier;
}

uint32_t Sampler::max_batch_shots() const {
    return impl_->max_batch;
}

uint32_t Sampler::max_concurrent_shots() const {
    return impl_->concurrency;
}

size_t Sampler::allocated_device_bytes() const {
    return impl_->allocated_device_bytes();
}

uint32_t Sampler::num_visible_records() const {
    return impl_->num_visible_records();
}

uint32_t Sampler::num_records() const {
    return impl_->num_records();
}

uint32_t Sampler::num_detectors() const {
    return impl_->num_detectors();
}

uint32_t Sampler::num_observables() const {
    return impl_->num_observables();
}

uint32_t Sampler::num_exp_vals() const {
    return impl_->num_exp_vals();
}

SamplingResult sample(const ExecutablePlan& executable, uint32_t shots,
                      const SamplingOptions& options) {
    validate_block_size(options.block_size);
    validate_max_batch_shots(options.max_batch_shots);
    if (executable.has_postselection()) {
        throw std::invalid_argument(
            "CUDA fixed-row sampling does not support postselection; use sample_survivors");
    }
    if (shots == 0) {
        return SamplingResult{};
    }
    Sampler sampler(executable, options.coefficient_precision,
                    std::min(shots, options.max_batch_shots), options.tier,
                    options.max_concurrent_shots);
    return sampler.sample(shots, options.seed, options.block_size);
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& executable, uint32_t shots,
                                        bool keep_records, const SamplingOptions& options) {
    validate_block_size(options.block_size);
    validate_max_batch_shots(options.max_batch_shots);
    if (shots == 0) {
        SamplingSurvivorResult empty;
        empty.total_shots = 0;
        return empty;
    }
    Sampler sampler(executable, options.coefficient_precision,
                    std::min(shots, options.max_batch_shots), options.tier,
                    options.max_concurrent_shots);
    return sampler.sample_survivors(shots, keep_records, options.seed, options.block_size);
}

ReplayResult replay_shot(const ExecutablePlan& executable, std::span<const uint8_t> forced_records,
                         CoefficientPrecision coefficient_precision) {
    validate_replay_input(executable, forced_records);
    Sampler sampler(executable, coefficient_precision, 1);
    return sampler.replay_shot(forced_records);
}

}  // namespace clifft::sampling::cuda
