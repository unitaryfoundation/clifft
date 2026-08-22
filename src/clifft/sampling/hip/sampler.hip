#include "clifft/sampling/hip/device_program.h"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace clifft::sampling::hip::detail {

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr double kMeasurementDustEpsilon = 1e-18;
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
__device__ ComplexValue<Coefficient> load(const Coefficient* real, const Coefficient* imag,
                                          uint64_t index) {
    return {real[index], imag[index]};
}

template <typename Coefficient>
__device__ void store(Coefficient* real, Coefficient* imag, uint64_t index,
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
__device__ void apply_rotation(const Action& action, bool sign, Coefficient* real,
                               Coefficient* imag) {
    const uint64_t size = uint64_t{1} << action.active_before;
    const Coefficient cosine = static_cast<Coefficient>(action.value0);
    const Coefficient sine =
        sign ? -static_cast<Coefficient>(action.value1) : static_cast<Coefficient>(action.value1);
    if (action.x == 0) {
        for (uint64_t basis = 0; basis < size; ++basis) {
            const Coefficient eigenvalue =
                parity(basis & action.z) ? Coefficient{-1} : Coefficient{1};
            const Coefficient signed_sine = sine * eigenvalue;
            const Coefficient input_real = real[basis];
            const Coefficient input_imag = imag[basis];
            real[basis] = cosine * input_real + signed_sine * input_imag;
            imag[basis] = cosine * input_imag - signed_sine * input_real;
        }
        return;
    }

    const bool real_phase = action.phase_real != 0;
    const Coefficient base_phase =
        static_cast<Coefficient>(real_phase ? action.phase_real : action.phase_imag);
    const Coefficient even_left_sine = sine * base_phase;
    const uint64_t pair_stride = action.auxiliary_mask;
    const uint64_t pair_period = pair_stride << 1;
    for (uint64_t block = 0; block < size; block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; ++offset) {
            const uint64_t left = block + offset;
            const uint64_t right = left ^ action.x;
            const Coefficient left_real = real[left];
            const Coefficient left_imag = imag[left];
            const Coefficient right_real = real[right];
            const Coefficient right_imag = imag[right];
            const Coefficient left_sine =
                parity(left & action.z) ? -even_left_sine : even_left_sine;
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
    }
}

template <typename Coefficient>
__device__ void apply_promotion(const Action& action, bool sign, Coefficient* real,
                                Coefficient* imag) {
    const uint64_t old_size = uint64_t{1} << action.active_before;
    const Coefficient cosine = static_cast<Coefficient>(action.value0);
    const Coefficient sine =
        sign ? -static_cast<Coefficient>(action.value1) : static_cast<Coefficient>(action.value1);
    for (uint64_t basis = 0; basis < old_size; ++basis) {
        const Coefficient input_real = real[basis];
        const Coefficient input_imag = imag[basis];
        real[basis] = cosine * input_real;
        imag[basis] = cosine * input_imag;
        real[old_size + basis] = sine * input_imag;
        imag[old_size + basis] = -sine * input_real;
    }
}

template <typename Coefficient>
__device__ ComplexValue<Coefficient> compact_nondiagonal(const Action& action,
                                                         const Coefficient* real,
                                                         const Coefficient* imag, uint64_t packed,
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

template <typename Coefficient, bool Replay>
__device__ bool measure_active(const ProgramView& program, const Action& action,
                               Xoshiro256PlusPlus& rng, const uint8_t* forced_records,
                               double* log_probability, uint8_t* symbols, uint8_t* records,
                               Coefficient* real, Coefficient* imag, Coefficient* scratch_real,
                               Coefficient* scratch_imag) {
    const uint64_t size = uint64_t{1} << action.active_before;
    const uint64_t output_size = size >> 1;
    double probability_zero = 0.0;
    double probability_one = 0.0;
    if (action.x == 0) {
        for (uint64_t basis = 0; basis < size; ++basis) {
            const double probability = norm(load(real, imag, basis));
            if (parity(basis & action.z)) {
                probability_one += probability;
            } else {
                probability_zero += probability;
            }
        }
    } else {
        for (uint64_t packed = 0; packed < output_size; ++packed) {
            probability_zero += norm(compact_nondiagonal(action, real, imag, packed, false));
            probability_one += norm(compact_nondiagonal(action, real, imag, packed, true));
        }
    }

    const double total = probability_zero + probability_one;
    const double epsilon = kMeasurementDustEpsilon * total;
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
    symbols[action.index0] = static_cast<uint8_t>(branch);
    records[action.index1] = static_cast<uint8_t>(evaluate(program, symbols, action.expression));

    if (action.x == 0) {
        for (uint64_t packed = 0; packed < output_size; ++packed) {
            const uint64_t without_pivot = insert_zero_bit(packed, action.index2);
            const bool other_parity = parity(without_pivot & action.auxiliary_mask);
            const bool pivot_value = branch != other_parity;
            const uint64_t source =
                without_pivot | (static_cast<uint64_t>(pivot_value) << action.index2);
            const ComplexValue<Coefficient> value = load(real, imag, source);
            store(real, imag, packed, normalize(value, inverse_norm));
        }
    } else {
        for (uint64_t packed = 0; packed < output_size; ++packed) {
            const ComplexValue<Coefficient> value =
                compact_nondiagonal(action, real, imag, packed, branch);
            store(scratch_real, scratch_imag, packed, normalize(value, inverse_norm));
        }
        for (uint64_t packed = 0; packed < output_size; ++packed) {
            real[packed] = scratch_real[packed];
            imag[packed] = scratch_imag[packed];
        }
    }
    return true;
}

template <typename Coefficient>
__device__ double expectation_value(const Action& action, const Coefficient* real,
                                    const Coefficient* imag) {
    if ((action.flags & kAbsentActiveProjection) != 0) {
        return 0.0;
    }
    if (action.x == 0 && action.z == 0) {
        return 1.0;
    }
    const uint64_t size = uint64_t{1} << action.active_before;
    double result = 0.0;
    if (action.x == 0) {
        for (uint64_t basis = 0; basis < size; ++basis) {
            const double eigenvalue = parity(basis & action.z) ? -1.0 : 1.0;
            result += eigenvalue * norm(load(real, imag, basis));
        }
        return result;
    }
    for (uint64_t basis = 0; basis < size; ++basis) {
        const ComplexValue<Coefficient> paired_coefficient = load(real, imag, basis ^ action.x);
        const ComplexValue<Coefficient> basis_coefficient = load(real, imag, basis);
        const ComplexValue<double> paired{static_cast<double>(paired_coefficient.real),
                                          static_cast<double>(paired_coefficient.imag)};
        const ComplexValue<double> basis_value{static_cast<double>(basis_coefficient.real),
                                               static_cast<double>(basis_coefficient.imag)};
        const ComplexValue<double> phased = multiply(phase_at<double>(action, basis), basis_value);
        result += paired.real * phased.real + paired.imag * phased.imag;
    }
    return result;
}

__device__ __forceinline__ void sample_noise(const ProgramView& program, Xoshiro256PlusPlus& rng,
                                             uint8_t* symbols) {
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
        symbols[program.noise_outcomes[outcome].symbol] = 1;
    }
}

template <typename Coefficient, bool Replay>
__global__ void interpret_shots(ProgramView program, SeedRoot seed_root, uint32_t shots,
                                Coefficient* coefficient_storage, uint8_t* symbol_storage,
                                uint8_t* record_storage, const uint8_t* forced_record_storage,
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

    for (uint64_t basis = 0; basis < (uint64_t{1} << program.initial_active_width); ++basis) {
        real[basis] = 0;
        imag[basis] = 0;
    }
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

    Xoshiro256PlusPlus rng = shot_rng(seed_root, shot);
    if constexpr (!Replay) {
        sample_noise(program, rng, symbols);
    }
    double log_probability = 0.0;
    bool reachable = true;
    bool discarded = false;
    for (uint32_t action_index = 0; action_index < program.action_count; ++action_index) {
        const Action action = program.actions[action_index];
        switch (action.tag) {
            case ActionTag::RotateActivePauli:
                apply_rotation(action, evaluate(program, symbols, action.expression), real, imag);
                break;
            case ActionTag::PromoteDormantRotation:
                apply_promotion(action, evaluate(program, symbols, action.expression), real, imag);
                break;
            case ActionTag::MeasureActivePauli:
                reachable = measure_active<Coefficient, Replay>(
                    program, action, rng, forced_records, &log_probability, symbols, records, real,
                    imag, scratch_real, scratch_imag);
                break;
            case ActionTag::MeasureDormantRandom: {
                const bool correction = evaluate(program, symbols, action.expression);
                const bool branch = Replay ? (forced_records[action.index1] != 0) != correction
                                           : rng.next_double() >= 0.5;
                if constexpr (Replay) {
                    log_probability += kLogHalf;
                }
                symbols[action.index0] = static_cast<uint8_t>(branch);
                records[action.index1] =
                    static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                break;
            }
            case ActionTag::RecordClassical:
                records[action.index0] =
                    static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                if constexpr (Replay) {
                    reachable = records[action.index0] == forced_records[action.index0];
                }
                break;
            case ActionTag::DefineSymbol:
                symbols[action.index0] =
                    static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                break;
            case ActionTag::ApplyReadoutNoise: {
                const bool source = evaluate(program, symbols, action.expression);
                const double probability = source ? action.value1 : action.value0;
                const bool flip =
                    probability >= 1.0 || (probability > 0.0 && rng.next_double() < probability);
                symbols[action.index0] = static_cast<uint8_t>(flip);
                records[action.index1] ^= static_cast<uint8_t>(flip);
                break;
            }
            case ActionTag::WriteDetector: {
                const bool outcome = evaluate(program, symbols, action.expression);
                detectors[action.index0] = static_cast<uint8_t>(outcome);
                if ((action.flags & kPostselected) != 0 && outcome) {
                    discarded = true;
                }
                break;
            }
            case ActionTag::WriteObservable:
                observables[action.index0] =
                    static_cast<uint8_t>(evaluate(program, symbols, action.expression));
                break;
            case ActionTag::WriteExpectationValue: {
                const double value = expectation_value(action, real, imag);
                exp_vals[action.index0] =
                    evaluate(program, symbols, action.expression) ? -value : value;
                break;
            }
        }
        if (!reachable || discarded) {
            break;
        }
    }
    if constexpr (Replay) {
        log_probability_storage[shot] = log_probability;
        reachable_storage[shot] = static_cast<uint8_t>(reachable);
    }
    survived[shot] = static_cast<uint8_t>(reachable && !discarded);
}

}  // namespace

}  // namespace clifft::sampling::hip::detail

#if !defined(__HIP_DEVICE_COMPILE__)

#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/hip/sampler.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft::sampling::hip {

namespace {

void check_hip(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("HIP ") + operation +
                                 " failed: " + hipGetErrorString(error));
    }
}

size_t checked_elements(uint32_t shots, uint64_t stride, const char* storage) {
    if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error(std::string("HIP ") + storage + " allocation exceeds size_t");
    }
    return static_cast<size_t>(shots) * static_cast<size_t>(stride);
}

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count != 0) {
            if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
                throw std::length_error("HIP allocation byte size exceeds size_t");
            }
            check_hip(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "allocation");
        }
    }

    explicit DeviceBuffer(std::span<const T> source) : DeviceBuffer(source.size()) {
        if (!source.empty()) {
            check_hip(hipMemcpy(data_, source.data(), source.size_bytes(), hipMemcpyHostToDevice),
                      "program upload");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)hipFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                (void)hipFree(data_);
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
            throw std::invalid_argument("HIP download destination has the wrong size");
        }
        if (!destination.empty()) {
            check_hip(hipMemcpy(destination.data(), data_, destination.size_bytes(),
                                hipMemcpyDeviceToHost),
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

template <typename Coefficient, bool Replay>
Rows run_device(const Executable& executable, uint32_t shots, const SamplingOptions& options,
                std::span<const uint8_t> forced_record_input = {}) {
    if (options.block_size == 0 || options.block_size > 1024) {
        throw std::invalid_argument("HIP block size must be between 1 and 1024");
    }
    UploadedProgram program(executable);
    const uint64_t capacity = uint64_t{1} << executable.peak_active_width();
    const uint64_t scratch_capacity = capacity > 1 ? capacity >> 1 : 1;
    const uint64_t coefficient_stride = 2 * capacity + 2 * scratch_capacity;

    DeviceBuffer<Coefficient> coefficients(
        checked_elements(shots, coefficient_stride, "coefficient"));
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
    const detail::SeedRoot device_root{{root.w[0], root.w[1], root.w[2], root.w[3]}};
    const uint32_t blocks = static_cast<uint32_t>(
        (static_cast<uint64_t>(shots) + options.block_size - 1) / options.block_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::interpret_shots<Coefficient, Replay>), dim3(blocks),
                       dim3(options.block_size), 0, 0, program.view, device_root, shots,
                       coefficients.data(), symbols.data(), records.data(), forced_records.data(),
                       detectors.data(), observables.data(), exp_vals.data(),
                       log_probabilities.data(), reachable.data(), survived.data());
    check_hip(hipGetLastError(), "kernel launch");
    check_hip(hipDeviceSynchronize(), "kernel execution");

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
    const SamplingOptions options{.coefficient_precision = coefficient_precision, .block_size = 1};
    if (coefficient_precision == CoefficientPrecision::FP32) {
        return run_device<float, true>(executable, 1, options, forced_records);
    }
    return run_device<double, true>(executable, 1, options, forced_records);
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
    return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

std::string backend_info() {
    int device_count = 0;
    const hipError_t count_error = hipGetDeviceCount(&device_count);
    if (count_error != hipSuccess) {
        return std::string("HIP runtime unavailable: ") + hipGetErrorString(count_error);
    }
    std::ostringstream output;
    output << "HIP devices: " << device_count;
    for (int device = 0; device < device_count; ++device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) == hipSuccess) {
            output << "\n[" << device << "] " << properties.name << " " << properties.gcnArchName;
        }
    }
    return output.str();
}

SamplingResult sample(const Executable& executable, uint32_t shots,
                      const SamplingOptions& options) {
    if (executable.has_postselection()) {
        throw std::invalid_argument(
            "HIP fixed-row sampling does not support postselection; use sample_survivors");
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
            "HIP replay requires one forced value for every visible and hidden record");
    }
    if (!std::all_of(forced_records.begin(), forced_records.end(),
                     [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("HIP replay record bytes must be Boolean");
    }
    if (!executable.noise_sites().empty()) {
        throw std::invalid_argument("HIP replay does not support presampled noise");
    }
    if (std::any_of(executable.actions().begin(), executable.actions().end(),
                    [](const detail::Action& action) {
                        return action.tag == detail::ActionTag::ApplyReadoutNoise;
                    })) {
        throw std::invalid_argument("HIP replay does not support readout noise");
    }

    Rows rows = execute_replay(executable, forced_records, coefficient_precision);
    ReplayResult result;
    result.reachable = rows.reachable[0] != 0;
    result.survived = rows.survived[0] != 0;
    result.log_probability = rows.log_probabilities[0];
    if (!result.reachable) {
        return result;
    }
    result.outputs.measurements.assign(rows.records.begin(),
                                       rows.records.begin() + executable.num_visible_records());
    result.outputs.detectors = std::move(rows.detectors);
    result.outputs.observables = std::move(rows.observables);
    result.outputs.exp_vals = std::move(rows.exp_vals);
    return result;
}

}  // namespace clifft::sampling::hip

#endif
