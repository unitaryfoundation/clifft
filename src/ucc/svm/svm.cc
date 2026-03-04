#include "ucc/svm/svm.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ucc {

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(uint32_t peak_rank, uint32_t num_measurements,
                                   uint32_t num_detectors, uint32_t num_observables, uint64_t seed)
    : peak_rank_(peak_rank), rng_(seed) {
    meas_record.resize(num_measurements, 0);
    det_record.resize(num_detectors, 0);
    obs_record.resize(num_observables, 0);

    // Allocate 2^peak_rank complex numbers, 64-byte aligned for AVX
    array_size_ = 1ULL << peak_rank;
    size_t bytes = array_size_ * sizeof(std::complex<double>);
    size_t aligned_bytes = (bytes + 63) & ~63ULL;
    v_ = static_cast<std::complex<double>*>(std::aligned_alloc(64, aligned_bytes));
    if (!v_) {
        throw std::bad_alloc();
    }

    // Initialize to |0...0>: coefficient 1 at index 0
    for (uint64_t i = 0; i < array_size_; ++i) {
        v_[i] = {0.0, 0.0};
    }
    v_[0] = {1.0, 0.0};
}

SchrodingerState::~SchrodingerState() {
    std::free(v_);
}

SchrodingerState::SchrodingerState(SchrodingerState&& other) noexcept
    : p_x(other.p_x),
      p_z(other.p_z),
      gamma(other.gamma),
      active_k(other.active_k),
      meas_record(std::move(other.meas_record)),
      det_record(std::move(other.det_record)),
      obs_record(std::move(other.obs_record)),
      v_(other.v_),
      array_size_(other.array_size_),
      peak_rank_(other.peak_rank_),
      rng_(std::move(other.rng_)) {
    other.v_ = nullptr;
    other.array_size_ = 0;
}

SchrodingerState& SchrodingerState::operator=(SchrodingerState&& other) noexcept {
    if (this != &other) {
        std::free(v_);
        v_ = other.v_;
        array_size_ = other.array_size_;
        peak_rank_ = other.peak_rank_;
        rng_ = std::move(other.rng_);
        p_x = other.p_x;
        p_z = other.p_z;
        gamma = other.gamma;
        active_k = other.active_k;
        meas_record = std::move(other.meas_record);
        det_record = std::move(other.det_record);
        obs_record = std::move(other.obs_record);
        other.v_ = nullptr;
        other.array_size_ = 0;
    }
    return *this;
}

void SchrodingerState::reset(uint64_t seed) {
    v_[0] = {1.0, 0.0};
    p_x = 0;
    p_z = 0;
    gamma = {1.0, 0.0};
    active_k = 0;

    if (!obs_record.empty()) {
        std::fill(obs_record.begin(), obs_record.end(), 0);
    }

    rng_.seed(seed);
}

// =============================================================================
// SVM Execution
// =============================================================================

void execute(const CompiledModule& program, SchrodingerState& state) {
    // TODO: Implement RISC opcode dispatch (Phase 1, Task 1.3)
    // Iterate over program.bytecode and dispatch each opcode.
    (void)program;
    (void)state;
}

SampleResult sample(const CompiledModule& program, uint32_t shots, uint64_t seed) {
    SampleResult result;
    if (shots == 0) {
        return result;
    }

    uint32_t num_meas = program.num_measurements;
    uint32_t num_det = program.num_detectors;
    uint32_t num_obs = program.num_observables;

    result.measurements.resize(static_cast<size_t>(shots) * num_meas);
    result.detectors.resize(static_cast<size_t>(shots) * num_det);
    result.observables.resize(static_cast<size_t>(shots) * num_obs);

    SchrodingerState state(program.peak_rank, num_meas, num_det, num_obs, seed);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            state.reset(seed + shot);
        }

        execute(program, state);

        std::copy(state.meas_record.begin(), state.meas_record.end(),
                  result.measurements.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_meas));
        std::copy(
            state.det_record.begin(), state.det_record.end(),
            result.detectors.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_det));
        std::copy(state.obs_record.begin(), state.obs_record.end(),
                  result.observables.begin() +
                      static_cast<ptrdiff_t>(static_cast<size_t>(shot) * num_obs));
    }

    return result;
}

// =============================================================================
// Statevector Expansion
// =============================================================================

std::vector<std::complex<double>> get_statevector(const SchrodingerState& state,
                                                  const ConstantPool& pool) {
    // TODO: Implement factored state expansion (Phase 2, Task 2.1)
    // 1. Expand 2^k elements of |phi>_A into dense 2^n array
    // 2. Apply Pauli frame P (using p_x, p_z)
    // 3. Apply U_C (final_tableau) via Stim VectorSimulator
    // 4. Multiply by gamma
    (void)state;
    (void)pool;
    return {};
}

}  // namespace ucc
