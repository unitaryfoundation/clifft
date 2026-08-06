#include "clifft/svm/svm.h"
#include "clifft/svm/svm_internal.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace clifft {

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(StateConfig cfg) : peak_rank_(cfg.peak_rank), rng_(0) {
    uint32_t peak_rank = cfg.peak_rank;
    if (peak_rank >= kDenseActiveWidthLimit) {
        throw std::invalid_argument(
            "peak_rank >= " + std::to_string(kDenseActiveWidthLimit) +
            " would overflow the amplitude array's byte size (2^peak_rank * 16 must fit in "
            "size_t); peak_rank " +
            std::to_string(peak_rank) + " is too large");
    }
    if (cfg.seed.has_value()) {
        rng_.seed(*cfg.seed);
    } else {
        rng_.seed_from_entropy();
    }
    meas_record.resize(cfg.num_measurements, 0);
    det_record.resize(cfg.num_detectors, 0);
    obs_record.resize(cfg.num_observables, 0);
    exp_vals.resize(cfg.num_exp_vals, 0.0);
    has_exp_vals = (cfg.num_exp_vals > 0);

    // Pauli frame is sized to ceil(num_qubits / 64) words. Fall back to the
    // peak_rank-derived width when num_qubits is unspecified -- tests that
    // construct SchrodingerState directly (without going through trace/lower)
    // use axes within the active region, so peak_rank is a safe upper bound.
    num_qubits = (cfg.num_qubits > 0) ? cfg.num_qubits : std::max(peak_rank, uint32_t{1});
    const size_t num_words = (num_qubits + 63) / 64;
    p_x.assign(num_words, 0);
    p_z.assign(num_words, 0);

    allocate_array(peak_rank);
    v()[0] = {1.0, 0.0};
}

void SchrodingerState::allocate_array(uint32_t peak_rank) {
    const uint64_t new_array_size = uint64_t{1} << peak_rank;
    PageAlignedAllocation allocation(new_array_size * sizeof(std::complex<double>));
    auto* values = static_cast<std::complex<double>*>(allocation.data());

    // Anonymous mappings are zero-filled by the kernel. Portable allocation
    // fallbacks need explicit zeroing. Parallelizing the fill
    // distributes physical pages across NUMA nodes via first-touch policy,
    // so later OpenMP worker threads access local memory.
    if (!allocation.zero_initialized()) {
        const int64_t n = static_cast<int64_t>(new_array_size);
        if (peak_rank >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                values[i] = {0.0, 0.0};
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                values[i] = {0.0, 0.0};
            }
        }
    }
    v_allocation_ = std::move(allocation);
    array_size_ = new_array_size;
    peak_rank_ = peak_rank;
}

void SchrodingerState::grow_for_continuation(uint32_t peak_rank) {
    assert(pending_trap.has_value() &&
           "the amplitude array may grow only at the trap boundary, under a pending trap");
    if (peak_rank >= kDenseActiveWidthLimit) {
        throw std::invalid_argument(
            "peak_rank >= " + std::to_string(kDenseActiveWidthLimit) +
            " would overflow the amplitude array's byte size (2^peak_rank * 16 must fit in "
            "size_t); peak_rank " +
            std::to_string(peak_rank) + " is too large");
    }
    if ((1ULL << peak_rank) <= array_size_) {
        return;
    }

    // Save the number of active amplitudes before replacing the buffer.
    const uint64_t live = v_size();

    // Allocate first so a failure leaves the existing state unchanged.
    const uint64_t new_array_size = uint64_t{1} << peak_rank;
    PageAlignedAllocation allocation(new_array_size * sizeof(std::complex<double>));
    auto* new_values = static_cast<std::complex<double>*>(allocation.data());

    // Anonymous mappings are already zero. Clear a portable fallback in full
    // because the continuation may use entries above the active region.
    if (!allocation.zero_initialized()) {
        const int64_t n = static_cast<int64_t>(new_array_size);
        if (peak_rank >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                new_values[i] = {0.0, 0.0};
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                new_values[i] = {0.0, 0.0};
            }
        }
    }

    // Copy the live prefix from the old buffer into the new one.
    std::memcpy(new_values, v(), live * sizeof(std::complex<double>));

    // Install the new allocation after all operations that can fail.
    v_allocation_ = std::move(allocation);
    array_size_ = new_array_size;
    peak_rank_ = peak_rank;
}

SchrodingerState::~SchrodingerState() = default;

SchrodingerState::SchrodingerState(SchrodingerState&& other) noexcept
    : p_x(std::move(other.p_x)),
      p_z(std::move(other.p_z)),
      num_qubits(other.num_qubits),
      active_k(other.active_k),
      discarded(other.discarded),
      has_exp_vals(other.has_exp_vals),
      meas_record(std::move(other.meas_record)),
      det_record(std::move(other.det_record)),
      obs_record(std::move(other.obs_record)),
      next_noise_idx(other.next_noise_idx),
      forced_faults(std::move(other.forced_faults)),
      dust_clamps(other.dust_clamps),
      gamma_(other.gamma_),
      v_allocation_(std::move(other.v_allocation_)),
      array_size_(other.array_size_),
      peak_rank_(other.peak_rank_),
      rng_(std::move(other.rng_)),
      exp_vals(std::move(other.exp_vals)),
      pending_trap(other.pending_trap),
      forced_record(other.forced_record),
      forced_log_probability(other.forced_log_probability),
      forced_reachable(other.forced_reachable) {
    other.array_size_ = 0;
    other.active_k = 0;
    other.peak_rank_ = 0;
    other.pending_trap.reset();
    other.forced_record = {};
    other.forced_log_probability = 0.0;
    other.forced_reachable = true;
}

SchrodingerState& SchrodingerState::operator=(SchrodingerState&& other) noexcept {
    if (this != &other) {
        v_allocation_ = std::move(other.v_allocation_);
        array_size_ = other.array_size_;
        peak_rank_ = other.peak_rank_;
        rng_ = std::move(other.rng_);
        p_x = std::move(other.p_x);
        p_z = std::move(other.p_z);
        num_qubits = other.num_qubits;
        gamma_ = other.gamma_;
        active_k = other.active_k;
        discarded = other.discarded;
        has_exp_vals = other.has_exp_vals;
        next_noise_idx = other.next_noise_idx;
        forced_faults = std::move(other.forced_faults);
        dust_clamps = other.dust_clamps;
        meas_record = std::move(other.meas_record);
        det_record = std::move(other.det_record);
        obs_record = std::move(other.obs_record);
        exp_vals = std::move(other.exp_vals);
        pending_trap = other.pending_trap;
        forced_record = other.forced_record;
        forced_log_probability = other.forced_log_probability;
        forced_reachable = other.forced_reachable;
        other.array_size_ = 0;
        other.active_k = 0;
        other.peak_rank_ = 0;
        other.pending_trap.reset();
        other.forced_record = {};
        other.forced_log_probability = 0.0;
        other.forced_reachable = true;
    }
    return *this;
}

void SchrodingerState::reset() {
    uint64_t active_size = (active_k > 0) ? (uint64_t{1} << active_k) : 1;
    const int64_t n = static_cast<int64_t>(active_size);
    std::complex<double>* values = v();
    if (active_k >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            values[i] = {0.0, 0.0};
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            values[i] = {0.0, 0.0};
        }
    }
    values[0] = {1.0, 0.0};
    std::fill(p_x.begin(), p_x.end(), 0);
    std::fill(p_z.begin(), p_z.end(), 0);
    gamma_ = {1.0, 0.0};
    active_k = 0;

    // Discarded and trapped shots exit before writing complete records. Clear
    // their partial measurement and detector data before reusing the state.
    if (discarded || pending_trap.has_value()) {
        std::fill(meas_record.begin(), meas_record.end(), 0);
        std::fill(det_record.begin(), det_record.end(), 0);
    }
    discarded = false;
    pending_trap.reset();

    // obs_record uses ^= accumulation and must always be cleared.
    std::fill(obs_record.begin(), obs_record.end(), 0);

    // exp_vals are written per-shot; zero for the next shot.
    if (has_exp_vals)
        std::fill(exp_vals.begin(), exp_vals.end(), 0.0);

    // Reset forced-fault cursors (vectors are refilled per shot externally).
    forced_faults.noise_pos = 0;
    forced_faults.readout_pos = 0;

    // Forced-execution state: span cleared, accumulator zeroed, reachable
    // back to true. Dormant in sampling mode; the forced path sets these
    // per record before calling execute().
    forced_record = {};
    forced_log_probability = 0.0;
    forced_reachable = true;

    // PRNG is NOT reseeded -- it streams forward naturally across shots.
}

}  // namespace clifft
