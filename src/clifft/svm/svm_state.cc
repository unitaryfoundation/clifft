#include "clifft/svm/svm.h"
#include "clifft/svm/svm_internal.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace clifft {

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(StateConfig cfg) : peak_rank_(cfg.peak_rank), rng_(0) {
    uint32_t peak_rank = cfg.peak_rank;
    if (peak_rank >= 60) {
        throw std::invalid_argument(
            "peak_rank >= 60 would overflow the amplitude array's byte size (2^peak_rank * 16 "
            "must fit in size_t); peak_rank " +
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
    v_[0] = {1.0, 0.0};
}

// Allocate an amplitude array without changing SchrodingerState. mmap returns
// zero-filled memory; aligned_alloc does not. Callers copy these values into
// the state only after allocation and initialization succeed.
struct AllocResult {
    std::complex<double>* ptr;
    size_t alloc_bytes;
    bool is_mmap;
};

static AllocResult alloc_amplitude_array(uint32_t peak_rank) {
    const uint64_t array_size = 1ULL << peak_rank;
    const size_t bytes = array_size * sizeof(std::complex<double>);
    // Round up to page boundary for mmap/aligned_alloc compatibility.
    const size_t aligned_bytes = (bytes + 4095) & ~4095ULL;

    AllocResult result{nullptr, aligned_bytes, false};

#if defined(__linux__)
    // Try MAP_HUGETLB for 2MB huge pages (works without THP kernel support).
    // Only worthwhile for allocations >= 2MB.
    static constexpr size_t kHugePageSize = 2 * 1024 * 1024;
    if (aligned_bytes >= kHugePageSize) {
        const size_t huge_aligned = (aligned_bytes + kHugePageSize - 1) & ~(kHugePageSize - 1);
        void* p = mmap(nullptr, huge_aligned, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
            result.ptr = static_cast<std::complex<double>*>(p);
            result.alloc_bytes = huge_aligned;
            result.is_mmap = true;
        }
    }

    if (!result.ptr) {
        // Align to huge page boundary so madvise(MADV_HUGEPAGE) works on any
        // architecture (ARM64 may use 16KB or 64KB base pages).
        const size_t alloc_bytes =
            (aligned_bytes >= kHugePageSize)
                ? ((aligned_bytes + kHugePageSize - 1) & ~(kHugePageSize - 1))
                : aligned_bytes;
        const size_t alloc_align = (aligned_bytes >= kHugePageSize) ? kHugePageSize : 4096;
        result.ptr =
            static_cast<std::complex<double>*>(aligned_alloc_portable(alloc_align, alloc_bytes));
        if (!result.ptr) {
            throw std::bad_alloc();
        }
        if (aligned_bytes >= kHugePageSize) {
            madvise(result.ptr, alloc_bytes, MADV_HUGEPAGE);
        }
        result.alloc_bytes = alloc_bytes;
    }
#else
    result.ptr = static_cast<std::complex<double>*>(aligned_alloc_portable(4096, aligned_bytes));
    if (!result.ptr) {
        throw std::bad_alloc();
    }
#endif

    return result;
}

void SchrodingerState::allocate_array(uint32_t peak_rank) {
    const AllocResult alloc = alloc_amplitude_array(peak_rank);

    peak_rank_ = peak_rank;
    v_ = alloc.ptr;
    v_is_mmap_ = alloc.is_mmap;
    array_size_ = 1ULL << peak_rank;
    v_alloc_bytes_ = alloc.alloc_bytes;

    // mmap(MAP_ANONYMOUS) guarantees zero-filled pages from the kernel.
    // Only aligned_alloc needs explicit zeroing. Parallelizing the fill
    // distributes physical pages across NUMA nodes via first-touch policy,
    // so later OpenMP worker threads access local memory.
    if (!v_is_mmap_) {
        int64_t n = static_cast<int64_t>(array_size_);
        if (peak_rank_ >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                v_[i] = {0.0, 0.0};
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                v_[i] = {0.0, 0.0};
            }
        }
    }
}

void SchrodingerState::free_array() noexcept {
#if defined(__linux__)
    if (v_is_mmap_) {
        munmap(v_, v_alloc_bytes_);
        return;
    }
#endif
    aligned_free_portable(v_);
}

void SchrodingerState::grow_for_continuation(uint32_t peak_rank) {
    assert(pending_trap.has_value() &&
           "the amplitude array may grow only at the trap boundary, under a pending trap");
    if (peak_rank >= 60) {
        throw std::invalid_argument(
            "peak_rank >= 60 would overflow the amplitude array's byte size (2^peak_rank * 16 "
            "must fit in size_t); peak_rank " +
            std::to_string(peak_rank) + " is too large");
    }
    if ((1ULL << peak_rank) <= array_size_) {
        return;
    }

    // Save the number of active amplitudes before replacing the buffer.
    const uint64_t live = v_size();

    // Allocate first so a failure leaves the existing state unchanged.
    const AllocResult alloc = alloc_amplitude_array(peak_rank);

    // mmap memory is already zero. Clear the full aligned_alloc array because
    // the continuation may use entries above the currently active region.
    if (!alloc.is_mmap) {
        const uint64_t new_array_size = 1ULL << peak_rank;
        int64_t n = static_cast<int64_t>(new_array_size);
        if (peak_rank >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                alloc.ptr[i] = {0.0, 0.0};
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                alloc.ptr[i] = {0.0, 0.0};
            }
        }
    }

    // Copy the live prefix from the old buffer into the new one.
    std::memcpy(alloc.ptr, v_, live * sizeof(std::complex<double>));

    // Keep the old allocation details until the new buffer is installed.
    std::complex<double>* old_v = v_;
    const size_t old_bytes = v_alloc_bytes_;
    const bool old_mmap = v_is_mmap_;

    // Install the new allocation after all operations that can fail.
    v_ = alloc.ptr;
    v_alloc_bytes_ = alloc.alloc_bytes;
    v_is_mmap_ = alloc.is_mmap;
    array_size_ = 1ULL << peak_rank;
    peak_rank_ = peak_rank;

#if defined(__linux__)
    if (old_mmap) {
        munmap(old_v, old_bytes);
    } else {
        aligned_free_portable(old_v);
    }
#else
    (void)old_bytes;
    (void)old_mmap;
    aligned_free_portable(old_v);
#endif
}

SchrodingerState::~SchrodingerState() {
    free_array();
}

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
      v_(other.v_),
      array_size_(other.array_size_),
      v_alloc_bytes_(other.v_alloc_bytes_),
      peak_rank_(other.peak_rank_),
      v_is_mmap_(other.v_is_mmap_),
      rng_(std::move(other.rng_)),
      exp_vals(std::move(other.exp_vals)),
      pending_trap(other.pending_trap),
      forced_record(other.forced_record),
      forced_log_probability(other.forced_log_probability),
      forced_reachable(other.forced_reachable) {
    other.v_ = nullptr;
    other.array_size_ = 0;
    other.v_alloc_bytes_ = 0;
    other.v_is_mmap_ = false;
    other.active_k = 0;
    other.peak_rank_ = 0;
    other.pending_trap.reset();
    other.forced_record = {};
    other.forced_log_probability = 0.0;
    other.forced_reachable = true;
}

SchrodingerState& SchrodingerState::operator=(SchrodingerState&& other) noexcept {
    if (this != &other) {
#if defined(__linux__)
        if (v_is_mmap_) {
            munmap(v_, v_alloc_bytes_);
        } else {
            aligned_free_portable(v_);
        }
#else
        aligned_free_portable(v_);
#endif
        v_ = other.v_;
        array_size_ = other.array_size_;
        v_alloc_bytes_ = other.v_alloc_bytes_;
        v_is_mmap_ = other.v_is_mmap_;
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
        other.v_ = nullptr;
        other.array_size_ = 0;
        other.v_alloc_bytes_ = 0;
        other.v_is_mmap_ = false;
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
    int64_t n = static_cast<int64_t>(active_size);
    if (active_k >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            v_[i] = {0.0, 0.0};
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            v_[i] = {0.0, 0.0};
        }
    }
    v_[0] = {1.0, 0.0};
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
