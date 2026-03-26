#include "ucc/svm/svm.h"
#include "ucc/svm/svm_internal.h"

#include <algorithm>
#include <new>
#include <random>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace ucc {

// =============================================================================
// PRNG Entropy Seeding
// =============================================================================

void Xoshiro256PlusPlus::seed_from_entropy() {
    // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94087
    // See https://github.com/quantumlib/Stim/issues/26
#if defined(__linux__) && defined(__GLIBCXX__) && __GLIBCXX__ >= 20200128
    std::random_device rd("/dev/urandom");
#else
    std::random_device rd;
#endif
    auto rd64 = [&rd]() -> uint64_t { return (static_cast<uint64_t>(rd()) << 32) | rd(); };
    seed_full(rd64(), rd64(), rd64(), rd64());
}

// =============================================================================
// SchrodingerState Implementation
// =============================================================================

SchrodingerState::SchrodingerState(uint32_t peak_rank, uint32_t num_measurements,
                                   uint32_t num_detectors, uint32_t num_observables,
                                   std::optional<uint64_t> seed, uint32_t num_exp_vals)
    : peak_rank_(peak_rank), rng_(0) {
    if (peak_rank >= 63) {
        throw std::invalid_argument(
            "peak_rank >= 63 would cause undefined behavior in 1ULL << peak_rank");
    }
    if (seed.has_value()) {
        rng_.seed(*seed);
    } else {
        rng_.seed_from_entropy();
    }
    meas_record.resize(num_measurements, 0);
    det_record.resize(num_detectors, 0);
    obs_record.resize(num_observables, 0);
    exp_vals.resize(num_exp_vals, 0.0);

    array_size_ = 1ULL << peak_rank;
    size_t bytes = array_size_ * sizeof(std::complex<double>);
    // Round up to page boundary for mmap/aligned_alloc compatibility.
    size_t aligned_bytes = (bytes + 4095) & ~4095ULL;
    v_alloc_bytes_ = aligned_bytes;

#if defined(__linux__)
    // Try MAP_HUGETLB for 2MB huge pages (works without THP kernel support).
    // Only worthwhile for allocations >= 2MB.
    static constexpr size_t kHugePageSize = 2 * 1024 * 1024;
    if (aligned_bytes >= kHugePageSize) {
        size_t huge_aligned = (aligned_bytes + kHugePageSize - 1) & ~(kHugePageSize - 1);
        void* p = mmap(nullptr, huge_aligned, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
            v_ = static_cast<std::complex<double>*>(p);
            v_alloc_bytes_ = huge_aligned;
            v_is_mmap_ = true;
        }
    }

    if (!v_) {
        // Align to huge page boundary so madvise(MADV_HUGEPAGE) works on any
        // architecture (ARM64 may use 16KB or 64KB base pages).
        size_t alloc_bytes = (aligned_bytes >= kHugePageSize)
                                 ? ((aligned_bytes + kHugePageSize - 1) & ~(kHugePageSize - 1))
                                 : aligned_bytes;
        size_t alloc_align = (aligned_bytes >= kHugePageSize) ? kHugePageSize : 4096;
        v_ = static_cast<std::complex<double>*>(aligned_alloc_portable(alloc_align, alloc_bytes));
        if (!v_) {
            throw std::bad_alloc();
        }
        if (aligned_bytes >= kHugePageSize) {
            madvise(v_, alloc_bytes, MADV_HUGEPAGE);
        }
        v_alloc_bytes_ = alloc_bytes;
    }
#else
    if (!v_) {
        v_ = static_cast<std::complex<double>*>(aligned_alloc_portable(4096, aligned_bytes));
        if (!v_) {
            throw std::bad_alloc();
        }
    }
#endif

    // mmap(MAP_ANONYMOUS) guarantees zero-filled pages from the kernel.
    // Only aligned_alloc needs explicit zeroing. Avoid touching the full
    // allocation upfront -- it defeats demand paging and forces the OS to
    // commit every physical page, causing latency spikes at large peak_rank.
    if (!v_is_mmap_) {
        std::fill(v_, v_ + array_size_, std::complex<double>(0.0, 0.0));
    }
    v_[0] = {1.0, 0.0};
}

SchrodingerState::~SchrodingerState() {
#if defined(__linux__)
    if (v_is_mmap_) {
        munmap(v_, v_alloc_bytes_);
        return;
    }
#endif
    aligned_free_portable(v_);
}

SchrodingerState::SchrodingerState(SchrodingerState&& other) noexcept
    : p_x(other.p_x),
      p_z(other.p_z),
      active_k(other.active_k),
      discarded(other.discarded),
      meas_record(std::move(other.meas_record)),
      det_record(std::move(other.det_record)),
      obs_record(std::move(other.obs_record)),
      exp_vals(std::move(other.exp_vals)),
      next_noise_idx(other.next_noise_idx),
      forced_faults(std::move(other.forced_faults)),
      dust_clamps(other.dust_clamps),
      gamma_(other.gamma_),
      v_(other.v_),
      array_size_(other.array_size_),
      v_alloc_bytes_(other.v_alloc_bytes_),
      peak_rank_(other.peak_rank_),
      v_is_mmap_(other.v_is_mmap_),
      rng_(std::move(other.rng_)) {
    other.v_ = nullptr;
    other.array_size_ = 0;
    other.v_alloc_bytes_ = 0;
    other.v_is_mmap_ = false;
    other.active_k = 0;
    other.peak_rank_ = 0;
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
        p_x = other.p_x;
        p_z = other.p_z;
        gamma_ = other.gamma_;
        active_k = other.active_k;
        discarded = other.discarded;
        next_noise_idx = other.next_noise_idx;
        forced_faults = std::move(other.forced_faults);
        dust_clamps = other.dust_clamps;
        meas_record = std::move(other.meas_record);
        det_record = std::move(other.det_record);
        obs_record = std::move(other.obs_record);
        exp_vals = std::move(other.exp_vals);
        other.v_ = nullptr;
        other.array_size_ = 0;
        other.v_alloc_bytes_ = 0;
        other.v_is_mmap_ = false;
        other.active_k = 0;
        other.peak_rank_ = 0;
    }
    return *this;
}

void SchrodingerState::reset() {
    uint64_t active_size = (active_k > 0) ? (uint64_t{1} << active_k) : 1;
    std::fill(v_, v_ + active_size, std::complex<double>(0.0, 0.0));
    v_[0] = {1.0, 0.0};
    p_x = 0;
    p_z = 0;
    gamma_ = {1.0, 0.0};
    active_k = 0;

    // If the previous shot was discarded by OP_POSTSELECT, the bytecode
    // loop exited early, leaving meas_record and det_record with stale
    // data from the aborted shot. Zero them out to avoid garbage.
    if (discarded) {
        std::fill(meas_record.begin(), meas_record.end(), 0);
        std::fill(det_record.begin(), det_record.end(), 0);
    }
    discarded = false;

    // obs_record uses ^= accumulation and must always be cleared.
    std::fill(obs_record.begin(), obs_record.end(), 0);

    // exp_vals are written per-shot; zero for the next shot.
    std::fill(exp_vals.begin(), exp_vals.end(), 0.0);

    // Reset forced-fault cursors (vectors are refilled per shot externally).
    forced_faults.noise_pos = 0;
    forced_faults.readout_pos = 0;

    // PRNG is NOT reseeded -- it streams forward naturally across shots.
}

}  // namespace ucc
