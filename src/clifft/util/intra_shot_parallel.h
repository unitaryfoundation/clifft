#pragma once

#include <cstdint>
#include <limits>

#if defined(CLIFFT_USE_OPENMP)
#include <omp.h>
#endif

namespace clifft {

#if defined(__GNUC__) || defined(__clang__)
#define CLIFFT_INTRA_SHOT_INLINE [[gnu::always_inline, gnu::flatten]] inline
#elif defined(_MSC_VER)
#define CLIFFT_INTRA_SHOT_INLINE __forceinline
#else
#define CLIFFT_INTRA_SHOT_INLINE inline
#endif

inline constexpr uint32_t kDefaultIntraShotMinActiveWidth = 18;

[[nodiscard]] inline constexpr bool intra_shot_parallelism_available() noexcept {
#if defined(CLIFFT_USE_OPENMP)
    return true;
#else
    return false;
#endif
}

[[nodiscard]] inline constexpr bool should_parallelize_intra_shot(
    uint32_t active_width, uint32_t workers, uint32_t min_active_width) noexcept {
    return intra_shot_parallelism_available() && workers > 1 && active_width >= min_active_width;
}

[[nodiscard]] inline bool openmp_process_binding_active() noexcept {
#if defined(CLIFFT_USE_OPENMP) && defined(_OPENMP) && _OPENMP >= 201307
    return omp_get_proc_bind() != omp_proc_bind_false;
#else
    return false;
#endif
}

// OpenMP owns the team lifetime, while explicit contiguous ranges keep every
// kernel's coefficient traversal deterministic and independent of runtime
// scheduling policy.
template <typename Kernel>
CLIFFT_INTRA_SHOT_INLINE void intra_shot_parallel_ranges(uint64_t count, uint32_t workers,
                                                         Kernel&& kernel) noexcept {
#if defined(CLIFFT_USE_OPENMP)
    const int requested_team_size =
        static_cast<int>(workers > static_cast<uint32_t>(std::numeric_limits<int>::max())
                             ? std::numeric_limits<int>::max()
                             : workers);
#pragma omp parallel num_threads(requested_team_size)
    {
        const uint64_t team_size = static_cast<uint64_t>(omp_get_num_threads());
        const uint64_t thread = static_cast<uint64_t>(omp_get_thread_num());
        const uint64_t range_size = count / team_size;
        const uint64_t extra = count % team_size;
        const uint64_t begin = thread * range_size + (thread < extra ? thread : extra);
        const uint64_t end = begin + range_size + (thread < extra ? 1 : 0);
        kernel(begin, end);
    }
#else
    static_cast<void>(workers);
    kernel(0, count);
#endif
}

#undef CLIFFT_INTRA_SHOT_INLINE

}  // namespace clifft
