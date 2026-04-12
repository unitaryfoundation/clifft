#pragma once

#include "clifft/svm/svm.h"

#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace clifft {

// =============================================================================
// Platform-portable aligned memory allocation
// =============================================================================

inline void* aligned_alloc_portable(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free_portable(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// =============================================================================
// Measurement branch sampling with IEEE-754 dust clamping
// =============================================================================

// Relative epsilon for detecting floating-point dust in measurement
// probabilities. Squared amplitudes from analytically-zero Clifford+T
// interference sit around 1e-30 to 1e-24; this threshold safely swallows
// that dust while preserving genuine probabilities (e.g. R_ZZ angles
// producing probabilities ~1e-16).
inline constexpr double kDustEpsilon = 1e-18;

// Minimum active_k for the AVX2 3D waterfall loops in 2-qubit gates.
// Below this rank the array fits in L1 cache and the flat pdep loop
// has less setup overhead; above it the structured stride pattern
// lets the hardware prefetcher hide main-memory latency.
inline constexpr uint16_t kMinRankFor3DLoop = 9;

// Minimum active_k for OpenMP multi-threading of array sweeps.
// At k=18 the statevector is 4 MB (256K complex doubles), which spills
// out of L3 cache on most CPUs. Below this threshold, a single AVX-512
// core processes the entire array in microseconds and thread wake-up
// overhead would dominate. The OpenMP `if()` clause uses this to bypass
// the thread dispatcher entirely for near-Clifford workloads.
inline constexpr uint16_t kMinRankForThreads = 18;

// Sample a binary measurement outcome from two branch probabilities,
// clamping IEEE-754 dust to avoid spurious PRNG rolls. Returns 0 if
// prob0 wins, 1 if prob1 wins. Deterministic when one branch is dust.
inline uint8_t sample_branch(SchrodingerState& state, double prob0, double prob1, double total) {
    double eps = kDustEpsilon * total;
    if (prob1 <= eps) {
        if (prob1 > 0.0)
            state.dust_clamps++;
        return 0;
    }
    if (prob0 <= eps) {
        if (prob0 > 0.0)
            state.dust_clamps++;
        return 1;
    }
    double rand = state.random_double();
    return (rand * total < prob0) ? 0 : 1;
}

// =============================================================================
// OpenMP Threading Helpers
// =============================================================================

// Simple parallel for: runs `kernel(0), kernel(1), ..., kernel(n-1)` in
// parallel when active_k >= kMinRankForThreads. Uses a C++ if-guard to
// avoid entering the OpenMP runtime (and its per-call futex overhead)
// for sub-threshold ranks.
template <typename KernelFunc>
inline void parallel_for(int64_t n, uint16_t active_k, KernelFunc&& kernel) {
    if (active_k >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            kernel(i);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            kernel(i);
        }
    }
}

// Parallel reduction with 2 accumulators: runs `kernel(i, acc1, acc2)` for
// i in [0, n) and sums acc1/acc2 across threads via OpenMP reduction.
// Uses local copies to avoid MSVC OpenMP 2.0 error C3030 (reduction on
// reference types is not supported).
template <typename KernelFunc>
inline void parallel_reduce(int64_t n, uint16_t active_k, double& acc1, double& acc2,
                            KernelFunc&& kernel) {
    if (active_k >= kMinRankForThreads) {
        double local1 = acc1, local2 = acc2;
#pragma omp parallel for schedule(static) reduction(+ : local1, local2)
        for (int64_t i = 0; i < n; ++i) {
            kernel(i, local1, local2);
        }
        acc1 = local1;
        acc2 = local2;
    } else {
        for (int64_t i = 0; i < n; ++i) {
            kernel(i, acc1, acc2);
        }
    }
}

// Parallel reduction with 3 accumulators.
template <typename KernelFunc>
inline void parallel_reduce(int64_t n, uint16_t active_k, double& acc1, double& acc2, double& acc3,
                            KernelFunc&& kernel) {
    if (active_k >= kMinRankForThreads) {
        double local1 = acc1, local2 = acc2, local3 = acc3;
#pragma omp parallel for schedule(static) reduction(+ : local1, local2, local3)
        for (int64_t i = 0; i < n; ++i) {
            kernel(i, local1, local2, local3);
        }
        acc1 = local1;
        acc2 = local2;
        acc3 = local3;
    } else {
        for (int64_t i = 0; i < n; ++i) {
            kernel(i, acc1, acc2, acc3);
        }
    }
}

// Parallel flat 1D loop: threads a simple `for (i = 0; i < count; i += stride)`
// loop when active_k is large enough to benefit from multi-threading.
// The kernel receives the loop index.
template <typename KernelFunc>
inline void parallel_flat_loop(uint64_t count, uint64_t stride, uint16_t active_k,
                               KernelFunc&& kernel) {
    int64_t n = static_cast<int64_t>(count / stride);
    if (active_k >= kMinRankForThreads) {
#pragma omp parallel for schedule(static)
        for (int64_t idx = 0; idx < n; ++idx) {
            kernel(static_cast<uint64_t>(idx) * stride);
        }
    } else {
        for (int64_t idx = 0; idx < n; ++idx) {
            kernel(static_cast<uint64_t>(idx) * stride);
        }
    }
}

// Parallel stride loop for 2D waterfall patterns (outer + inner).
// Parallelizes the outermost dimension that has enough iterations (>=4),
// guaranteeing the thread pool spawns at most once when outer_n >= 4.
// Falls back to parallelizing the inner loop when the outer has too few
// iterations (spawns at most outer_n <= 3 times).
//
// The kernel receives (outer_base, inner_offset) corresponding to the
// start of the outer block and the offset within it.
template <typename KernelFunc>
inline void parallel_stride_loop(uint64_t total, uint64_t step, uint64_t inner_stride,
                                 uint16_t active_k, KernelFunc&& kernel) {
    int64_t outer_n = static_cast<int64_t>(total / (2 * step));
    int64_t inner_n = static_cast<int64_t>(step / inner_stride);

    if (active_k >= kMinRankForThreads) {
        if (outer_n >= 4) {
#pragma omp parallel for schedule(static)
            for (int64_t oi = 0; oi < outer_n; ++oi) {
                uint64_t outer = static_cast<uint64_t>(oi) * 2 * step;
                for (uint64_t k = 0; k < step; k += inner_stride) {
                    kernel(outer, k);
                }
            }
        } else {
            for (int64_t oi = 0; oi < outer_n; ++oi) {
                uint64_t outer = static_cast<uint64_t>(oi) * 2 * step;
#pragma omp parallel for schedule(static)
                for (int64_t ki = 0; ki < inner_n; ++ki) {
                    kernel(outer, static_cast<uint64_t>(ki) * inner_stride);
                }
            }
        }
    } else {
        for (int64_t oi = 0; oi < outer_n; ++oi) {
            uint64_t outer = static_cast<uint64_t>(oi) * 2 * step;
            for (uint64_t k = 0; k < step; k += inner_stride) {
                kernel(outer, k);
            }
        }
    }
}

// Parallel 3D stride loop for 2-qubit gates with two nested stride dimensions.
// Iterates over all blocks where both axis bits are zero:
//   for (i in 0..total by 2*step_hi)
//     for (j in 0..step_hi by 2*step_lo)
//       for (k in 0..step_lo by inner_stride)
//         kernel(i + j, k)
//
// Parallelizes the outermost dimension that has enough iterations.
template <typename KernelFunc>
inline void parallel_3d_stride_loop(uint64_t total, uint64_t step_hi, uint64_t step_lo,
                                    uint64_t inner_stride, uint16_t active_k, KernelFunc&& kernel) {
    int64_t outer_n = static_cast<int64_t>(total / (2 * step_hi));
    int64_t mid_n = static_cast<int64_t>(step_hi / (2 * step_lo));
    int64_t inner_n = static_cast<int64_t>(step_lo / inner_stride);

    if (active_k >= kMinRankForThreads) {
        if (outer_n >= 4) {
#pragma omp parallel for schedule(static)
            for (int64_t oi = 0; oi < outer_n; ++oi) {
                uint64_t i = static_cast<uint64_t>(oi) * 2 * step_hi;
                for (int64_t mi = 0; mi < mid_n; ++mi) {
                    uint64_t base = i + static_cast<uint64_t>(mi) * 2 * step_lo;
                    for (int64_t ki = 0; ki < inner_n; ++ki) {
                        kernel(base, static_cast<uint64_t>(ki) * inner_stride);
                    }
                }
            }
        } else if (mid_n >= 4) {
            for (int64_t oi = 0; oi < outer_n; ++oi) {
                uint64_t i = static_cast<uint64_t>(oi) * 2 * step_hi;
#pragma omp parallel for schedule(static)
                for (int64_t mi = 0; mi < mid_n; ++mi) {
                    uint64_t base = i + static_cast<uint64_t>(mi) * 2 * step_lo;
                    for (int64_t ki = 0; ki < inner_n; ++ki) {
                        kernel(base, static_cast<uint64_t>(ki) * inner_stride);
                    }
                }
            }
        } else {
            for (int64_t oi = 0; oi < outer_n; ++oi) {
                uint64_t i = static_cast<uint64_t>(oi) * 2 * step_hi;
                for (int64_t mi = 0; mi < mid_n; ++mi) {
                    uint64_t base = i + static_cast<uint64_t>(mi) * 2 * step_lo;
#pragma omp parallel for schedule(static)
                    for (int64_t ki = 0; ki < inner_n; ++ki) {
                        kernel(base, static_cast<uint64_t>(ki) * inner_stride);
                    }
                }
            }
        }
    } else {
        for (int64_t oi = 0; oi < outer_n; ++oi) {
            uint64_t i = static_cast<uint64_t>(oi) * 2 * step_hi;
            for (int64_t mi = 0; mi < mid_n; ++mi) {
                uint64_t base = i + static_cast<uint64_t>(mi) * 2 * step_lo;
                for (int64_t ki = 0; ki < inner_n; ++ki) {
                    kernel(base, static_cast<uint64_t>(ki) * inner_stride);
                }
            }
        }
    }
}

}  // namespace clifft
