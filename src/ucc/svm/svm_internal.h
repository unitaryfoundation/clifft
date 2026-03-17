#pragma once

#include "ucc/svm/svm.h"

#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace ucc {

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

}  // namespace ucc
