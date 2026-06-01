#pragma once

// Shared numeric helpers for noncomputational model validation.

#include <cstdint>
#include <cstring>
#include <limits>

namespace clifft {

// The IEEE 754 bit tricks below assume that layout. Make it explicit.
static_assert(std::numeric_limits<double>::is_iec559, "clifft::noncomp requires IEEE 754 doubles");

// Tolerance for derived-quantity bounds (column sums, the
// initial-state sum, source-independence equality). Raw user-supplied
// matrix entries are checked strictly against [0, 1]; only quantities
// derived from them tolerate this much floating drift.
inline constexpr double kProbTolerance = 1e-12;

// Release builds use -ffast-math, which implies -ffinite-math-only.
// That folds away std::isfinite() and lets `v >= 0.0 && v <= 1.0`
// pass NaN through. Inspect the IEEE 754 bit pattern instead: a
// non-finite double has all exponent bits set.
inline bool is_finite_robust(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
}

}  // namespace clifft
