#pragma once

// Numeric checks that must remain valid under Release -ffast-math.

#include <cstdint>
#include <cstring>
#include <limits>

namespace clifft {

// Dense active states contain 2^k complex<double> coefficients. At k=60 the
// byte size no longer fits in a 64-bit size_t, so every dense executor and its
// planner must reject that width before allocation or bit-index arithmetic.
inline constexpr uint32_t kDenseActiveWidthLimit = 60;

// Relative epsilon shared by sampling backends when analytically-zero branch
// probabilities contain floating-point dust. This is part of Clifft's record
// reachability semantics, including forced-outcome replay.
inline constexpr double kMeasurementDustEpsilon = 1e-18;

// The IEEE 754 bit trick below assumes that layout. Make it explicit.
static_assert(std::numeric_limits<double>::is_iec559,
              "Clifft probability validation requires IEEE 754 doubles");

// -ffast-math implies -ffinite-math-only, which can fold away
// std::isfinite() and NaN-aware comparisons. Inspect the exponent bits
// instead: a non-finite double has all exponent bits set.
inline bool is_finite_robust(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
}

// The finite check must precede the comparisons: under -ffast-math the
// compiler may otherwise assume a NaN input is impossible.
inline bool is_probability(double value) {
    return is_finite_robust(value) && value >= 0.0 && value <= 1.0;
}

}  // namespace clifft
