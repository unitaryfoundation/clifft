#pragma once

// Numeric checks that must remain valid under Release -ffast-math.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace clifft {

// Dense active states contain 2^k complex<double> coefficients. At k=60 the
// byte size no longer fits in a 64-bit size_t, so every dense executor and its
// planner must reject that width before allocation or bit-index arithmetic.
inline constexpr uint32_t kDenseActiveWidthLimit = 60;

// Relative epsilon shared by measurement-branch handling and instrument
// sampling when analytically-zero probabilities contain floating-point dust.
// This is part of Clifft's record-reachability semantics.
inline constexpr double kMeasurementDustEpsilon = 1e-18;

// Absolute tolerance in half-turn units for replacing a rotation with its
// canonical Clifford representative. This intentionally absorbs numerical
// noise from circuit generation and serialization, so it is part of the
// compiler's approximation policy rather than a floating-point implementation
// detail.
inline constexpr double kRotationCanonicalizationTolerance = 1e-12;

enum class CliffordRotation : uint8_t { IDENTITY = 0, SQRT = 1, PAULI = 2, SQRT_DAG = 3 };

// The IEEE 754 bit trick below assumes that layout. Make it explicit.
static_assert(std::numeric_limits<double>::is_iec559,
              "Clifft probability validation requires IEEE 754 doubles");

// -ffast-math implies -ffinite-math-only, which can fold away
// std::isfinite() and NaN-aware comparisons. Inspect the exponent bits
// instead: a non-finite double has all exponent bits set. Keep the integer
// predicate out of line so the compiler cannot propagate finite-math
// assumptions from the floating-point value into the bit test.
namespace detail {

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
inline bool
binary64_bits_are_finite(uint64_t bits) {
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
}

}  // namespace detail

inline bool is_finite_robust(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return detail::binary64_bits_are_finite(bits);
}

// Return the Clifford representative when alpha is within the shared absolute
// tolerance of a multiple of 0.5 half-turns. Reducing modulo two before any
// scaling also keeps the calculation finite for every finite binary64 input.
inline std::optional<CliffordRotation> classify_clifford_rotation(double alpha) {
    if (!is_finite_robust(alpha)) {
        return std::nullopt;
    }

    double reduced = std::fmod(alpha, 2.0);
    if (reduced < 0.0) {
        reduced += 2.0;
    }

    const double nearest_step = std::round(2.0 * reduced);
    if (std::abs(reduced - 0.5 * nearest_step) >= kRotationCanonicalizationTolerance) {
        return std::nullopt;
    }

    const auto residue = static_cast<uint32_t>(nearest_step) & 3U;
    return static_cast<CliffordRotation>(residue);
}

// The finite check must precede the comparisons: under -ffast-math the
// compiler may otherwise assume a NaN input is impossible.
inline bool is_probability(double value) {
    return is_finite_robust(value) && value >= 0.0 && value <= 1.0;
}

}  // namespace clifft
