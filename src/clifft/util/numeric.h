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

// Make a binary64 representation opaque to floating-point value propagation.
// Apple Clang 17 on arm64 was observed to fold exponent checks derived directly
// from a double when -ffast-math asserted that the source must be finite.
namespace detail {

inline uint64_t opaque_binary64_bits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__EMSCRIPTEN__)
    // An empty register barrier emits no instructions while making the value
    // opaque to the optimizer.
    __asm__ __volatile__("" : "+r"(bits));
#else
    volatile uint64_t opaque_bits = bits;
    bits = opaque_bits;
#endif
    return bits;
}

}  // namespace detail

// -ffast-math implies -ffinite-math-only, which can fold away
// std::isfinite() and NaN-aware comparisons. Inspect the exponent bits instead:
// a non-finite double has all exponent bits set.
inline bool is_finite_robust(double value) {
    const uint64_t bits = detail::opaque_binary64_bits(value);
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
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

// Compare binary64 magnitudes so -ffast-math cannot assume that NaN and
// infinity inputs are impossible. IEEE 754 ordering matches unsigned integer
// ordering for nonnegative finite values; preserve the usual acceptance of
// negative zero as a probability.
inline bool is_probability(double value) {
    const uint64_t bits = detail::opaque_binary64_bits(value);
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    constexpr uint64_t kMagnitudeMask = ~kSignMask;
    constexpr uint64_t kOneBits = 0x3FF0000000000000ULL;
    const uint64_t magnitude = bits & kMagnitudeMask;
    return magnitude == 0 || ((bits & kSignMask) == 0 && magnitude <= kOneBits);
}

}  // namespace clifft
