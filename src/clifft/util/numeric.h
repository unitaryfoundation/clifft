#pragma once

// Numeric checks that must remain valid under Release -ffast-math.

#include <cstdint>
#include <cstring>
#include <limits>

namespace clifft {

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

}  // namespace clifft
