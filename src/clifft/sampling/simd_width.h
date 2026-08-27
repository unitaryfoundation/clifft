#pragma once

#include <bit>
#include <cstdint>

namespace clifft::sampling {

// Number of double lanes in one 128-bit Apple arm64 NEON vector.
inline constexpr uint64_t kNeonDoubleLanes = 2;
inline constexpr uint32_t kNeonLaneIndexBits = std::countr_zero(kNeonDoubleLanes);

// Number of double lanes in one 256-bit vector. Portable selectors use this
// value without exposing vector types outside ISA-specific translation units.
inline constexpr uint64_t kAvx2DoubleLanes = 4;
inline constexpr uint32_t kAvx2LaneIndexBits = std::countr_zero(kAvx2DoubleLanes);

// Number of double lanes in one 512-bit vector. Portable selectors use this
// value without exposing vector types outside ISA-specific translation units.
inline constexpr uint64_t kAvx512DoubleLanes = 8;
inline constexpr uint32_t kAvx512LaneIndexBits = std::countr_zero(kAvx512DoubleLanes);

}  // namespace clifft::sampling
