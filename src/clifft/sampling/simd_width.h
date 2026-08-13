#pragma once

#include <cstdint>

namespace clifft::sampling {

// Architecture-neutral lane counts used to select SIMD kernels without
// exposing vector types outside their ISA-specific translation units.
inline constexpr uint64_t kAvx2DoubleLanes = 4;
inline constexpr uint64_t kAvx512DoubleLanes = 8;

}  // namespace clifft::sampling
