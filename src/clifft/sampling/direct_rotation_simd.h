#pragma once

#include "clifft/sampling/direct_rotation_dispatch.h"

#include <cstdint>

namespace clifft::sampling {

// Lane count of one 256-bit double vector. The portable selector uses this
// value without exposing vector types outside the AVX2 translation unit.
inline constexpr uint64_t kAvx2DoubleLanes = 4;

// Lane count of one 512-bit double vector. Both the portable kernel selector
// and the AVX-512 translation unit measure shape eligibility against it.
inline constexpr uint64_t kAvx512DoubleLanes = 8;

// These functions are linked only on x86-64 runtime-dispatch builds and must
// be called only after the dispatcher has selected their corresponding ISA.
void apply_direct_rotation_avx2(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
