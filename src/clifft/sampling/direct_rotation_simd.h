#pragma once

#include "clifft/sampling/direct_rotation_dispatch.h"
#include "clifft/sampling/simd_width.h"

namespace clifft::sampling {

// These functions are linked only on x86-64 runtime-dispatch builds and must
// be called only after the dispatcher has selected their corresponding ISA.
void apply_direct_rotation_avx2(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
