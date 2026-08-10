#pragma once

#include "clifft/sampling/direct_rotation_dispatch.h"

namespace clifft::sampling {

// This function is linked only on x86-64 runtime-dispatch builds and must be
// called only after the dispatcher has selected AVX-512.
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
