#pragma once

#include "clifft/sampling/fused_rotation_dispatch.h"

namespace clifft::sampling {

// This function is linked only on x86-64 runtime-dispatch builds and must be
// called only after the dispatcher has selected the AVX-512 implementation.
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(
    const PreparedFusedRotation& rotation);

}  // namespace clifft::sampling
