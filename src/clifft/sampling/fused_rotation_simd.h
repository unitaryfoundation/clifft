#pragma once

#include "clifft/sampling/fused_rotation_dispatch.h"

namespace clifft::sampling {

// These functions are linked only on x86-64 runtime-dispatch builds and must
// be called only after the dispatcher has selected their corresponding ISA.
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx2_sidecar(
    const PreparedFusedRotation& rotation);
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(
    const PreparedFusedRotation& rotation);

}  // namespace clifft::sampling
