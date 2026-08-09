#pragma once

#include "clifft/sampling/fused_rotation.h"

#include <memory>

namespace clifft::sampling {

using FusedRotationKernel = void (*)(State&, const PreparedFusedRotation&, const void*) noexcept;

// Type-erases an optional host-specific descriptor. The scalar descriptor
// remains portable, and common code never names or embeds SIMD vector types.
struct FusedRotationSidecar {
    std::shared_ptr<const void> storage;
    FusedRotationKernel kernel = nullptr;

    [[nodiscard]] explicit operator bool() const noexcept {
        return storage != nullptr && kernel != nullptr;
    }
};

// This function is linked only on x86-64 runtime-dispatch builds and must be
// called only after the dispatcher has selected the AVX-512 implementation.
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(
    const PreparedFusedRotation& rotation);

}  // namespace clifft::sampling
