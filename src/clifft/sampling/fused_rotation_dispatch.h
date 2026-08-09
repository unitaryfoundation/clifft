#pragma once

#include "clifft/sampling/fused_rotation.h"

#include <memory>

namespace clifft::internal {
enum class RuntimeIsa;
}

namespace clifft::sampling {

using FusedRotationKernel = void (*)(State&, const PreparedFusedRotation&, const void*) noexcept;

// Type-erases optional host-specific preparation without exposing vector types
// to the portable executable plan.
struct FusedRotationSidecar {
    std::shared_ptr<const void> storage;
    FusedRotationKernel kernel = nullptr;

    [[nodiscard]] explicit operator bool() const noexcept {
        return storage != nullptr && kernel != nullptr;
    }
};

// Owns the portable fused descriptor and any optional host-specific
// preparation selected for it. Construction happens before hot execution.
class PreparedFusedRotationExecution {
  public:
    PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                   internal::RuntimeIsa runtime_isa);

    void apply(State& state) const noexcept {
        if (sidecar_) {
            sidecar_.kernel(state, rotation_, sidecar_.storage.get());
        } else {
            apply_fused_rotation(state, rotation_);
        }
    }

  private:
    PreparedFusedRotation rotation_;
    FusedRotationSidecar sidecar_;
};

}  // namespace clifft::sampling
