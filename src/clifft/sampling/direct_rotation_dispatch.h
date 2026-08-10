#pragma once

#include "clifft/sampling/kernels.h"

#include <cstdint>

namespace clifft::internal {
enum class RuntimeIsa;
}

namespace clifft::sampling {

// This selector occupies ExecuteRotation's existing tail padding. It carries
// no architecture-specific data and does not expand the action descriptor.
enum class DirectRotationKernel : uint8_t {
    Scalar,
    Diagonal,
    HighPivot,
};

static_assert(sizeof(DirectRotationKernel) == 1);

[[nodiscard]] DirectRotationKernel resolve_direct_rotation_kernel(
    const PreparedRotation& rotation, internal::RuntimeIsa runtime_isa) noexcept;

// Keeps ISA selection out of the executor while retaining the scalar kernel as
// the portable fallback and correctness implementation.
void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
