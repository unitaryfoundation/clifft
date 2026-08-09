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
    Avx512Diagonal,
    Avx512HighPivot,
};

static_assert(sizeof(DirectRotationKernel) == 1);

// Exposed separately from host ISA resolution so portable tests can verify the
// structural eligibility boundaries even on machines without AVX-512.
[[nodiscard]] inline DirectRotationKernel select_direct_rotation_kernel(
    const PreparedRotation& rotation, bool use_avx512) noexcept {
    if (!use_avx512 || rotation.pauli.is_identity()) {
        return DirectRotationKernel::Scalar;
    }
    if (rotation.pauli.is_diagonal()) {
        return rotation.pauli.active_width >= 3 ? DirectRotationKernel::Avx512Diagonal
                                                : DirectRotationKernel::Scalar;
    }
    const uint64_t pairing_bit = rotation.pauli.pair_selector;
    // On the current Zen 4 performance host, pivot-four sweeps regress at every
    // measured active width, unlike both the pivot-three boundary and pivots
    // five and above.
    return pairing_bit >= (uint64_t{1} << 3) && pairing_bit != (uint64_t{1} << 4)
               ? DirectRotationKernel::Avx512HighPivot
               : DirectRotationKernel::Scalar;
}

[[nodiscard]] DirectRotationKernel resolve_direct_rotation_kernel(
    const PreparedRotation& rotation, internal::RuntimeIsa runtime_isa) noexcept;

// Keeps ISA selection out of the executor while retaining the scalar kernel as
// the portable fallback and correctness implementation.
void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
