#pragma once

#include "clifft/sampling/kernels.h"

#include <cstdint>

namespace clifft::sampling {

// The selector fits in ExecuteRotation's existing tail padding. Architecture-
// specific data stays in its translation unit instead of expanding every
// direct-rotation descriptor.
enum class DirectRotationKernel : uint8_t {
    Scalar,
    Avx512Diagonal,
    Avx512HighPivot,
};

static_assert(sizeof(DirectRotationKernel) == 1);

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
    // Pivot-four sweeps regress relative to scalar at every measured active
    // width, unlike both the pivot-three boundary and pivots five and above.
    return pairing_bit >= (uint64_t{1} << 3) && pairing_bit != (uint64_t{1} << 4)
               ? DirectRotationKernel::Avx512HighPivot
               : DirectRotationKernel::Scalar;
}

// This function is linked only on x86-64 runtime-dispatch builds and must be
// called only after the dispatcher has selected AVX-512.
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;

}  // namespace clifft::sampling
