#include "clifft/sampling/direct_rotation_dispatch.h"

#include "clifft/sampling/direct_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

namespace {

DirectRotationKernel select_direct_rotation_avx512(const PreparedRotation& rotation) noexcept {
    if (rotation.pauli.is_identity()) {
        return DirectRotationKernel::Scalar;
    }
    if (rotation.pauli.is_diagonal()) {
        return rotation.pauli.active_width >= 3 ? DirectRotationKernel::Diagonal
                                                : DirectRotationKernel::Scalar;
    }
    const uint64_t pairing_bit = rotation.pauli.pair_selector;
    // Pivot four has a distinct stride-16 access pattern that regressed against
    // scalar at every measured width, so it remains on the fallback until a
    // kernel designed for that shape is available.
    return pairing_bit >= (uint64_t{1} << 3) && pairing_bit != (uint64_t{1} << 4)
               ? DirectRotationKernel::HighPivot
               : DirectRotationKernel::Scalar;
}

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

// The selected ISA is process-wide, while each action stores only a shape. A
// predicted branch and direct call are cheaper here than the SVM's function
// pointer pattern, which amortizes one indirect call over an entire program.
const internal::RuntimeIsa kResolvedDirectRotationIsa = internal::runtime_isa();

#endif

}  // namespace

DirectRotationKernel resolve_direct_rotation_kernel(const PreparedRotation& rotation,
                                                    internal::RuntimeIsa runtime_isa) noexcept {
    if (runtime_isa == internal::RuntimeIsa::Avx512) {
        return select_direct_rotation_avx512(rotation);
    }
    return DirectRotationKernel::Scalar;
}

void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    if (kernel != DirectRotationKernel::Scalar) {
        assert(kResolvedDirectRotationIsa == internal::RuntimeIsa::Avx512 &&
               "vector direct rotation shape requires the selected AVX-512 implementation");
        apply_direct_rotation_avx512(state, rotation, kernel, sign);
        return;
    }
    apply_rotation(state, rotation, sign);
#else
    assert(kernel == DirectRotationKernel::Scalar &&
           "portable direct rotation dispatch requires the scalar kernel");
    static_cast<void>(kernel);
    apply_rotation(state, rotation, sign);
#endif
}

}  // namespace clifft::sampling
