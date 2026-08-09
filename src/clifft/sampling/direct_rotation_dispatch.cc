#include "clifft/sampling/direct_rotation_dispatch.h"

#include "clifft/sampling/direct_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

DirectRotationKernel resolve_direct_rotation_kernel(const PreparedRotation& rotation,
                                                    internal::RuntimeIsa runtime_isa) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    return select_direct_rotation_kernel(rotation, runtime_isa == internal::RuntimeIsa::Avx512);
#else
    (void)runtime_isa;
    return select_direct_rotation_kernel(rotation, false);
#endif
}

void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    if (kernel != DirectRotationKernel::Scalar) {
        apply_direct_rotation_avx512(state, rotation, kernel, sign);
        return;
    }
#else
    assert(kernel == DirectRotationKernel::Scalar &&
           "portable direct rotation dispatch requires the scalar kernel");
#endif
    apply_rotation(state, rotation, sign);
}

}  // namespace clifft::sampling
