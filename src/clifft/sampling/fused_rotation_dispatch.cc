#include "clifft/sampling/fused_rotation_dispatch.h"

#include "clifft/sampling/fused_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <utility>

namespace clifft::sampling {

PreparedFusedRotationExecution::PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                                               internal::RuntimeIsa runtime_isa)
    : rotation_(std::move(rotation)) {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    switch (runtime_isa) {
        case internal::RuntimeIsa::Avx2:
            sidecar_ = prepare_fused_rotation_avx2_sidecar(rotation_);
            break;
        case internal::RuntimeIsa::Avx512:
            sidecar_ = prepare_fused_rotation_avx512_sidecar(rotation_);
            break;
        case internal::RuntimeIsa::Scalar:
        case internal::RuntimeIsa::TrapAvx2:
        case internal::RuntimeIsa::TrapAvx512:
        case internal::RuntimeIsa::TrapUnknown:
            break;
    }
#else
    (void)runtime_isa;
#endif
}

}  // namespace clifft::sampling
