#include "clifft/sampling/fused_rotation_dispatch.h"

#include "clifft/sampling/fused_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <utility>

namespace clifft::sampling {

PreparedFusedRotationExecution::PreparedFusedRotationExecution(PreparedFusedRotation rotation,
                                                               internal::RuntimeIsa runtime_isa)
    : rotation_(std::move(rotation)) {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    if (runtime_isa == internal::RuntimeIsa::Avx512) {
        sidecar_ = prepare_fused_rotation_avx512_sidecar(rotation_);
    }
#else
    (void)runtime_isa;
#endif
}

}  // namespace clifft::sampling
