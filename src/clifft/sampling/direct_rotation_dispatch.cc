#include "clifft/sampling/direct_rotation_dispatch.h"

#include "clifft/sampling/direct_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

namespace {

constexpr uint32_t kMinAvx2ActiveWidth = 2;
constexpr uint32_t kMinAvx512ActiveWidth = 3;
constexpr uint64_t kNoExcludedPairSelector = 0;
constexpr uint64_t kPivotFourSelector = uint64_t{1} << 4;
static_assert(uint64_t{1} << kMinAvx2ActiveWidth == kAvx2DoubleLanes);
static_assert(uint64_t{1} << kMinAvx512ActiveWidth == kAvx512DoubleLanes);

DirectRotationKernel select_direct_rotation(const PreparedRotation& rotation, uint64_t vector_lanes,
                                            uint32_t min_active_width,
                                            uint64_t excluded_pair_selector) noexcept {
    if (rotation.pauli.is_identity()) {
        return DirectRotationKernel::Scalar;
    }
    if (rotation.pauli.is_diagonal()) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::Diagonal
                                                               : DirectRotationKernel::Scalar;
    }
    const uint64_t pairing_bit = rotation.pauli.pair_selector;
    if (pairing_bit < vector_lanes) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::LanePaired
                                                               : DirectRotationKernel::Scalar;
    }
    if (pairing_bit == excluded_pair_selector) {
        return DirectRotationKernel::Scalar;
    }
    return DirectRotationKernel::HighPivot;
}

DirectRotationKernel select_direct_rotation_avx2(const PreparedRotation& rotation) noexcept {
    // Stride-16 pairing was neutral or faster than scalar across the measured
    // AVX2 active widths, so every high pivot uses the vector kernel.
    return select_direct_rotation(rotation, kAvx2DoubleLanes, kMinAvx2ActiveWidth,
                                  kNoExcludedPairSelector);
}

DirectRotationKernel select_direct_rotation_avx512(const PreparedRotation& rotation) noexcept {
    // Stride-16 pairing regressed against scalar at every measured active
    // width on the AVX-512 performance host.
    return select_direct_rotation(rotation, kAvx512DoubleLanes, kMinAvx512ActiveWidth,
                                  kPivotFourSelector);
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
    if (runtime_isa == internal::RuntimeIsa::Avx2) {
        return select_direct_rotation_avx2(rotation);
    }
    if (runtime_isa == internal::RuntimeIsa::Avx512) {
        return select_direct_rotation_avx512(rotation);
    }
    return DirectRotationKernel::Scalar;
}

void apply_direct_rotation(State& state, const PreparedRotation& rotation,
                           DirectRotationKernel kernel, bool sign) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    assert(kernel == resolve_direct_rotation_kernel(rotation, kResolvedDirectRotationIsa) &&
           "direct rotation kernel must match the process ISA");
    if (kernel != DirectRotationKernel::Scalar) {
        if (kResolvedDirectRotationIsa == internal::RuntimeIsa::Avx2) {
            apply_direct_rotation_avx2(state, rotation, kernel, sign);
        } else if (kResolvedDirectRotationIsa == internal::RuntimeIsa::Avx512) {
            apply_direct_rotation_avx512(state, rotation, kernel, sign);
        } else {
            assert(false && "vector direct rotation requires a selected SIMD implementation");
            apply_rotation(state, rotation, sign);
        }
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
