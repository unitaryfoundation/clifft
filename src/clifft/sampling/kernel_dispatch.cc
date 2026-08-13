#include "clifft/sampling/kernel_dispatch.h"

#include "clifft/sampling/simd_kernels.h"
#include "clifft/sampling/simd_width.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>
#include <utility>

namespace clifft::sampling {

namespace {

// Active measurements.
constexpr uint32_t kAvx2LaneIndexBits = 2;
constexpr uint32_t kAvx512LaneIndexBits = 3;
constexpr uint32_t kMinProfitableAvx2MeasurementWidth = 2;
constexpr uint32_t kMinProfitableAvx512MeasurementWidth = 4;
static_assert(uint64_t{1} << kAvx2LaneIndexBits == kAvx2DoubleLanes);
static_assert(uint64_t{1} << kAvx512LaneIndexBits == kAvx512DoubleLanes);

ActiveMeasurementKernel select_active_measurement(const PreparedMeasurement& measurement,
                                                  uint64_t vector_lanes,
                                                  uint32_t lane_index_bits,
                                                  uint32_t min_active_width) noexcept {
    if (measurement.pauli.is_diagonal() || measurement.pauli.x >= vector_lanes ||
        measurement.pivot >= lane_index_bits || measurement.pauli.active_width < min_active_width) {
        return ActiveMeasurementKernel::Scalar;
    }
    return ActiveMeasurementKernel::LanePaired;
}

ActiveMeasurementKernel select_active_measurement_avx2(
    const PreparedMeasurement& measurement) noexcept {
    // The probability-plus-collapse pair wins from one AVX2 block onward.
    return select_active_measurement(measurement, kAvx2DoubleLanes, kAvx2LaneIndexBits,
                                     kMinProfitableAvx2MeasurementWidth);
}

ActiveMeasurementKernel select_active_measurement_avx512(
    const PreparedMeasurement& measurement) noexcept {
    // A single AVX-512 block regressed against scalar on the performance host.
    return select_active_measurement(measurement, kAvx512DoubleLanes, kAvx512LaneIndexBits,
                                     kMinProfitableAvx512MeasurementWidth);
}

// Direct rotations.
constexpr uint32_t kMinAvx2RotationWidth = 2;
constexpr uint32_t kMinAvx512RotationWidth = 3;
constexpr uint64_t kNoExcludedPairingBit = 0;
constexpr uint64_t kPivotFourPairingBit = uint64_t{1} << 4;
static_assert(uint64_t{1} << kMinAvx2RotationWidth == kAvx2DoubleLanes);
static_assert(uint64_t{1} << kMinAvx512RotationWidth == kAvx512DoubleLanes);

DirectRotationKernel select_direct_rotation(const PreparedRotation& rotation, uint64_t vector_lanes,
                                            uint32_t min_active_width,
                                            uint64_t excluded_pairing_bit) noexcept {
    if (rotation.pauli.is_identity()) {
        return DirectRotationKernel::Scalar;
    }
    if (rotation.pauli.is_diagonal()) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::Diagonal
                                                               : DirectRotationKernel::Scalar;
    }
    const uint64_t pairing_bit = rotation.pauli.pairing_bit;
    if (pairing_bit < vector_lanes) {
        return rotation.pauli.active_width >= min_active_width ? DirectRotationKernel::LanePaired
                                                               : DirectRotationKernel::Scalar;
    }
    if (pairing_bit == excluded_pairing_bit) {
        return DirectRotationKernel::Scalar;
    }
    return DirectRotationKernel::HighPivot;
}

DirectRotationKernel select_direct_rotation_avx2(const PreparedRotation& rotation) noexcept {
    // Stride-16 pairing was neutral or faster than scalar across the measured
    // AVX2 active widths, so every high pivot uses the vector kernel.
    return select_direct_rotation(rotation, kAvx2DoubleLanes, kMinAvx2RotationWidth,
                                  kNoExcludedPairingBit);
}

DirectRotationKernel select_direct_rotation_avx512(const PreparedRotation& rotation) noexcept {
    // Stride-16 pairing regressed against scalar at every measured active
    // width on the AVX-512 performance host.
    return select_direct_rotation(rotation, kAvx512DoubleLanes, kMinAvx512RotationWidth,
                                  kPivotFourPairingBit);
}

// New-X instrument activation.
// One four-coefficient block was neutral in the direct microbenchmark, while
// every wider measured width won; smaller states stay on the scalar path.
constexpr uint32_t kMinAvx2InstrumentWidth = 2;
static_assert(uint64_t{1} << kMinAvx2InstrumentWidth == kAvx2DoubleLanes);

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

const internal::RuntimeIsa kResolvedKernelIsa = internal::runtime_isa();

#endif

}  // namespace

// Active measurements.
ActiveMeasurementKernel resolve_active_measurement_kernel(
    const PreparedMeasurement& measurement, internal::RuntimeIsa runtime_isa) noexcept {
    if (runtime_isa == internal::RuntimeIsa::Avx2) {
        return select_active_measurement_avx2(measurement);
    }
    if (runtime_isa == internal::RuntimeIsa::Avx512) {
        return select_active_measurement_avx512(measurement);
    }
    return ActiveMeasurementKernel::Scalar;
}

MeasurementProbabilities active_measurement_probabilities(const State& state,
                                                          const PreparedMeasurement& measurement,
                                                          ActiveMeasurementKernel kernel) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    assert(kernel ==
               resolve_active_measurement_kernel(measurement, kResolvedKernelIsa) &&
           "active measurement kernel must match the process ISA");
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        if (kResolvedKernelIsa == internal::RuntimeIsa::Avx2) {
            return active_measurement_probabilities_avx2(state, measurement);
        }
        if (kResolvedKernelIsa == internal::RuntimeIsa::Avx512) {
            return active_measurement_probabilities_avx512(state, measurement);
        }
        assert(false && "vector active measurement requires a selected SIMD implementation");
    }
#else
    assert(kernel == ActiveMeasurementKernel::Scalar &&
           "portable active measurement dispatch requires the scalar kernel");
    static_cast<void>(kernel);
#endif
    return measurement_probabilities(state, measurement);
}

void collapse_active_measurement(State& state, const PreparedMeasurement& measurement,
                                 ActiveMeasurementKernel kernel, bool branch,
                                 double branch_probability) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    assert(kernel ==
               resolve_active_measurement_kernel(measurement, kResolvedKernelIsa) &&
           "active measurement kernel must match the process ISA");
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        if (kResolvedKernelIsa == internal::RuntimeIsa::Avx2) {
            collapse_active_measurement_avx2(state, measurement, branch, branch_probability);
        } else if (kResolvedKernelIsa == internal::RuntimeIsa::Avx512) {
            collapse_active_measurement_avx512(state, measurement, branch, branch_probability);
        } else {
            assert(false && "vector active measurement requires a selected SIMD implementation");
            collapse_measurement(state, measurement, branch, branch_probability);
        }
        return;
    }
#else
    assert(kernel == ActiveMeasurementKernel::Scalar &&
           "portable active measurement dispatch requires the scalar kernel");
    static_cast<void>(kernel);
#endif
    collapse_measurement(state, measurement, branch, branch_probability);
}

// Direct rotations.
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
    assert(kernel == resolve_direct_rotation_kernel(rotation, kResolvedKernelIsa) &&
           "direct rotation kernel must match the process ISA");
    if (kernel != DirectRotationKernel::Scalar) {
        if (kResolvedKernelIsa == internal::RuntimeIsa::Avx2) {
            apply_direct_rotation_avx2(state, rotation, kernel, sign);
        } else if (kResolvedKernelIsa == internal::RuntimeIsa::Avx512) {
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

// Fused rotations.
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

// New-X instrument activation.
NewXInstrumentKernel resolve_new_x_instrument_kernel(uint32_t active_width,
                                                     internal::RuntimeIsa runtime_isa) noexcept {
    // The AVX-512 tier includes every feature required by the AVX2 kernel, and
    // reusing it avoids falling back to baseline scalar code in portable wheels.
    if ((runtime_isa == internal::RuntimeIsa::Avx2 ||
         runtime_isa == internal::RuntimeIsa::Avx512) &&
        active_width >= kMinAvx2InstrumentWidth) {
        return NewXInstrumentKernel::Avx2;
    }
    return NewXInstrumentKernel::Scalar;
}

void apply_new_x_instrument_no_fire_dispatched(State& state, double factor_zero, double factor_one,
                                               double no_fire_probability,
                                               NewXInstrumentKernel kernel) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    assert(kernel == resolve_new_x_instrument_kernel(state.active_width(), kResolvedKernelIsa) &&
           "new-X instrument kernel must match the process ISA");
    if (kernel == NewXInstrumentKernel::Avx2) {
        if (kResolvedKernelIsa == internal::RuntimeIsa::Avx2 ||
            kResolvedKernelIsa == internal::RuntimeIsa::Avx512) {
            apply_new_x_instrument_no_fire_avx2(state, factor_zero, factor_one,
                                                no_fire_probability);
            return;
        }
        assert(false && "AVX2 new-X instrument kernel requires an AVX2-capable process ISA");
    }
#else
    assert(kernel == NewXInstrumentKernel::Scalar &&
           "portable new-X instrument dispatch requires the scalar kernel");
    static_cast<void>(kernel);
#endif
    apply_new_x_instrument_no_fire(state, factor_zero, factor_one, no_fire_probability);
}

}  // namespace clifft::sampling
