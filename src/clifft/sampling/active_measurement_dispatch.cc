#include "clifft/sampling/active_measurement_dispatch.h"

#include "clifft/sampling/active_measurement_simd.h"
#include "clifft/sampling/simd_width.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

namespace {

constexpr uint32_t kAvx2LaneIndexBits = 2;
constexpr uint32_t kAvx512LaneIndexBits = 3;
static_assert(uint64_t{1} << kAvx2LaneIndexBits == kAvx2DoubleLanes);
static_assert(uint64_t{1} << kAvx512LaneIndexBits == kAvx512DoubleLanes);

// A single AVX-512 block regressed against scalar on its performance host,
// while the AVX2 probability-plus-collapse pair wins from one block onward.
constexpr uint32_t kMinProfitableAvx2ActiveWidth = 2;
constexpr uint32_t kMinProfitableAvx512ActiveWidth = 4;

ActiveMeasurementKernel select_active_measurement(const PreparedMeasurement& measurement,
                                                  uint64_t vector_lanes, uint32_t lane_index_bits,
                                                  uint32_t min_active_width) noexcept {
    if (measurement.pauli.is_diagonal() || measurement.pauli.x >= vector_lanes ||
        measurement.pivot >= lane_index_bits || measurement.pauli.active_width < min_active_width) {
        return ActiveMeasurementKernel::Scalar;
    }
    return ActiveMeasurementKernel::LanePaired;
}

ActiveMeasurementKernel select_active_measurement_avx2(
    const PreparedMeasurement& measurement) noexcept {
    return select_active_measurement(measurement, kAvx2DoubleLanes, kAvx2LaneIndexBits,
                                     kMinProfitableAvx2ActiveWidth);
}

ActiveMeasurementKernel select_active_measurement_avx512(
    const PreparedMeasurement& measurement) noexcept {
    return select_active_measurement(measurement, kAvx512DoubleLanes, kAvx512LaneIndexBits,
                                     kMinProfitableAvx512ActiveWidth);
}

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

const internal::RuntimeIsa kResolvedActiveMeasurementIsa = internal::runtime_isa();

#endif

}  // namespace

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
               resolve_active_measurement_kernel(measurement, kResolvedActiveMeasurementIsa) &&
           "active measurement kernel must match the process ISA");
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        if (kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx2) {
            return active_measurement_probabilities_avx2(state, measurement);
        }
        if (kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx512) {
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
               resolve_active_measurement_kernel(measurement, kResolvedActiveMeasurementIsa) &&
           "active measurement kernel must match the process ISA");
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        if (kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx2) {
            collapse_active_measurement_avx2(state, measurement, branch, branch_probability);
        } else if (kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx512) {
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

}  // namespace clifft::sampling
