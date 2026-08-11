#include "clifft/sampling/active_measurement_dispatch.h"

#include "clifft/sampling/active_measurement_simd.h"
#include "clifft/sampling/direct_rotation_simd.h"
#include "clifft/util/runtime_isa.h"

#include <cassert>

namespace clifft::sampling {

namespace {

constexpr uint32_t kVectorLaneIndexBits = 3;
static_assert(uint64_t{1} << kVectorLaneIndexBits == kAvx512DoubleLanes);

// A single vector block regressed against the scalar probability-plus-collapse
// pair on this host; two blocks amortize the vector setup.
constexpr uint32_t kMinProfitableActiveWidth = 4;

ActiveMeasurementKernel select_active_measurement_avx512(
    const PreparedMeasurement& measurement) noexcept {
    if (measurement.pauli.is_diagonal() || measurement.pauli.x >= kAvx512DoubleLanes ||
        measurement.pivot >= kVectorLaneIndexBits ||
        measurement.pauli.active_width < kMinProfitableActiveWidth) {
        return ActiveMeasurementKernel::Scalar;
    }
    return ActiveMeasurementKernel::LanePaired;
}

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

const internal::RuntimeIsa kResolvedActiveMeasurementIsa = internal::runtime_isa();

#endif

}  // namespace

ActiveMeasurementKernel resolve_active_measurement_kernel(
    const PreparedMeasurement& measurement, internal::RuntimeIsa runtime_isa) noexcept {
    if (runtime_isa == internal::RuntimeIsa::Avx512) {
        return select_active_measurement_avx512(measurement);
    }
    return ActiveMeasurementKernel::Scalar;
}

MeasurementProbabilities active_measurement_probabilities(const State& state,
                                                          const PreparedMeasurement& measurement,
                                                          ActiveMeasurementKernel kernel) noexcept {
#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        assert(kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx512 &&
               "vector active measurement requires the selected AVX-512 implementation");
        return active_measurement_probabilities_avx512(state, measurement);
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
    if (kernel == ActiveMeasurementKernel::LanePaired) {
        assert(kResolvedActiveMeasurementIsa == internal::RuntimeIsa::Avx512 &&
               "vector active measurement requires the selected AVX-512 implementation");
        collapse_active_measurement_avx512(state, measurement, branch, branch_probability);
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
