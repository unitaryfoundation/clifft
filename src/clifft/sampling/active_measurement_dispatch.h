#pragma once

#include "clifft/sampling/kernels.h"

#include <cstdint>

namespace clifft::internal {
enum class RuntimeIsa;
}

namespace clifft::sampling {

// Architecture-neutral selection stored in an executable measurement action.
enum class ActiveMeasurementKernel : uint8_t {
    Scalar,
    LanePaired,
};

static_assert(sizeof(ActiveMeasurementKernel) == 1);

[[nodiscard]] ActiveMeasurementKernel resolve_active_measurement_kernel(
    const PreparedMeasurement& measurement, internal::RuntimeIsa runtime_isa) noexcept;

[[nodiscard]] MeasurementProbabilities active_measurement_probabilities(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept;
void collapse_active_measurement(State& state, const PreparedMeasurement& measurement,
                                 ActiveMeasurementKernel kernel, bool branch,
                                 double branch_probability) noexcept;

}  // namespace clifft::sampling
