#pragma once

#include "clifft/sampling/active_measurement_dispatch.h"

namespace clifft::sampling {

// Linked only on x86-64 runtime-dispatch builds and called only after AVX-512
// has been selected for this process and measurement shape.
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_avx512(
    const State& state, const PreparedMeasurement& measurement) noexcept;
void collapse_active_measurement_avx512(State& state, const PreparedMeasurement& measurement,
                                        bool branch, double branch_probability) noexcept;

}  // namespace clifft::sampling
