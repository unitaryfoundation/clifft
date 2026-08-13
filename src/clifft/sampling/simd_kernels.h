#pragma once

#include "clifft/sampling/kernel_dispatch.h"

namespace clifft::sampling {

// Private declarations for the separately compiled x86 implementations. Each
// function is called only after dispatch selects a compatible process ISA and
// operation shape.
void apply_direct_rotation_avx2(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept;
void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept;

[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_avx2(
    const State& state, const PreparedMeasurement& measurement) noexcept;
void collapse_active_measurement_avx2(State& state, const PreparedMeasurement& measurement,
                                      bool branch, double branch_probability) noexcept;
[[nodiscard]] MeasurementProbabilities active_measurement_probabilities_avx512(
    const State& state, const PreparedMeasurement& measurement) noexcept;
void collapse_active_measurement_avx512(State& state, const PreparedMeasurement& measurement,
                                        bool branch, double branch_probability) noexcept;

[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx2_sidecar(
    const PreparedFusedRotation& rotation);
[[nodiscard]] FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(
    const PreparedFusedRotation& rotation);

void apply_new_x_instrument_no_fire_avx2(State& state, double factor_zero, double factor_one,
                                         double no_fire_probability) noexcept;

}  // namespace clifft::sampling
