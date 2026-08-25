#pragma once

#include "clifft/sampling/pauli_preparation.h"
#include "clifft/sampling/state.h"

#include <cstdint>

namespace clifft::sampling {

// Portable scalar kernel entry points used by CPU execution. ISA-specific
// selection and implementations stay outside this header.

struct MeasurementProbabilities {
    double zero = 0.0;
    double one = 0.0;

    [[nodiscard]] double total() const { return zero + one; }
    [[nodiscard]] double for_branch(bool branch) const { return branch ? one : zero; }
};

// Computes <P> on normalized active-coordinate coefficients without mutating
// the state.
[[nodiscard]] double expectation_value(const State& state, const PreparedPauli& pauli) noexcept;

// Runtime signs have already been evaluated from the plan expression. A true
// sign negates the Pauli, equivalently negating the prepared sine.
void apply_rotation(State& state, const PreparedRotation& rotation, bool sign) noexcept;
void apply_promotion(State& state, const PreparedPromotion& promotion, bool sign) noexcept;
void apply_rotation_parallel(State& state, const PreparedRotation& rotation, bool sign,
                             uint32_t workers, uint32_t min_active_width) noexcept;
void apply_promotion_parallel(State& state, const PreparedPromotion& promotion, bool sign,
                              uint32_t workers, uint32_t min_active_width) noexcept;

// Probability evaluation does not mutate the normalized coefficient arrays.
// Collapse consumes the selected probability so the caller can sample or force
// a branch once and reuse exactly that value for normalization and
// log-probability accounting.
[[nodiscard]] MeasurementProbabilities measurement_probabilities(
    const State& state, const PreparedMeasurement& measurement) noexcept;
void collapse_measurement(State& state, const PreparedMeasurement& measurement, bool branch,
                          double branch_probability) noexcept;

// Instrument back-action preserves active coordinates. An activating site adds
// one |0> coordinate first; its prepared source Pauli already uses that wider
// layout. The population reduction is shared with measurement preparation,
// while filter and collapse keep the full coefficient array instead of
// compacting a coordinate.
void activate_zero_coordinate(State& state) noexcept;
void apply_new_x_instrument_no_fire(State& state, double factor_zero, double factor_one,
                                    double no_fire_probability) noexcept;
void collapse_new_x_instrument_source(State& state, bool branch,
                                      double branch_probability) noexcept;
void apply_instrument_no_fire(State& state, const PreparedPauli& source, double factor_zero,
                              double factor_one, double no_fire_probability) noexcept;
void collapse_instrument_source(State& state, const PreparedPauli& source, bool branch,
                                double branch_probability) noexcept;

}  // namespace clifft::sampling
