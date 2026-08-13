#pragma once

#include "clifft/sampling/plan.h"
#include "clifft/sampling/state.h"

#include <complex>
#include <cstdint>

namespace clifft::sampling {

// Portable descriptors and scalar kernel entry points used by CPU execution.
// ISA-specific selection and implementations stay outside this header.

// Describes a Pauli operation to apply to the state vector. Its masks and
// precomputed index values identify affected coefficients without referring to
// the State's storage.
struct PreparedPauli {
    uint32_t active_width = 0;
    uint64_t x = 0;
    uint64_t z = 0;
    // Highest set X bit. It selects the pair stride and half-space for a
    // non-diagonal Pauli, and is zero only when the Pauli is diagonal.
    uint64_t pairing_bit = 0;
    std::complex<double> even_phase = {1.0, 0.0};

    [[nodiscard]] bool is_identity() const { return x == 0 && z == 0; }
    [[nodiscard]] bool is_diagonal() const { return x == 0; }
};

struct PreparedRotation {
    PreparedPauli pauli;
    double cosine = 1.0;
    double sine = 0.0;
};

struct PreparedPromotion {
    double cosine = 1.0;
    double sine = 0.0;
};

struct PreparedMeasurement {
    PreparedPauli pauli;
    uint32_t pivot = 0;
    uint64_t output_size = 0;
    uint64_t z_without_pivot = 0;
};

struct MeasurementProbabilities {
    double zero = 0.0;
    double one = 0.0;

    [[nodiscard]] double total() const { return zero + one; }
    [[nodiscard]] double for_branch(bool branch) const { return branch ? one : zero; }
};

[[nodiscard]] PreparedPauli prepare_pauli(ActivePauli pauli, uint32_t active_width);
[[nodiscard]] PreparedRotation prepare_rotation(ActivePauli pauli, uint32_t active_width,
                                                double half_turns);
[[nodiscard]] PreparedPromotion prepare_promotion(double half_turns);
[[nodiscard]] PreparedMeasurement prepare_measurement(ActivePauli pauli, uint32_t active_width,
                                                      uint32_t pivot);

// Computes <P> on normalized active-coordinate coefficients. The common
// global scalar cancels and the state is not mutated.
[[nodiscard]] double expectation_value(const State& state, const PreparedPauli& pauli) noexcept;

// Runtime signs have already been evaluated from the plan expression. A true
// sign negates the Pauli, equivalently negating the prepared sine.
void apply_rotation(State& state, const PreparedRotation& rotation, bool sign) noexcept;
void apply_promotion(State& state, const PreparedPromotion& promotion, bool sign) noexcept;

// Probability evaluation uses the normalized coefficient arrays and does not
// mutate the state; the common global scalar therefore does not affect it.
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
