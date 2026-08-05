#pragma once

#include "clifft/sampling/plan.h"
#include "clifft/sampling/soa_state.h"

#include <complex>
#include <cstdint>

namespace clifft::sampling {

// Fixed-size execution descriptor derived before hot dispatch. It contains the
// mask relationships needed by direct kernels but no semantic-plan storage.
struct PreparedPauli {
    uint32_t active_width = 0;
    uint64_t x = 0;
    uint64_t z = 0;
    uint64_t pair_selector = 0;
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

// Runtime signs have already been evaluated from the plan expression. A true
// sign negates the Pauli, equivalently negating the prepared sine.
void apply_rotation(SoaState& state, const PreparedRotation& rotation, bool sign);
void apply_promotion(SoaState& state, const PreparedPromotion& promotion, bool sign);

// Probability evaluation uses the normalized coefficient planes and does not
// mutate the state; the common global scalar therefore does not affect it.
// Collapse consumes the selected probability so the caller can sample or force
// a branch once and reuse exactly that value for normalization and
// log-probability accounting.
[[nodiscard]] MeasurementProbabilities measurement_probabilities(
    const SoaState& state, const PreparedMeasurement& measurement);
void collapse_measurement(SoaState& state, const PreparedMeasurement& measurement, bool branch,
                          double branch_probability);

}  // namespace clifft::sampling
