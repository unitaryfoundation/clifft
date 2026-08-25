#pragma once

#include "clifft/sampling/plan.h"

#include <complex>
#include <cstdint>

namespace clifft::sampling {

// Execution-ready Pauli geometry shared by CPU and device lowering.
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

[[nodiscard]] PreparedPauli prepare_pauli(ActivePauli pauli, uint32_t active_width);
[[nodiscard]] PreparedRotation prepare_rotation(ActivePauli pauli, uint32_t active_width,
                                                double half_turns);
[[nodiscard]] PreparedPromotion prepare_promotion(double half_turns);
[[nodiscard]] PreparedMeasurement prepare_measurement(ActivePauli pauli, uint32_t active_width,
                                                      uint32_t pivot);

}  // namespace clifft::sampling
