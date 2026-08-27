#pragma once

// Exact phase-aware selected computational-basis amplitude queries.

#include "clifft/circuit/circuit.h"
#include "clifft/sampling/executable_plan.h"

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

class BasisAmplitudeQuery {
  public:
    BasisAmplitudeQuery(const Circuit& circuit, std::span<const uint64_t> output_basis,
                        std::complex<double> input_phase = {1.0, 0.0});

    [[nodiscard]] uint32_t num_qubits() const { return plan_.num_qubits(); }
    [[nodiscard]] uint32_t peak_active_width() const { return plan_.peak_active_width(); }
    [[nodiscard]] std::complex<double> evaluate() const;

  private:
    struct ScalarRotation {
        double half_turns = 0.0;
        bool sign_constant = false;
        std::vector<uint32_t> sign_symbols;
    };

    struct Prepared {
        SamplingPlan plan;
        std::vector<uint8_t> output_records;
        std::complex<double> phase;
        std::vector<ScalarRotation> scalar_rotations;
    };

    explicit BasisAmplitudeQuery(Prepared prepared);
    [[nodiscard]] static Prepared prepare(const Circuit& circuit,
                                          std::span<const uint64_t> output_basis,
                                          std::complex<double> input_phase);

    ExecutablePlan plan_;
    std::vector<uint8_t> output_records_;
    std::complex<double> phase_;
    std::vector<ScalarRotation> scalar_rotations_;
};

}  // namespace clifft::sampling
