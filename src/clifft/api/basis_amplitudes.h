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
    struct Prepared {
        SamplingPlan plan;
        std::vector<uint64_t> execution_basis;
        std::complex<double> phase;
        bool conjugate_result = false;
    };

    BasisAmplitudeQuery(const Circuit& circuit, Prepared prepared);
    [[nodiscard]] static Prepared prepare(const Circuit& circuit,
                                          std::span<const uint64_t> output_basis,
                                          std::complex<double> input_phase);

    ExecutablePlan plan_;
    std::vector<uint64_t> output_basis_;
    std::complex<double> phase_;
    bool conjugate_result_ = false;
};

}  // namespace clifft::sampling
