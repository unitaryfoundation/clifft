#pragma once

// Phase-sensitive front-end result used by exact amplitude compilation.

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"
#include "clifft/tableau/stabilizer_ch_form.h"

#include <complex>
#include <span>
#include <variant>
#include <vector>

namespace clifft {

class PhaseAwareCliffordFrame {
  public:
    struct NamedOperation {
        GateType gate;
        std::vector<uint32_t> targets;
    };

    explicit PhaseAwareCliffordFrame(uint32_t num_qubits);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    void apply_named_gate(GateType gate, std::span<const uint32_t> targets);
    void apply_pauli_rotation(PauliStringView axis, bool dagger);

    // Composes a Clifford on the input side of the accumulated operator. The
    // operations are in circuit order and therefore precede earlier input-side
    // changes when another operator is composed later.
    void compose_input(std::span<const NamedOperation> operations);

    // Constructs the exact stabilizer state U_C^dagger|basis> for the
    // accumulated Clifford operator U_C.
    [[nodiscard]] StabilizerChForm inverse_on_basis(std::span<const uint64_t> basis) const;

  private:
    struct PauliRotation {
        PauliString axis;
        bool dagger;
    };

    uint32_t num_qubits_;
    std::vector<std::variant<NamedOperation, PauliRotation>> operations_;
};

struct PhaseAwareHir {
    HirModule hir;
    PhaseAwareCliffordFrame final_clifford_frame;
    std::complex<double> source_scalar{1.0, 0.0};
};

// Trace a pure-unitary circuit while retaining the scalar discarded by the
// ordinary projective front end. Non-unitary nodes are rejected before trace.
[[nodiscard]] PhaseAwareHir trace_phase_aware(const Circuit& circuit);

// Query-private variant for a unitary circuit followed only by computational-
// basis measurements. The measurement effects retain the unitary source phase.
[[nodiscard]] PhaseAwareHir trace_phase_aware_terminal_measurements(const Circuit& circuit);

}  // namespace clifft
