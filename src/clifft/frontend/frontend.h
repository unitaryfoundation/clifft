#pragma once

// Front-End: Trace Generation
//
// The Front-End consumes a parsed Circuit and produces a HirModule.
// It drives Stim's TableauSimulator to absorb Clifford gates, and emits
// HeisenbergOps for non-Clifford gates (T, T_DAG) and measurements.
//
// Key algorithm:
// 1. Initialize TableauSimulator with identity tableau
// 2. For each gate in the circuit:
//    - Clifford: apply to simulator (absorbed into tableau)
//    - T/T_DAG: extract rewound Z from inv_state.zs[q], emit HeisenbergOp
//    - Measurement: extract rewound observable, handle AG pivot if needed
//    - Classical feedback: extract rewound Pauli, emit CONDITIONAL_PAULI
// 3. Return HirModule with all emitted operations

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"

namespace clifft {

// Trace a circuit through the Front-End, producing a HirModule.
//
// This is the main entry point for the Front-End. It:
// - Absorbs all Clifford gates into the tableau
// - Emits HeisenbergOps for T/T_DAG gates with rewound Pauli masks
// - Emits HeisenbergOps for measurements
// - Emits HeisenbergOps for classical feedback (CX/CZ with rec targets)
//
// Throws std::runtime_error if the circuit exceeds CLIFFT_MAX_QUBITS.
[[nodiscard]] HirModule trace(const Circuit& circuit);

}  // namespace clifft
