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
//    - Clifford and SPP gates: apply to simulator (absorbed into tableau)
//    - T/T_DAG and TPP gates: emit HeisenbergOps with rewound Pauli masks
//    - Measurement: extract rewound observable, emit MEASURE
//    - Classical feedback: extract rewound Pauli, emit CONDITIONAL_PAULI
// 3. Return HirModule with all emitted operations

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"

#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace clifft {

// Opt-in instrument materialization for trace(). `transitions` maps a
// LEVEL_TRANSITION tag to its spec; LEAKAGE and LOSS need no entry (their
// probabilities are inline and their destinations are entirely in the trap
// remainder).
struct InstrumentTraceOptions {
    // The front-end remains model-free: the noncomputational layer compresses
    // each five-level matrix into InstrumentProbabilities before tracing.
    std::map<std::string, InstrumentProbabilities, std::less<>> transitions;

    // damping="neglect": dormant-random sites skip the expansion and the
    // no-fire back-action. trace() copies this module-wide setting once to
    // HirModule::neglect_instrument_damping.
    bool neglect_instrument_damping = false;

    // When set, trace() reports the hidden measurement slot it assigns to
    // the reset at this node index (in the circuit being traced) through
    // HirModule::forced_traceout_slot. The named node must be a single-target
    // pure reset (R/RX/RY), which trace() validates before processing the
    // circuit. nullopt means no slot is requested.
    std::optional<size_t> forced_traceout_node;
};

// Trace a circuit through the Front-End, producing a HirModule.
//
// This is the main entry point for the Front-End. It:
// - Absorbs all Clifford gates, including SPP/SPP_DAG, into the tableau
// - Emits HeisenbergOps for T/T_DAG and TPP/TPP_DAG with rewound Pauli masks
// - Emits HeisenbergOps for measurements
// - Emits HeisenbergOps for classical feedback (CX/CZ with rec targets)
// - With `instruments` supplied, materializes LEVEL_TRANSITION, LEAKAGE, and
//   LOSS annotations into INSTRUMENT ops (one per target, at their circuit
//   positions, mask = the rewound source projector Z_q); without it,
//   annotations reject with a pointer to sample_noncomputational.
//
// Throws std::runtime_error if the circuit exceeds the conservative
// 65536-qubit safety ceiling. Pauli mask storage is otherwise sized at
// runtime.
[[nodiscard]] HirModule trace(const Circuit& circuit,
                              const InstrumentTraceOptions* instruments = nullptr);

}  // namespace clifft
