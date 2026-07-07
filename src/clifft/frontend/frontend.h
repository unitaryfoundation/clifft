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
//    - Measurement: extract rewound observable, emit MEASURE
//    - Classical feedback: extract rewound Pauli, emit CONDITIONAL_PAULI
// 3. Return HirModule with all emitted operations

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"

#include <cstddef>
#include <map>
#include <string>

namespace clifft {

// Per-transition probabilities for materializing LEVEL_TRANSITION
// annotations into INSTRUMENT ops, in the InstrumentSite convention
// (source/destination index 0 is the |0> level, 1 the |1> level; the
// per-source trap remainder is p_total[s] - p_dest[s][0] - p_dest[s][1]).
// The front-end knows nothing of level tables or transition matrices:
// the noncomputational layer resolves a model into these plain numbers
// (instrument_trace_options() in noncomp/instrument_options.h).
struct InstrumentSpec {
    double p_total[2] = {0.0, 0.0};
    double p_dest[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
};

// Opt-in instrument materialization for trace(). `transitions` maps a
// LEVEL_TRANSITION tag to its spec; LOSS needs no entry (its probability
// is inline and its destination is entirely the trap remainder).
struct InstrumentTraceOptions {
    std::map<std::string, InstrumentSpec, std::less<>> transitions;

    // damping="neglect": dormant-random sites skip the expansion and the
    // no-fire back-action. Copied onto every materialized site.
    bool neglect_damping = false;

    // When set to a value other than SIZE_MAX, trace() reports the hidden
    // measurement slot it assigns to the reset at this node index (in the
    // circuit being traced) through HirModule::forced_traceout_slot. The
    // caller is responsible for ensuring the named node is a single-target
    // pure reset (R/RX/RY); trace() sets the output field when it processes
    // that node's hidden-branch target. SIZE_MAX means no slot is requested.
    size_t forced_traceout_node = SIZE_MAX;
};

// Trace a circuit through the Front-End, producing a HirModule.
//
// This is the main entry point for the Front-End. It:
// - Absorbs all Clifford gates into the tableau
// - Emits HeisenbergOps for T/T_DAG gates with rewound Pauli masks
// - Emits HeisenbergOps for measurements
// - Emits HeisenbergOps for classical feedback (CX/CZ with rec targets)
// - With `instruments` supplied, materializes LEVEL_TRANSITION and LOSS
//   annotations into INSTRUMENT ops (one per target, at their circuit
//   positions, mask = the rewound source projector Z_q); without it,
//   annotations reject with a pointer to sample_noncomputational.
//
// Throws std::runtime_error if the circuit exceeds the 65536-qubit VM
// axis ceiling (the only remaining hard upper bound; Pauli mask storage
// is sized at runtime).
[[nodiscard]] HirModule trace(const Circuit& circuit,
                              const InstrumentTraceOptions* instruments = nullptr);

}  // namespace clifft
