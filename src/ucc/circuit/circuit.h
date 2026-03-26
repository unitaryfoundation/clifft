#pragma once

// Circuit AST representation.
//
// The parser produces a Circuit containing a flat vector of AstNode structs.
// Each AstNode represents a single operation with its targets.
//
// Key design decisions:
// - Targets use the 32-bit encoding from target.h (qubit, rec, or Pauli-tagged)
// - Resets (R, RX, MR, MRX) are kept as first-class operations (not decomposed)
// - MPP with multiple products is unrolled into separate AstNodes
// - REPEAT blocks are unrolled at parse time (text-level replay)

#include "ucc/circuit/gate_data.h"
#include "ucc/circuit/target.h"

#include <cstdint>
#include <vector>

namespace ucc {

// A single AST node representing one circuit operation.
struct AstNode {
    GateType gate;

    // Targets for this operation.
    // See target.h for the Target type which encodes:
    // - Plain gates: qubit indices
    // - Measurements with rec references: rec-encoded targets
    // - MPP: Pauli-tagged qubit targets (one product per AstNode)
    // - CX/CZ with classical control: first target is rec, second is qubit
    std::vector<Target> targets;

    // Gate arguments (e.g., noise probabilities).
    // Most gates use args[0] for a single parameter.
    // PAULI_CHANNEL_1 uses 3 args, PAULI_CHANNEL_2 uses 15 args.
    std::vector<double> args;

    // Source line number in the original input text (1-based).
    // 0 means no source line information available.
    uint32_t source_line = 0;
};

// A parsed circuit ready for compilation.
struct Circuit {
    // Flat list of operations in execution order.
    std::vector<AstNode> nodes;

    uint32_t num_qubits = 0;

    // Number of visible measurements (M, MX, MY, MPP, MR, MRX).
    // R and RX are resets without visible measurements.
    uint32_t num_measurements = 0;

    // Number of DETECTOR declarations.
    uint32_t num_detectors = 0;

    // Number of observables (max observable index + 1).
    uint32_t num_observables = 0;

    // Number of EXP_VAL expectation value probes (one per Pauli product).
    uint32_t num_exp_vals = 0;
};

}  // namespace ucc
