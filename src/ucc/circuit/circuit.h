#pragma once

// Circuit AST representation.
//
// The parser produces a Circuit containing a flat vector of AstNode structs.
// Each AstNode represents a single operation with its targets.
//
// Key design decisions:
// - Targets use the 32-bit encoding from target.h (qubit, rec, or Pauli-tagged)
// - Resets (R, RX) are decomposed by the parser into M + CX/CZ rec[-1]
// - MPP with multiple products is unrolled into separate AstNodes
// - REPEAT blocks are not supported in MVP (parser will error)

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

    // Optional gate argument (e.g., noise probability).
    // Currently unused in MVP but reserved for future noise support.
    double arg = 0.0;
};

// A parsed circuit ready for compilation.
struct Circuit {
    // Flat list of operations in execution order.
    std::vector<AstNode> nodes;

    // Number of qubits (max qubit index + 1).
    uint32_t num_qubits = 0;

    // Number of measurements in the circuit.
    // Used for resolving rec[-k] references during parsing.
    uint32_t num_measurements = 0;
};

}  // namespace ucc
