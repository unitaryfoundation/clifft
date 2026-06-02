#pragma once

// Classify an operation's targets into the qubit operands that carry
// noncomputational semantics, with the role each plays.
//
// The same GateType can be physically different things, so the history
// sampler and the rewriter must not key behavior on GateType alone: a
// CX/CZ with a record control is a virtual frame correction (Feedback),
// not a physical entangler; MPAD targets are classical literals, not
// qubits; detector/observable targets are record references. This shared
// helper resolves those, so both consumers agree on what is a qubit and
// what it means.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/status_step.h"

#include <cstdint>
#include <vector>

namespace clifft {

struct QubitOperand {
    uint32_t qubit;
    OperandRole role;
};

// The qubit operands of a node, in target order, skipping non-qubit
// targets (record references, MPAD pad literals). A node with a record
// control is classical feedback, so its qubit operands get the Feedback
// role; otherwise operands are Physical.
std::vector<QubitOperand> qubit_operands(const AstNode& node);

}  // namespace clifft
