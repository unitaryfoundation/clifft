#pragma once

// Classify an operation's targets into the qubit operands that carry
// noncomputational semantics, with the role each plays.
//
// The same GateType can be physically different things, so the status
// walks must not key behavior on GateType alone: a
// CX/CZ with a record control is a virtual frame correction (Feedback),
// not a physical entangler; MPAD targets are classical literals, not
// qubits; detector/observable targets are record references. This shared
// helper resolves those, so both consumers agree on what is a qubit and
// what it means.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/status_step.h"

#include <cstdint>
#include <optional>
#include <string_view>
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

// One ordinary (non-annotation) node of the shared status walk: range-check
// operands, decide the whole-op policy verdict, and advance `status`.
// A rejecting operand throws std::invalid_argument (`caller` names the
// context); when any operand drops, the whole operation drops and every
// operand holds its entry status. For a measurement whose operand enters
// leaked or lost, reports that entry level -- the classifier consult the
// rewriter turns into a record substitution.
struct OrdinaryStep {
    bool dropped = false;
    std::optional<Level> measured_noncomp_level;
};
OrdinaryStep advance_ordinary_node(const AstNode& node, uint32_t op_index,
                                   std::vector<QubitStatus>& status,
                                   const NonComputationalPolicy& policy, std::string_view caller);

}  // namespace clifft
