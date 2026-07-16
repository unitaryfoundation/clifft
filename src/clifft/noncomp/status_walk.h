#pragma once

// Shared rules for interpreting circuit operations during noncomputational
// sampling.
//
// The rewriter and runtime driver must agree on which qubits an operation
// affects, whether to apply, drop, or reject it, how it changes each qubit's
// computational, leaked, or lost status, and which measurements must be
// handled outside the VM. This file keeps those decisions in one place.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace clifft {

// The validated loss probability of a LOSS annotation node's argument list:
// exactly one argument, finite, in [0, 1]. `caller` prefixes diagnostics.
double loss_probability(const std::vector<double>& args, uint32_t op_index,
                        std::string_view caller);

// The role a qubit operand plays in an operation. A CX/CZ with a record
// control is a virtual frame correction, not a physical gate application.
enum class OperandRole {
    Physical,
    Feedback,
};

struct QubitOperand {
    uint32_t qubit;
    OperandRole role;
};

// The qubit operands of a node, in target order, skipping record references
// and MPAD literals. Throws when a record accompanies a qubit on a gate other
// than CX/CZ feedback.
std::vector<QubitOperand> qubit_operands(const AstNode& node);

// How the policy handles an operation for one operand. Callers combine the
// results across all operands: Reject takes precedence, and any Drop makes
// the whole operation the identity.
enum class OperandAction { Apply, Drop, Reject };
OperandAction operand_action(GateType gate, QubitStatus status,
                             const NonComputationalPolicy& policy);

// The qubit's status after the operation's normal effect, before any attached
// transition. Feedback is virtual; only a restoring reset changes a
// noncomputational status.
QubitStatus normal_post_op_status(QubitStatus entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy);

struct ClassifiedOperand {
    uint32_t qubit;
    Level level;
};

// Result of advancing one ordinary, non-annotation circuit node.
struct OrdinaryStep {
    bool dropped = false;
    std::optional<ClassifiedOperand> classified_measurement;
};

// Range-check the node's operands, apply the whole-operation policy, advance
// `status`, and report the qubit and leaked/lost level when a measurement
// must be classified outside the SVM.
OrdinaryStep advance_ordinary_node(const AstNode& node, uint32_t op_index,
                                   std::vector<QubitStatus>& status,
                                   const NonComputationalPolicy& policy, std::string_view caller);

}  // namespace clifft
