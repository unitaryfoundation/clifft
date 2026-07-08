#pragma once

// Shared qubit-status stepper for the noncomputational layer.
//
// The rewriter's circuit walk and the exact driver's classical-outcome
// walk advance a qubit's status the same way; that logic lives here, in
// one place, so the two stay in sync. It covers the per-operand policy
// actions, reset-restore of leaked/lost qubits, and the feedback rules.

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace clifft {

// The validated loss probability of a LOSS annotation node's argument
// list: exactly one argument, finite, in [0, 1]. The parser guarantees
// this shape for parsed circuits; a hand-built node that violates it is
// rejected here rather than silently defaulting to a no-op. `caller`
// prefixes the error message.
double loss_probability(const std::vector<double>& args, uint32_t op_index,
                        std::string_view caller);

// The role a qubit operand plays in an operation. The same GateType can
// mean physically different things: a CX with two qubit operands is a
// physical entangler, but a CX with a record control is a virtual,
// frame-level Pauli correction. The role, not the gate alone, drives the
// noncomputational status effect.
enum class OperandRole {
    Physical,  // a real qubit operand of a physical operation
    Feedback,  // target of a classically-controlled correction (CX/CZ rec q):
               // virtual (no physical pulse), cannot move a qubit between categories
};

// How the trajectory policy handles the base operation for one operand,
// keyed on the operand's status at op entry. Computational operands always
// apply; the table only governs leaked and lost operands. Aggregated
// across an operation's operands by the caller: any Reject rejects the
// whole operation, otherwise any Drop drops it whole (identity on the
// surviving operands). A dropped operation has no physical effect, so a
// surviving operand's status keeps its entry value unless a recorded jump
// overrides it; attached transitions still fire on every operand (the
// noise process is not gated by whether the intended gate could act).
// Measurements are never dropped: their visible record slot must survive
// so rec[-k] references do not shift.
enum class OperandAction { Apply, Drop, Reject };
OperandAction operand_action(GateType gate, QubitStatus status,
                             const NonComputationalPolicy& policy);

// The qubit's status after an operation given only the operation's
// normal status effect (no transition fired). No normal operation moves
// a computational qubit between categories, so its status is unchanged;
// a Leaked/Lost qubit is changed only by a reset (any flavor) that
// restores it to computational. Feedback operands are virtual and never
// change a status.
QubitStatus normal_post_op_status(QubitStatus entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy);

}  // namespace clifft
