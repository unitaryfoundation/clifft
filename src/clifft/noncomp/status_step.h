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
#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace clifft {

// The unique Lost-category level id, if the table has exactly one. The
// LOSS annotation resolves its destination through this; a table with no
// or several Lost levels cannot host it.
std::optional<uint8_t> sole_lost_level(const LevelSet& levels);

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
// noncomputational status effect. The two feedback roles are named for
// the correction's basis; both are virtual and status-preserving, but
// the policy scan still needs to tell them apart from physical operands
// on a vacated site.
enum class OperandRole {
    Physical,   // a real qubit operand of a physical operation
    FeedbackX,  // target of a classically-controlled X (CX rec q)
    FeedbackZ,  // target of a classically-controlled Z (CZ rec q)
};

// Outcome of consulting one transition instrument for a single
// (operation, qubit operand).
struct TransitionOutcome {
    bool jumped = false;
    uint8_t destination_level = kInvalidLevel;  // valid iff jumped
};

// How the trajectory policy handles the base operation for one operand,
// keyed on the operand's status at op entry. Computational operands always
// apply; the table only governs leaked and lost operands. Aggregated
// across an operation's operands by the caller: any Reject rejects the
// whole operation, otherwise any Drop drops it whole (identity on the
// surviving operands). A dropped operation has no physical effect, so a
// surviving operand's status keeps its entry value unless a sampled jump
// overrides it; attached transitions still fire on every operand (the
// noise process is not gated by whether the intended gate could act).
// Measurements are never dropped: their visible record slot must survive
// so rec[-k] references do not shift.
enum class OperandAction { Apply, Drop, Reject };
OperandAction operand_action(GateType gate, QubitStatusKind kind,
                             const NonComputationalPolicy& policy);

// The qubit's status after an operation given only the operation's
// normal status effect (no transition fired). No normal operation moves
// a computational qubit between categories, so its status is unchanged;
// a Leaked/Lost qubit is changed only by a reset (any flavor) that
// restores it to computational. Feedback operands are virtual and never
// change a status.
QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy);

// Full per-target step: a sampled jump destination wins; otherwise the
// operation's normal status effect applies.
QubitStatus step_status(const QubitStatus& entry, GateType gate, OperandRole role,
                        const TransitionOutcome& outcome, const NonComputationalPolicy& policy,
                        const LevelSet& levels);

// Per-target step when the base operation is dropped: the operation has
// no physical effect, so only a sampled jump changes the status.
QubitStatus step_status_dropped(const QubitStatus& entry, const TransitionOutcome& outcome,
                                const LevelSet& levels);

}  // namespace clifft
