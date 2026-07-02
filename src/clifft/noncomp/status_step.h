#pragma once

// Shared qubit-status stepper for the noncomputational history layer.
//
// Both the history sampler (which samples a transition outcome) and the
// rewriter (which replays a recorded one) advance a qubit's status the
// same way; that logic lives here, in one place, so the two stay in
// sync. It covers gate demotion, the measurement and reset effects
// (including the pre-SVM-known measurement rule), reset-restore of
// leaked/lost qubits, and the feedback rules.

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <optional>

namespace clifft {

// The unique Lost-category level id, if the table has exactly one. The
// LOSS annotation resolves its destination through this; a table with no
// or several Lost levels cannot host it.
std::optional<uint8_t> sole_lost_level(const LevelSet& levels);

// The role a qubit operand plays in an operation. The same GateType can
// mean physically different things: a CX with two qubit operands is a
// physical entangler, but a CX with a record control is a virtual,
// frame-level Pauli correction. The role, not the gate alone, drives the
// noncomputational status effect. The two feedback roles split on the
// correction's basis: a conditional X can flip the energy level, a
// conditional Z is phase-only and leaves it intact.
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
// normal status effect (no transition fired). For a Physical operand:
// Z-basis reset -> Known(g); X/Y reset -> Unknown; Z-basis measurement
// preserves the pre-SVM-known status; non-destructive probes preserve
// status; a Z-diagonal gate (Z/S/T/CZ...) preserves a known level and an
// X-type gate (X/Y) flips it to the other known level; every other
// quantum operation demotes a computational qubit to Unknown; a
// Leaked/Lost qubit is only changed by a reset that restores it. A feedback operand never leaks
// (the correction is virtual): a FeedbackX may flip g<->e on a control bit unknown before SVM
// execution, so it demotes a known computational qubit; a FeedbackZ is phase-only and leaves the
// status untouched.
QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy, const LevelSet& levels);

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
