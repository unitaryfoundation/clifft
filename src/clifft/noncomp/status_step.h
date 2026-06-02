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

namespace clifft {

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

// The qubit's status after an operation given only the operation's
// normal status effect (no transition fired). For a Physical operand:
// Z-basis reset -> Known(g); X/Y reset -> Unknown; Z-basis measurement
// preserves the pre-SVM-known status; non-destructive probes preserve
// status; every other quantum operation demotes a computational qubit to
// Unknown; a Leaked/Lost qubit is only changed by a reset that restores
// it. A feedback operand never leaks (the correction is virtual): a
// FeedbackX may flip g<->e on a control bit unknown before SVM execution,
// so it demotes a known computational qubit; a FeedbackZ is phase-only and
// leaves the status untouched.
QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy, const LevelSet& levels);

// Full per-target step: a sampled jump destination wins; otherwise the
// operation's normal status effect applies.
QubitStatus step_status(const QubitStatus& entry, GateType gate, OperandRole role,
                        const TransitionOutcome& outcome, const NonComputationalPolicy& policy,
                        const LevelSet& levels);

}  // namespace clifft
