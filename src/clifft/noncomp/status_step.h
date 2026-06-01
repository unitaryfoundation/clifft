#pragma once

// Shared qubit-status stepper for the noncomputational history layer.
//
// Both the history sampler (which samples a transition outcome) and the
// rewriter (which replays a recorded one) advance a qubit's status the
// same way; that logic lives here, in one place, so the two stay in
// sync. It implements the section 5.2.1 / 5.2.2 status-transition rules
// plus the section 5.2 reset-restore behavior.

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"

#include <cstdint>

namespace clifft {

// Outcome of consulting one transition instrument for a single
// (operation, qubit operand).
struct TransitionOutcome {
    bool jumped = false;
    uint8_t destination_level = kInvalidLevel;  // valid iff jumped
};

// The qubit's status after an operation given only the operation's
// normal status effect (no transition fired). Implements: Z-basis reset
// -> Known(g); X/Y reset -> Unknown; Z-basis measurement preserves the
// pre-SVM-known status; non-destructive probes preserve status; every
// other quantum operation demotes a computational qubit to Unknown; a
// Leaked/Lost qubit is only changed by a reset that restores it.
QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate,
                                  const NonComputationalPolicy& policy, const LevelSet& levels);

// Full per-target step: a sampled jump destination wins; otherwise the
// operation's normal status effect applies (section 5.2 per-target order
// steps 4-5).
QubitStatus step_status(const QubitStatus& entry, GateType gate, const TransitionOutcome& outcome,
                        const NonComputationalPolicy& policy, const LevelSet& levels);

}  // namespace clifft
