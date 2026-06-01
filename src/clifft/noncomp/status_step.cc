#include "clifft/noncomp/status_step.h"

namespace clifft {

QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy, const LevelSet& levels) {
    const QubitStatusKind kind = entry.kind();

    if (role == OperandRole::FeedbackX) {
        // A classically-controlled X is virtual (no leakage), but may flip
        // g<->e on a control bit unknown before SVM execution, so a known
        // computational qubit demotes; noncomputational and already-unknown
        // qubits are left as they are.
        if (kind == QubitStatusKind::ComputationalKnown) {
            return QubitStatus::computational_unknown();
        }
        return entry;
    }
    if (role == OperandRole::FeedbackZ) {
        // A classically-controlled Z is phase-only: it cannot change the
        // energy level, so the status is unchanged.
        return entry;
    }

    const bool z_reset = gate == GateType::R || gate == GateType::MR;
    const bool xy_reset = gate == GateType::RX || gate == GateType::RY || gate == GateType::MRX ||
                          gate == GateType::MRY;

    if (kind == QubitStatusKind::Leaked || kind == QubitStatusKind::Lost) {
        // A noncomputational qubit's status changes only when a reset
        // restores it; every other operation leaves it untouched (the
        // rewriter's policy table decides drop vs. reject). Lost is only
        // restorable when the policy opts in; Leaked always restores.
        const bool restorable = kind == QubitStatusKind::Leaked || policy.reset_restores_lost;
        if (z_reset && restorable) {
            return levels.computational_known(levels.computational_zero_id());
        }
        if (xy_reset && restorable) {
            return QubitStatus::computational_unknown();
        }
        return entry;
    }

    // Computational source (Known or Unknown).
    if (z_reset) {
        return levels.computational_known(levels.computational_zero_id());
    }
    if (xy_reset) {
        return QubitStatus::computational_unknown();
    }
    if (gate == GateType::M) {
        // Pre-SVM-known semantics: a Z-basis measurement does not pin a
        // value the history layer can use. Known stays Known (its value
        // was already known); Unknown stays Unknown (the outcome lives in
        // the SVM, not here).
        return entry;
    }
    if (gate == GateType::EXP_VAL || gate == GateType::MPAD) {
        return entry;  // non-destructive probe / classical measurement pad
    }
    // Any other quantum operation -- a gate, Pauli noise, or an X/Y or
    // multi-qubit measurement -- collapses a definite energy level, so a
    // known computational qubit demotes to unknown.
    return QubitStatus::computational_unknown();
}

QubitStatus step_status(const QubitStatus& entry, GateType gate, OperandRole role,
                        const TransitionOutcome& outcome, const NonComputationalPolicy& policy,
                        const LevelSet& levels) {
    if (outcome.jumped) {
        return levels.status_for(outcome.destination_level);
    }
    return normal_post_op_status(entry, gate, role, policy, levels);
}

}  // namespace clifft
