#include "clifft/noncomp/status_step.h"

#include "clifft/noncomp/numeric.h"

#include <stdexcept>
#include <string>

namespace clifft {

double loss_probability(const std::vector<double>& args, uint32_t op_index,
                        std::string_view caller) {
    if (args.size() != 1) {
        throw std::invalid_argument(std::string(caller) + ": LOSS at op " +
                                    std::to_string(op_index) +
                                    " requires exactly one argument (the loss probability)");
    }
    const double p = args[0];
    // is_finite_robust runs first because -ffast-math folds
    // std::isfinite() / NaN-aware comparisons away.
    if (!is_finite_robust(p) || p < 0.0 || p > 1.0) {
        throw std::invalid_argument(std::string(caller) + ": LOSS probability at op " +
                                    std::to_string(op_index) + " = " + std::to_string(p) +
                                    " is not finite or is out of [0, 1]");
    }
    return p;
}

std::optional<uint8_t> sole_lost_level(const LevelSet& levels) {
    std::optional<uint8_t> found;
    for (uint8_t l = 0; l < levels.size(); ++l) {
        if (levels.at(l).category == LevelCategory::Lost) {
            if (found.has_value()) {
                return std::nullopt;
            }
            found = l;
        }
    }
    return found;
}

OperandAction operand_action(GateType gate, QubitStatusKind kind,
                             const NonComputationalPolicy& policy) {
    if (kind == QubitStatusKind::Computational) {
        return OperandAction::Apply;
    }

    const bool lost = kind == QubitStatusKind::Lost;

    // An identity no-op is harmless to keep on any qubit.
    if (is_identity_noop(gate)) {
        return OperandAction::Apply;
    }
    // A measure-and-reset both records an outcome and restores the site. The
    // recorded outcome is supplied by the model's classifier downstream, so
    // the operation is kept: the record slot survives, and the site either
    // restores (leaked always; lost when the policy opts in) or simply stays
    // lost. This admits the X/Y-basis forms (MRX/MRY) too: on a vacated
    // carrier the classifier readout is basis-agnostic and the reset -- not
    // the readout -- is the operation's effect. A non-reset X/Y measurement
    // has no such reset and no faithful record bit, and rejects.
    if (is_measure_reset(gate)) {
        return OperandAction::Apply;
    }
    // A plain measurement keeps its visible record slot so the record and its
    // rec[-k] references do not shift; on a leaked/lost qubit the outcome is
    // supplied by the model's classifier downstream. That substitution is a
    // single record bit, faithful only for a Z-basis M. An X/Y-basis (MX/MY)
    // or multi-qubit-parity (MPP) measurement has no faithful single-bit form
    // on a noncomputational operand, so it is not representable and rejects.
    // This is a representability limit, not a policy choice.
    if (is_measurement(gate)) {
        return gate == GateType::M ? OperandAction::Apply : OperandAction::Reject;
    }
    // A reset restores a leaked qubit always, a lost qubit only by policy. A
    // non-restoring lost-qubit reset acts on a vacated site, so it drops.
    if (is_reset(gate)) {
        if (!lost || policy.reset_restores_lost) {
            return OperandAction::Apply;
        }
        return OperandAction::Drop;
    }
    // Any other operation on a leaked or lost operand -- a single-qubit gate
    // or noise channel, a two-qubit gate or noise channel, or classical
    // feedback onto a vacated site -- addresses a carrier outside the
    // computational subspace, so the interaction physically cannot happen and
    // it drops whole (identity on the surviving operands).
    return OperandAction::Drop;
}

QubitStatus normal_post_op_status(const QubitStatus& entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy) {
    const QubitStatusKind kind = entry.kind();

    if (role == OperandRole::FeedbackX || role == OperandRole::FeedbackZ) {
        // A classically-controlled correction is virtual (no physical
        // pulse): it may act within H_C but cannot move a qubit between
        // categories.
        return entry;
    }

    if (kind == QubitStatusKind::Leaked || kind == QubitStatusKind::Lost) {
        // A noncomputational qubit's status changes only when a reset
        // restores it; every other operation leaves it untouched (whether
        // that operation is dropped or rejected is decided later by the
        // rewriter). Lost is only restorable when the policy opts in;
        // Leaked always restores.
        const bool reset = gate == GateType::R || gate == GateType::MR || gate == GateType::RX ||
                           gate == GateType::RY || gate == GateType::MRX || gate == GateType::MRY;
        const bool restorable = kind == QubitStatusKind::Leaked || policy.reset_restores_lost;
        if (reset && restorable) {
            return QubitStatus::computational();
        }
        return entry;
    }

    // A computational qubit stays computational under every normal
    // operation: no gate, measurement, or reset moves it between
    // categories, and which basis state it holds is SVM runtime
    // information the ledger never tracks.
    return entry;
}

QubitStatus step_status(const QubitStatus& entry, GateType gate, OperandRole role,
                        const TransitionOutcome& outcome, const NonComputationalPolicy& policy,
                        const LevelSet& levels) {
    if (outcome.jumped) {
        return levels.status_for(outcome.destination_level);
    }
    return normal_post_op_status(entry, gate, role, policy);
}

QubitStatus step_status_dropped(const QubitStatus& entry, const TransitionOutcome& outcome,
                                const LevelSet& levels) {
    if (outcome.jumped) {
        return levels.status_for(outcome.destination_level);
    }
    return entry;
}

}  // namespace clifft
