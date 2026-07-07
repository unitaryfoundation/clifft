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

OperandAction operand_action(GateType gate, QubitStatus status,
                             const NonComputationalPolicy& policy) {
    if (is_computational(status)) {
        return OperandAction::Apply;
    }

    const bool lost = is_lost(status);

    // An identity no-op is harmless to keep on any qubit.
    if (is_identity_noop(gate)) {
        return OperandAction::Apply;
    }
    // A measure-and-reset both records an outcome and restores the site. The
    // recorded outcome is supplied by the model's classifier downstream, so
    // the operation is kept: the record slot survives, and the site either
    // restores (leaked always; lost when the policy opts in) or simply stays
    // lost. This admits the X/Y-basis forms (MRX/MRY): on a vacated carrier
    // the classifier readout is basis-agnostic — the reset, not the readout,
    // is the operation's effect.
    if (is_measure_reset(gate)) {
        return OperandAction::Apply;
    }
    // A plain measurement keeps its visible record slot so the record and its
    // rec[-k] references do not shift; on a leaked/lost qubit the outcome is
    // supplied by the model's classifier downstream. The classifier substitutes
    // a single record bit — the readout basis is incidental on a vacated
    // carrier, so Z-basis M, X-basis MX, and Y-basis MY are all equivalent
    // from the model's perspective and all classify. A multi-qubit parity
    // measurement (MPP, MXX, MYY, MZZ) spans more than one qubit and has no
    // faithful single-bit substitution on a noncomputational operand; it
    // rejects. MXX/MYY/MZZ desugar to MPP at parse time, so only MPP can
    // appear here.
    if (is_measurement(gate)) {
        return (gate == GateType::M || gate == GateType::MX || gate == GateType::MY)
                   ? OperandAction::Apply
                   : OperandAction::Reject;
    }
    // A reset restores a leaked qubit always, a lost qubit only by policy. A
    // non-restoring lost-qubit reset acts on a vacated site, so it drops.
    if (is_reset(gate)) {
        if (!lost || policy.reset_restores_lost) {
            return OperandAction::Apply;
        }
        return OperandAction::Drop;
    }
    // A correlated-error chain member must keep its slot in the
    // else-conditioning regardless of its operands' levels: dropping the
    // head would orphan the ELSE members and silently change each member's
    // firing probability. The effect of the chain on a vacated carrier is a
    // Pauli frame flip that nothing ever reads -- a noncomputational qubit's
    // records come from the classifier, and any restoration begins with a
    // reset -- so the operation is physically harmless and the chain's
    // else-conditioning structure is preserved.
    if (gate == GateType::CORRELATED_ERROR || gate == GateType::ELSE_CORRELATED_ERROR) {
        return OperandAction::Apply;
    }
    // Any other operation on a leaked or lost operand -- a single-qubit gate
    // or noise channel, a two-qubit gate or noise channel, or classical
    // feedback onto a vacated site -- addresses a carrier outside the
    // computational subspace, so the interaction physically cannot happen and
    // it drops whole (identity on the surviving operands).
    return OperandAction::Drop;
}

QubitStatus normal_post_op_status(QubitStatus entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy) {
    if (role == OperandRole::FeedbackX || role == OperandRole::FeedbackZ) {
        // A classically-controlled correction is virtual (no physical
        // pulse): it may act within H_C but cannot move a qubit between
        // categories.
        return entry;
    }

    if (is_leaked(entry) || is_lost(entry)) {
        // A noncomputational qubit's status changes only when a reset
        // restores it; every other operation leaves it untouched (whether
        // that operation is dropped or rejected is decided later by the
        // rewriter). Lost is only restorable when the policy opts in;
        // Leaked always restores.
        const bool reset = gate == GateType::R || gate == GateType::MR || gate == GateType::RX ||
                           gate == GateType::RY || gate == GateType::MRX || gate == GateType::MRY;
        const bool restorable = is_leaked(entry) || policy.reset_restores_lost;
        if (reset && restorable) {
            return QubitStatus::Computational;
        }
        return entry;
    }

    // A computational qubit stays computational under every normal
    // operation: no gate, measurement, or reset moves it between
    // categories, and which basis state it holds is SVM runtime
    // information the ledger never tracks.
    return entry;
}

}  // namespace clifft
