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
    // A measure-and-reset is kept: the record slot survives (the classifier
    // supplies the bit downstream) and the site restores per policy or stays
    // lost. The X/Y-basis forms behave identically -- the readout basis is
    // incidental on a vacated carrier.
    if (is_measure_reset(gate)) {
        return OperandAction::Apply;
    }
    // A plain measurement keeps its record slot, so rec[-k] references do
    // not shift; M, MX, and MY classify alike. A multi-qubit parity
    // measurement has no faithful single-bit substitution and rejects;
    // MXX/MYY/MZZ desugar to MPP at parse time, so only MPP appears here.
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
    // else-conditioning: dropping one would change every later member's
    // firing probability. Its effect on a vacated carrier is a frame flip
    // nothing reads (records come from the classifier; any restoration
    // begins with a reset), so keeping it is harmless.
    if (gate == GateType::CORRELATED_ERROR || gate == GateType::ELSE_CORRELATED_ERROR) {
        return OperandAction::Apply;
    }
    // Anything else addressing a leaked or lost operand drops whole, acting
    // as the identity on the surviving operands.
    return OperandAction::Drop;
}

QubitStatus normal_post_op_status(QubitStatus entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy) {
    if (role == OperandRole::Feedback) {
        // A classically-controlled correction is virtual (no physical
        // pulse): it may act within H_C but cannot move a qubit between
        // categories.
        return entry;
    }

    if (is_leaked(entry) || is_lost(entry)) {
        // A noncomputational qubit's status changes only when a reset
        // restores it; every other operation leaves it untouched (drop
        // versus reject is operand_action's decision). Lost is restorable
        // only when the policy opts in; Leaked always restores.
        const bool reset = is_reset(gate) || is_measure_reset(gate);
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
