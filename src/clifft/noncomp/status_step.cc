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

namespace {

// Z-diagonal operations preserve computational-basis populations, so a
// known energy level survives them unchanged; X-type operations flip the
// populations, mapping each known level to the other one. Everything
// else mixes the basis and demotes knownness. Conservative by default: a
// newly added gate demotes until it is deliberately classified here.
bool preserves_known_level(GateType g) {
    switch (g) {
        case GateType::Z:
        case GateType::S:
        case GateType::S_DAG:
        case GateType::T:
        case GateType::T_DAG:
        case GateType::R_Z:
        case GateType::CZ:
        case GateType::R_ZZ:
            return true;
        default:
            return false;
    }
}

bool flips_known_level(GateType g) {
    return g == GateType::X || g == GateType::Y;
}

}  // namespace

OperandAction operand_action(GateType gate, QubitStatusKind kind,
                             const NonComputationalPolicy& policy) {
    if (kind == QubitStatusKind::ComputationalKnown ||
        kind == QubitStatusKind::ComputationalUnknown) {
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
    // lost.
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
        // restores it; every other operation leaves it untouched (whether
        // that operation is dropped or rejected is decided later by the
        // rewriter). Lost is only restorable when the policy opts in;
        // Leaked always restores.
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
    if (kind == QubitStatusKind::ComputationalKnown) {
        // Z-diagonal gates leave a definite energy level definite; X-type
        // gates map it to the other level, still definite.
        if (preserves_known_level(gate)) {
            return entry;
        }
        if (flips_known_level(gate)) {
            const uint8_t flipped = entry.level_id() == levels.computational_zero_id()
                                        ? levels.computational_one_id()
                                        : levels.computational_zero_id();
            return levels.computational_known(flipped);
        }
    }
    // Any other quantum operation -- a basis-mixing gate, Pauli noise, or
    // an X/Y or multi-qubit measurement -- makes the energy level
    // indefinite, so a known computational qubit demotes to unknown.
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

QubitStatus step_status_dropped(const QubitStatus& entry, const TransitionOutcome& outcome,
                                const LevelSet& levels) {
    if (outcome.jumped) {
        return levels.status_for(outcome.destination_level);
    }
    return entry;
}

}  // namespace clifft
