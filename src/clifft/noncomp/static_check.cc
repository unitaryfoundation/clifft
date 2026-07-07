// Static pre-sampling validation for circuit/model pairs.
//
// Runs an abstract interpretation over per-qubit sets of reachable
// QubitStatus values (a uint8_t bitmask), calling the same policy
// primitives the rewriter calls on concrete statuses.  A set bit at
// position s means QubitStatus(s) is reachable for that qubit.  The
// governing rule for stochastic channels: a member survives a channel
// only when its no-event branch is reachable -- a source whose jump
// away is certain is replaced by its destinations, exactly as the
// concrete draw can never keep it.
//
// Node kinds and how the abstract walk handles each:
//
//   LEVEL_TRANSITION[tag]  Look up instrument->prob(dest, source) for each
//                          source level of the target qubit's current status
//                          members.  For Computational members, the sources
//                          are G and E; for a noncomp member s, the source
//                          is noncomp_level(s).  Every destination level
//                          with prob > 0 adds status_for(dest); the member
//                          itself survives only when some source column
//                          leaves no-fire mass (column_sum < 1).
//
//   LOSS(p)                p > 0 adds QubitStatus::Lost to each target's
//                          set; computational and leaked members survive
//                          only when p < 1 (an already-lost member is a
//                          no-op).  p == 0 is a no-op entirely (mirrors
//                          trace()'s zero-fire elision).
//
//   All other nodes        For each qubit operand and each member status s:
//                          * operand_action(gate, s, policy) == Reject  →
//                            throw std::invalid_argument immediately.
//                          * is_measurement(gate) and s is not
//                            Computational  →  require model.classifier()
//                            != nullptr, else throw.
//                          * New member set is the image
//                            { normal_post_op_status(s, gate, role, policy)
//                              : s ∈ set, action != Reject }
//                            when the action is Apply, and s itself when
//                            the action is Drop.

#include "clifft/noncomp/static_check.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

// A bitmask over QubitStatus enumerators.  Bit i is set when
// QubitStatus(i) is in the reachable set.  The four enumerators are
// Computational=0, LeakG=1, LeakE=2, Lost=3, so a uint8_t suffices.
using StatusSet = uint8_t;

inline StatusSet set_of(QubitStatus s) {
    return static_cast<uint8_t>(1u << static_cast<uint8_t>(s));
}

inline void add_status(StatusSet& set, QubitStatus s) {
    set |= set_of(s);
}

inline bool has_status(StatusSet set, QubitStatus s) {
    return (set & set_of(s)) != 0;
}

// Initial status set for one qubit: every level with initial mass
// contributes its corresponding QubitStatus (status_for collapses g and
// e onto Computational). A model whose initial state carries no
// computational mass starts with no Computational member.
StatusSet initial_set(const NonComputationalModel& model) {
    StatusSet s = 0;
    for (const Level l : kAllLevels) {
        if (model.initial_probability(l) > 0.0) {
            add_status(s, status_for(l));
        }
    }
    return s;
}

// Advance `set` through a LEVEL_TRANSITION node's transition matrix.
// Each member is replaced by the image of its source levels: every
// destination with prob > 0 contributes status_for(dest), and the
// member itself survives only when some source column leaves no-fire
// mass (column_sum < 1 -- from_matrix clamps in-tolerance sums to
// exactly 1, so the comparison is exact for certain columns).
void advance_transition(StatusSet& set, const TransitionInstrument& instrument) {
    StatusSet next = 0;
    for (uint8_t si = 0; si < 4; ++si) {
        const QubitStatus member = static_cast<QubitStatus>(si);
        if (!has_status(set, member)) {
            continue;
        }
        // Determine the source levels that correspond to this status.
        // Computational covers both G and E (the SVM holds which one at
        // runtime); noncomputational statuses map to a single definite level.
        if (is_computational(member)) {
            for (const Level src : {Level::G, Level::E}) {
                for (const Level dest : kAllLevels) {
                    if (instrument.prob(dest, src) > 0.0) {
                        add_status(next, status_for(dest));
                    }
                }
                if (instrument.column_sum(src) < 1.0) {
                    add_status(next, QubitStatus::Computational);
                }
            }
        } else {
            const Level src = noncomp_level(member);
            for (const Level dest : kAllLevels) {
                if (instrument.prob(dest, src) > 0.0) {
                    add_status(next, status_for(dest));
                }
            }
            if (instrument.column_sum(src) < 1.0) {
                add_status(next, member);
            }
        }
    }
    set = next;
}

// Advance `set` through an ordinary (non-annotation) node for one
// qubit operand with the given role.  Each member status is processed
// through operand_action and normal_post_op_status:
//   Apply  →  post = normal_post_op_status(s, gate, role, policy)
//   Drop   →  post = s (entry status preserved)
//   Reject →  caller has already thrown; unreachable here
StatusSet advance_ordinary(StatusSet set, GateType gate, OperandRole role,
                           const NonComputationalPolicy& policy) {
    StatusSet next = 0;
    for (uint8_t si = 0; si < 4; ++si) {
        const QubitStatus s = static_cast<QubitStatus>(si);
        if (!has_status(set, s)) {
            continue;
        }
        const OperandAction action = operand_action(gate, s, policy);
        if (action == OperandAction::Reject) {
            // Callers check for Reject before calling this function.
            // This branch is unreachable.
            continue;
        }
        if (action == OperandAction::Drop) {
            add_status(next, s);
        } else {
            add_status(next, normal_post_op_status(s, gate, role, policy));
        }
    }
    return next;
}

}  // namespace

void validate_static(const Circuit& annotated, const NonComputationalModel& model) {
    const NonComputationalPolicy& policy = model.policy();
    const uint32_t nq = annotated.num_qubits;

    // Initialize per-qubit reachable-status sets from the model's initial
    // distribution.
    std::vector<StatusSet> sets(nq, initial_set(model));

    for (uint32_t op_index = 0; op_index < static_cast<uint32_t>(annotated.nodes.size());
         ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        const GateType gate = node.gate;

        // --- Annotation nodes ---
        if (gate == GateType::LEVEL_TRANSITION) {
            // Guard against a null instrument (malformed model reference);
            // the driver's up-front resolve_annotation will also catch this,
            // but a defensive null check here keeps this function safe even
            // when called before that loop.
            const TransitionInstrument* instrument = model.transition_named(node.tag);
            if (instrument == nullptr) {
                // Unresolved tag: skip; resolve_annotation will throw.
                continue;
            }
            for (const Target& t : node.targets) {
                if (t.is_rec()) {
                    continue;
                }
                const uint32_t q = t.value();
                if (q < nq) {
                    advance_transition(sets[q], *instrument);
                }
            }
            continue;
        }

        if (gate == GateType::LOSS) {
            // A LOSS with args size != 1 or out-of-range probability is
            // rejected by the driver's up-front resolve_annotation; skip
            // gracefully here rather than re-validating.
            if (node.args.size() != 1) {
                continue;
            }
            const double p = node.args[0];
            if (p > 0.0) {
                for (const Target& t : node.targets) {
                    if (t.is_rec()) {
                        continue;
                    }
                    const uint32_t q = t.value();
                    if (q >= nq) {
                        continue;
                    }
                    // Computational and leaked members can vacate; they
                    // survive only when the no-fire branch is reachable
                    // (p < 1). An already-lost member is a no-op.
                    StatusSet next = 0;
                    for (uint8_t si = 0; si < 4; ++si) {
                        const QubitStatus s = static_cast<QubitStatus>(si);
                        if (!has_status(sets[q], s)) {
                            continue;
                        }
                        add_status(next, QubitStatus::Lost);
                        if (s == QubitStatus::Lost || p < 1.0) {
                            add_status(next, s);
                        }
                    }
                    sets[q] = next;
                }
            }
            // p == 0 is a no-op: the channel can never fire, so Lost is
            // not added to the reachable set (mirrors trace()'s zero-fire
            // elision).
            continue;
        }

        // --- Ordinary nodes ---
        // First pass: any Reject or missing-classifier member throws.
        // Second pass: advance each operand's set pointwise (Drop members
        // keep their entry status, Apply members map through
        // normal_post_op_status).  The concrete walk drops a node whole
        // when any operand drops, but that coupling needs no modeling
        // here: only resets change statuses, and resets are single-qubit,
        // so the pointwise image is exact.
        const std::vector<QubitOperand> operands = qubit_operands(node);
        for (const QubitOperand& operand : operands) {
            const uint32_t q = operand.qubit;
            if (q >= nq) {
                continue;
            }
            const StatusSet qset = sets[q];
            for (uint8_t si = 0; si < 4; ++si) {
                const QubitStatus s = static_cast<QubitStatus>(si);
                if (!has_status(qset, s)) {
                    continue;
                }
                switch (operand_action(gate, s, policy)) {
                    case OperandAction::Reject:
                        throw std::invalid_argument(
                            "sample_noncomputational: operation '" + std::string(gate_name(gate)) +
                            "' on qubit " + std::to_string(q) + " at op " +
                            std::to_string(op_index) + " can meet a " + status_name(s) +
                            " qubit under this model and is not representable"
                            "; rejecting before sampling");
                    case OperandAction::Drop:
                        break;
                    case OperandAction::Apply:
                        break;
                }
                // Classifier check: a measurement on a noncomputational
                // qubit requires a classifier.
                if (is_measurement(gate) && !is_computational(s)) {
                    if (model.classifier() == nullptr) {
                        throw std::invalid_argument(
                            "sample_noncomputational: measurement '" +
                            std::string(gate_name(gate)) + "' on qubit " + std::to_string(q) +
                            " at op " + std::to_string(op_index) + " can meet a " + status_name(s) +
                            " qubit and requires a classifier, but the model has none"
                            "; rejecting before sampling");
                    }
                }
            }
        }

        for (const QubitOperand& operand : operands) {
            const uint32_t q = operand.qubit;
            if (q >= nq) {
                continue;
            }
            sets[q] = advance_ordinary(sets[q], gate, operand.role, policy);
        }
    }
}

}  // namespace clifft
