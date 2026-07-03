#include "clifft/noncomp/sampler.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/util/xoshiro.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace clifft {

namespace {

// Consult one transition instrument for one qubit at an annotation site:
// pick the source column from the qubit's status at the site (the
// positional convention), apply the unknown-source policy, and sample the
// outcome. `site` names the annotation in error messages.
TransitionOutcome consult_transition(const TransitionInstrument& instrument,
                                     const QubitStatus& s_in, const LevelSet& levels,
                                     const NonComputationalPolicy& policy, Xoshiro256PlusPlus& rng,
                                     const std::string& site, uint32_t qubit, uint32_t op_index) {
    const size_t num_levels = levels.size();

    // Source-context check and column selection: an unknown computational
    // source has no definite level, so a source-dependent instrument
    // cannot pick a column for it exactly. The policy chooses between
    // rejecting and the equalized-rates approximation.
    bool equalize = false;
    uint8_t source_col = 0;
    if (s_in.kind() == QubitStatusKind::ComputationalUnknown) {
        if (instrument.is_source_independent_on_computational()) {
            // Computational columns are identical here, so any one
            // serves; use g.
            source_col = levels.computational_zero_id();
        } else if (policy.unknown_source_policy == UnknownSourcePolicy::EqualizeRates) {
            equalize = true;
        } else {
            throw std::invalid_argument(
                "sample_history: source-dependent transition '" + site +
                "' fired on ComputationalUnknown qubit " + std::to_string(qubit) + " at op " +
                std::to_string(op_index) +
                "; a source-dependent instrument cannot be applied to a qubit whose "
                "computational state is unknown");
        }
    } else {
        source_col = s_in.level_id();
    }

    TransitionOutcome outcome;
    if (equalize) {
        // Equalized-rates draw: every computational column is padded
        // with a diagonal pseudo-jump up to the maximum computational
        // jump rate p_max, so firing is source-independent and can be
        // drawn here. On fire the source is drawn uniformly over the
        // computational levels and the destination from that padded,
        // renormalized column. A pseudo-jump lands on the source
        // level itself: a transition event whose only effect is the
        // carrier collapse the rewriter materializes.
        double p_max = 0.0;
        uint8_t num_comp = 0;
        for (uint8_t l = 0; l < num_levels; ++l) {
            if (levels.at(l).category == LevelCategory::Computational) {
                ++num_comp;
                const double s = instrument.column_sum(l);
                if (s > p_max) {
                    p_max = s;
                }
            }
        }
        const double u = rng.next_double();
        if (u >= 1.0 - p_max) {
            uint8_t source = levels.computational_zero_id();
            uint8_t seen = 0;
            const double pick = rng.next_double() * num_comp;
            for (uint8_t l = 0; l < num_levels; ++l) {
                if (levels.at(l).category != LevelCategory::Computational) {
                    continue;
                }
                source = l;
                if (pick < ++seen) {
                    break;
                }
            }
            const double deficit = p_max - instrument.column_sum(source);
            const double v = rng.next_double() * p_max;
            double acc = 0.0;
            int last_positive = -1;
            for (uint8_t to = 0; to < num_levels; ++to) {
                const double p = instrument.prob(to, source) + (to == source ? deficit : 0.0);
                if (p > 0.0) {
                    last_positive = to;
                }
                acc += p;
                if (v < acc) {
                    outcome.jumped = true;
                    outcome.destination_level = to;
                    break;
                }
            }
            if (!outcome.jumped && last_positive >= 0) {
                outcome.jumped = true;
                outcome.destination_level = static_cast<uint8_t>(last_positive);
            }
        }
    } else {
        // Sample the outcome: the no-jump weight occupies [0, w), the
        // jump targets partition [w, 1). last_positive catches a
        // floating-point tail so u >= w always resolves to a jump.
        const double u = rng.next_double();
        const double no_jump = instrument.no_jump_weight(source_col);
        if (u >= no_jump) {
            double acc = no_jump;
            int last_positive = -1;
            for (uint8_t to = 0; to < num_levels; ++to) {
                const double p = instrument.prob(to, source_col);
                if (p > 0.0) {
                    last_positive = to;
                }
                acc += p;
                if (u < acc) {
                    outcome.jumped = true;
                    outcome.destination_level = to;
                    break;
                }
            }
            if (!outcome.jumped && last_positive >= 0) {
                outcome.jumped = true;
                outcome.destination_level = static_cast<uint8_t>(last_positive);
            }
        }
    }
    return outcome;
}

}  // namespace

uint8_t draw_initial_level(const NonComputationalModel& model, Xoshiro256PlusPlus& rng) {
    const size_t num_levels = model.levels().size();
    const double u = rng.next_double();
    double acc = 0.0;
    int last_positive = 0;
    for (uint8_t l = 0; l < num_levels; ++l) {
        const double p = model.initial_probability(l);
        if (p > 0.0) {
            last_positive = l;
        }
        acc += p;
        if (u < acc) {
            return l;
        }
    }
    return static_cast<uint8_t>(last_positive);
}

HistorySample sample_history(const Circuit& circuit, const NonComputationalModel& model,
                             uint64_t seed) {
    const LevelSet& levels = model.levels();
    const NonComputationalPolicy& policy = model.policy();
    const size_t num_levels = levels.size();

    Xoshiro256PlusPlus rng(seed);

    HistorySample result;
    result.history.initial_status.reserve(circuit.num_qubits);

    // Sample each qubit's initial level independently from the shared
    // initial-state distribution.
    for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
        result.history.initial_status.push_back(levels.status_for(draw_initial_level(model, rng)));
    }

    std::vector<QubitStatus> status = result.history.initial_status;

    auto check_qubit = [&](uint32_t qubit, uint32_t op_index) {
        if (qubit >= status.size()) {
            throw std::invalid_argument("sample_history: operand qubit " + std::to_string(qubit) +
                                        " is out of range (circuit declares " +
                                        std::to_string(status.size()) + " qubits) at op " +
                                        std::to_string(op_index));
        }
    };

    // Walk the operations. Transition consults happen only at LEVEL_TRANSITION
    // and LOSS annotations -- positioned where they fire, with the source
    // taken from the qubit's status there -- while every other operation
    // just advances statuses through its normal effect.
    for (uint32_t op_index = 0; op_index < circuit.nodes.size(); ++op_index) {
        const AstNode& node = circuit.nodes[op_index];
        const GateType gate = node.gate;

        if (gate == GateType::LEVEL_TRANSITION) {
            const TransitionInstrument* instrument = model.transition_named(node.tag);
            if (instrument == nullptr) {
                throw std::invalid_argument("sample_history: LEVEL_TRANSITION[" + node.tag +
                                            "] at op " + std::to_string(op_index) +
                                            " does not name a transition in the model");
            }
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                check_qubit(qubit, op_index);
                const TransitionOutcome outcome = consult_transition(
                    *instrument, status[qubit], levels, policy, rng, node.tag, qubit, op_index);
                result.history.transitions.push_back(
                    TransitionRecord{op_index, qubit, outcome.jumped,
                                     outcome.jumped ? outcome.destination_level : kInvalidLevel});
                if (outcome.jumped) {
                    status[qubit] = levels.status_for(outcome.destination_level);
                }
            }
            continue;
        }
        if (gate == GateType::LOSS) {
            const std::optional<uint8_t> lost = sole_lost_level(levels);
            if (!lost.has_value()) {
                throw std::invalid_argument(
                    "sample_history: LOSS at op " + std::to_string(op_index) +
                    " requires a level table with exactly one Lost-category level");
            }
            const double p = node.args.empty() ? 0.0 : node.args[0];
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                check_qubit(qubit, op_index);
                TransitionOutcome outcome;
                if (rng.next_double() < p) {
                    outcome.jumped = true;
                    outcome.destination_level = *lost;
                }
                result.history.transitions.push_back(
                    TransitionRecord{op_index, qubit, outcome.jumped,
                                     outcome.jumped ? outcome.destination_level : kInvalidLevel});
                if (outcome.jumped) {
                    status[qubit] = levels.status_for(outcome.destination_level);
                }
            }
            continue;
        }

        // Policy pre-scan over entry statuses, mirroring the rewriter: when
        // the policy drops the operation whole, a surviving operand's
        // stepping must know the base operation has no effect. A rejecting
        // operand is the rewriter's error to report; stepping here treats it
        // as applied, which is moot once the rewrite throws.
        bool drop_op = false;
        for (const QubitOperand& operand : qubit_operands(node)) {
            check_qubit(operand.qubit, op_index);
            if (operand_action(gate, status[operand.qubit].kind(), policy) == OperandAction::Drop) {
                drop_op = true;
            }
        }

        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            const QubitStatus s_in = status[qubit];
            status[qubit] =
                drop_op ? s_in : normal_post_op_status(s_in, gate, operand.role, policy, levels);
        }
    }

    result.final_status = status;
    return result;
}

}  // namespace clifft
