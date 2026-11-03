#include "clifft/noncomp/model.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/status_walk.h"
#include "clifft/util/numeric.h"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

// A transition instrument models the noncomputational side effects of a
// physical qubit operation, so it may key only a gate that represents
// one. Classification lives in the gate table's trait flags, whose
// defaults reject a new GateType until it is deliberately classified
// there. Excludes annotations, identity no-ops, the classical
// measurement pad (MPAD), the expectation-value probe, and noise
// channels (whose composition with a level transition is deliberately
// left unmodeled). Parser-desugared gates are rejected as hook keys
// before this predicate is consulted.
bool supports_transition(GateType g) {
    if (is_unitary(g) && !is_identity_noop(g)) {
        return true;  // unitary gates, Clifford or not
    }
    if (is_reset(g)) {
        return true;  // R, RX, RY
    }
    if (is_physical_measurement(g)) {
        return true;  // M, MX, MY, MR, MRX, MRY, MPP
    }
    return false;
}

}  // namespace

NonComputationalModel::NonComputationalModel(
    std::vector<double> initial_state, std::map<std::string, TransitionInstrument> transitions,
    std::optional<MeasurementClassifier> classifier, NonComputationalPolicy policy)
    : classifier_(std::move(classifier)), policy_(policy) {
    // Initial state must be a probability vector over the levels.
    if (initial_state.size() != kNumLevels) {
        throw std::invalid_argument("NonComputationalModel: initial_state has " +
                                    std::to_string(initial_state.size()) + " entries; expected " +
                                    std::to_string(kNumLevels) + " (one per level)");
    }
    double sum = 0.0;
    for (size_t i = 0; i < kNumLevels; ++i) {
        const double p = initial_state[i];
        if (!is_probability(p)) {
            throw std::invalid_argument("NonComputationalModel: initial_state entry " +
                                        std::to_string(i) + " = " + std::to_string(p) +
                                        " is not finite or is out of [0, 1]");
        }
        sum += p;
    }
    if (sum < 1.0 - kProbTolerance || sum > 1.0 + kProbTolerance) {
        throw std::invalid_argument("NonComputationalModel: initial_state sums to " +
                                    std::to_string(sum) + "; must sum to 1");
    }
    // Normalize away the within-tolerance drift so the sampler never has
    // to compensate for an unsampled tail.
    for (size_t i = 0; i < kNumLevels; ++i) {
        initial_state_[i] = initial_state[i] / sum;
    }

    // A transition key is referenced from circuits by LEVEL_TRANSITION[key]
    // annotations, so it must survive the tag grammar. A key that names a
    // hookable physical gate additionally registers a gate hook,
    // canonicalized to GateType so no two spellings of one gate shadow
    // each other.
    for (auto& [name, instrument] : transitions) {
        if (!is_representable_tag(name)) {
            throw std::invalid_argument(
                "NonComputationalModel: transition key '" + name +
                "' cannot be written as a LEVEL_TRANSITION tag: a key must be nonempty, have no "
                "leading or trailing whitespace, and contain no ']', '#', or newline");
        }
        // Only a key naming a hookable physical gate registers a hook.
        // Arbitrary names remain available for explicit LEVEL_TRANSITION
        // annotations. A recognized but non-hookable instruction name is
        // rejected because accepting it as named-only would look like a hook
        // that silently never fires.
        const GateType gate = parse_gate_name(name);
        if (gate != GateType::UNKNOWN && is_parser_desugared(gate)) {
            throw std::invalid_argument(
                "NonComputationalModel: transition key '" + name +
                "' cannot key a gate hook: " + name +
                " is rewritten at parse time and never appears as a circuit node, so the hook "
                "could "
                "never fire; place LEVEL_TRANSITION[...] annotations explicitly");
        }
        if (gate != GateType::UNKNOWN && is_identity_noop(gate)) {
            throw std::invalid_argument(
                "NonComputationalModel: transition key '" + name +
                "' cannot key a gate hook: identity no-ops emit no circuit nodes, so the hook "
                "could never fire; place LEVEL_TRANSITION[...] annotations at the intended "
                "positions");
        }
        if (gate != GateType::UNKNOWN && !supports_transition(gate)) {
            throw std::invalid_argument(
                "NonComputationalModel: transition key '" + name +
                "' cannot key a gate hook: this instruction does not support transition hooks; "
                "choose a non-gate transition name and place LEVEL_TRANSITION[...] annotations "
                "explicitly");
        }
        if (gate != GateType::UNKNOWN && supports_transition(gate)) {
            auto [it, inserted] = hooks_.emplace(gate, name);
            if (!inserted) {
                throw std::invalid_argument(
                    "NonComputationalModel: transition keys '" + it->second + "' and '" + name +
                    "' both resolve to gate '" + std::string(gate_name(gate)) + "'");
            }
        }
        transitions_.emplace(name, std::move(instrument));
    }

    // Policy must hold recognized enum values.
    switch (policy_.damping) {
        case DampingPolicy::Exact:
        case DampingPolicy::Neglect:
            break;
        default:
            throw std::invalid_argument("NonComputationalModel: unrecognized damping value");
    }
}

NonComputationalModel NonComputationalModel::from_spec(
    std::vector<double> initial_state,
    const std::map<std::string, std::vector<std::vector<double>>>& transition_matrices,
    std::optional<std::vector<std::vector<double>>> classifier_matrix,
    NonComputationalPolicy policy) {
    return from_spec(std::move(initial_state), transition_matrices, std::move(classifier_matrix),
                     policy, {}, {});
}

NonComputationalModel NonComputationalModel::from_spec(
    std::vector<double> initial_state,
    const std::map<std::string, std::vector<std::vector<double>>>& transition_matrices,
    std::optional<std::vector<std::vector<double>>> classifier_matrix,
    NonComputationalPolicy policy, GatePartnerEffects gate_partner_effects,
    const std::vector<InteractionRule>& interactions) {
    std::map<std::string, TransitionInstrument> transitions;
    for (const auto& [gate, matrix] : transition_matrices) {
        try {
            transitions.emplace(gate, TransitionInstrument::from_matrix(matrix));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("NonComputationalModel: transition '" + gate +
                                        "': " + e.what());
        }
    }

    std::optional<MeasurementClassifier> classifier;
    if (classifier_matrix.has_value()) {
        classifier = MeasurementClassifier::from_matrix(*classifier_matrix);
    }

    NonComputationalModel model(std::move(initial_state), std::move(transitions),
                                std::move(classifier), policy);
    if (gate_partner_effects.leaked) {
        gate_partner_effects.leaked->validate(true, "gate_partner_effects leaked");
    }
    if (gate_partner_effects.lost) {
        gate_partner_effects.lost->validate(false, "gate_partner_effects lost");
    }
    model.gate_partner_effects_ = std::move(gate_partner_effects);
    for (const auto& rule : interactions) {
        const std::string context = "interaction rule for '" + rule.gate + "'";
        const GateType gate = parse_gate_name(rule.gate);
        if (!supports_partner_effect(gate)) {
            throw std::invalid_argument(context +
                                        " requires a supported native two-qubit unitary gate");
        }
        if (rule.source_operand > 1) {
            throw std::invalid_argument(context + " source_operand must be 0 or 1");
        }
        if (rule.source_status != InteractionSource::Leaked &&
            rule.source_status != InteractionSource::Lost) {
            throw std::invalid_argument(context + " has an unrecognized source status");
        }
        rule.effect.validate(rule.source_status == InteractionSource::Leaked, context);
        const auto key = std::make_tuple(gate, rule.source_operand, rule.source_status);
        if (!model.interactions_.emplace(key, rule.effect).second) {
            throw std::invalid_argument(context + " duplicates a rule for canonical gate '" +
                                        std::string(gate_name(gate)) +
                                        "' with the same direction and status");
        }
    }
    return model;
}

const PartnerEffect* NonComputationalModel::partner_effect(GateType gate, uint32_t source_operand,
                                                           InteractionSource source_status) const {
    if (!supports_partner_effect(gate)) {
        return nullptr;
    }
    const auto it = interactions_.find({gate, source_operand, source_status});
    if (it != interactions_.end()) {
        return &it->second;
    }
    const auto& effect = source_status == InteractionSource::Leaked ? gate_partner_effects_.leaked
                                                                    : gate_partner_effects_.lost;
    return effect ? &*effect : nullptr;
}

const TransitionInstrument* NonComputationalModel::transition_named(std::string_view name) const {
    const auto it = transitions_.find(name);
    return it == transitions_.end() ? nullptr : &it->second;
}

namespace {
bool has_enabled_partner_effect(const NonComputationalModel& model, GateType gate) {
    for (uint32_t source = 0; source < 2; ++source) {
        for (const auto status : {InteractionSource::Leaked, InteractionSource::Lost}) {
            const PartnerEffect* effect = model.partner_effect(gate, source, status);
            if (effect != nullptr && !effect->empty()) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

Circuit expand_transition_hooks(const Circuit& circuit, const NonComputationalModel& model,
                                std::vector<std::optional<uint32_t>>* generated_by) {
    const auto& hooks = model.transition_hooks();

    Circuit out = circuit.metadata_only_copy();
    out.nodes.reserve(circuit.nodes.size() * 2);
    if (generated_by != nullptr) {
        generated_by->clear();
        generated_by->reserve(circuit.nodes.size() * 2);
    }
    auto append = [&](AstNode node, std::optional<uint32_t> parent) {
        out.nodes.push_back(std::move(node));
        if (generated_by != nullptr) {
            generated_by->push_back(parent);
        }
    };

    for (const AstNode& node : circuit.nodes) {
        const auto gate_index = static_cast<uint32_t>(out.nodes.size());
        append(node, std::nullopt);
        if (supports_partner_effect(node.gate) && has_enabled_partner_effect(model, node.gate)) {
            const auto operands = qubit_operands(node);
            if (operands.size() == 2 && operands[0].role == OperandRole::Physical &&
                operands[1].role == OperandRole::Physical) {
                for (uint32_t source = 0; source < 2; ++source) {
                    for (const auto status : {InteractionSource::Leaked, InteractionSource::Lost}) {
                        const PartnerEffect* effect =
                            model.partner_effect(node.gate, source, status);
                        if (effect == nullptr || effect->empty()) {
                            continue;
                        }
                        const bool leakage = status == InteractionSource::Leaked;
                        std::vector<double> args(effect->pauli.begin(), effect->pauli.end());
                        if (leakage) {
                            args.push_back(effect->spread_probability);
                        }
                        append(AstNode{leakage ? GateType::LEAKAGE_INTERACTION
                                               : GateType::LOSS_INTERACTION,
                                       {node.targets[source], node.targets[1 - source]},
                                       std::move(args),
                                       node.source_line},
                               gate_index);
                    }
                }
            } else if (operands.size() != 1 || operands[0].role != OperandRole::Feedback) {
                throw std::invalid_argument(
                    "configured partner effects require one physical gate pair per node");
            }
        }
        const auto hook = hooks.find(node.gate);
        if (hook == hooks.end()) {
            continue;
        }
        // Add a transition for each qubit the gate acts on directly.
        // Record-controlled feedback does not physically execute the gate.
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand.role != OperandRole::Physical) {
                continue;
            }
            append(AstNode{GateType::LEVEL_TRANSITION,
                           {Target::qubit(operand.qubit)},
                           {},
                           node.source_line,
                           hook->second},
                   gate_index);
        }
    }
    return out;
}

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model) {
    constexpr Level computational[2] = {Level::G, Level::E};

    InstrumentTraceOptions options;
    options.neglect_instrument_damping = model.policy().damping == DampingPolicy::Neglect;

    for (const auto& [name, instrument] : model.transitions()) {
        InstrumentProbabilities probabilities;
        // Compress only the columns whose source remains quantum. p_fire
        // retains each full five-level column sum, while
        // p_computational_dest retains its G/E destination entries. Their
        // difference is the aggregate LeakG/LeakE/Lost trap probability.
        // Columns sourced at a noncomputational level stay in the model and
        // are consulted classically by the trajectory driver.
        for (int s = 0; s < 2; ++s) {
            probabilities.p_fire[s] = instrument.column_sum(computational[s]);
            for (int d = 0; d < 2; ++d) {
                probabilities.p_computational_dest[s][d] =
                    instrument.prob(computational[d], computational[s]);
            }
        }
        options.transitions.emplace(name, probabilities);
    }
    return options;
}

}  // namespace clifft
