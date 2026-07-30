#include "clifft/noncomp/model.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/noncomp/numeric.h"

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
    if (is_measurement(g) && g != GateType::MPAD) {
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

    return NonComputationalModel(std::move(initial_state), std::move(transitions),
                                 std::move(classifier), policy);
}

const TransitionInstrument* NonComputationalModel::transition_named(std::string_view name) const {
    const auto it = transitions_.find(name);
    return it == transitions_.end() ? nullptr : &it->second;
}

}  // namespace clifft
