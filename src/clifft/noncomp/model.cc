#include "clifft/noncomp/model.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/numeric.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

// A transition instrument models the noncomputational side effects of a
// physical qubit operation, so it may key only a gate that represents
// one. This allowlist defaults to false: a newly added GateType is
// rejected until it is deliberately classified here, rather than being
// silently accepted. Excludes annotations, identity no-ops, the
// classical measurement pad (MPAD), the expectation-value probe, and
// noise channels (whose composition with a level transition is
// deliberately left unmodeled).
bool supports_transition(GateType g) {
    if (is_clifford(g)) {
        return true;  // single- and two-qubit Clifford gates
    }
    if (is_reset(g)) {
        return true;  // R, RX, RY
    }
    if (is_measurement(g) && g != GateType::MPAD) {
        return true;  // M, MX, MY, MR, MRX, MRY, MPP, MXX, MYY, MZZ
    }
    switch (g) {
        // Non-Clifford unitaries: no trait flag distinguishes these, so
        // they are named explicitly.
        case GateType::T:
        case GateType::T_DAG:
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::U3:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
            return true;
        default:
            return false;
    }
}

}  // namespace

NonComputationalModel::NonComputationalModel(
    LevelSet levels, std::vector<double> initial_state,
    std::map<std::string, TransitionInstrument> transitions,
    std::optional<MeasurementClassifier> classifier, NonComputationalPolicy policy)
    : levels_(std::move(levels)),
      initial_state_(std::move(initial_state)),
      classifier_(std::move(classifier)),
      policy_(policy) {
    const size_t n = levels_.size();
    const uint64_t fingerprint = levels_.fingerprint();

    // Initial state must be a probability vector over the levels.
    if (initial_state_.size() != n) {
        throw std::invalid_argument("NonComputationalModel: initial_state has " +
                                    std::to_string(initial_state_.size()) + " entries; expected " +
                                    std::to_string(n) + " (one per level)");
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = initial_state_[i];
        // is_finite_robust runs first because -ffast-math folds
        // std::isfinite() / NaN-aware comparisons away.
        if (!is_finite_robust(p) || p < 0.0 || p > 1.0) {
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
    for (double& p : initial_state_) {
        p /= sum;
    }

    // Transition keys must name hookable physical gates, canonicalized to
    // GateType so the sampler resolves them against the parsed circuit
    // and no two spellings of one gate shadow each other. Each instrument
    // must have been built against this model's level table.
    std::map<GateType, std::string> first_spelling;
    for (auto& [name, instrument] : transitions) {
        const GateType gate = parse_gate_name(name);
        if (gate == GateType::UNKNOWN) {
            throw std::invalid_argument("NonComputationalModel: transition key '" + name +
                                        "' is not a recognized gate name");
        }
        if (!supports_transition(gate)) {
            throw std::invalid_argument("NonComputationalModel: transition key '" + name + "' (" +
                                        std::string(gate_name(gate)) +
                                        ") does not support a transition instrument");
        }
        if (instrument.num_levels() != n) {
            throw std::invalid_argument("NonComputationalModel: transition '" + name + "' spans " +
                                        std::to_string(instrument.num_levels()) +
                                        " levels; expected " + std::to_string(n) +
                                        " to match the level table");
        }
        if (instrument.level_fingerprint() != fingerprint) {
            throw std::invalid_argument("NonComputationalModel: transition '" + name +
                                        "' was built against a different level table");
        }
        auto [it, inserted] = transitions_.emplace(gate, std::move(instrument));
        if (!inserted) {
            throw std::invalid_argument(
                "NonComputationalModel: transition keys '" + first_spelling.at(gate) + "' and '" +
                name + "' both resolve to gate '" + std::string(gate_name(gate)) + "'");
        }
        first_spelling.emplace(gate, name);
    }

    // The classifier, if present, must have been built against the same
    // level table.
    if (classifier_.has_value()) {
        if (classifier_->num_levels() != n) {
            throw std::invalid_argument("NonComputationalModel: classifier spans " +
                                        std::to_string(classifier_->num_levels()) +
                                        " levels; expected " + std::to_string(n) +
                                        " to match the level table");
        }
        if (classifier_->level_fingerprint() != fingerprint) {
            throw std::invalid_argument(
                "NonComputationalModel: classifier was built against a different level table");
        }
    }

    // Policy must hold recognized enum values.
    switch (policy_.unknown_source_policy) {
        case UnknownSourcePolicy::Reject:
            break;
        default:
            throw std::invalid_argument(
                "NonComputationalModel: unrecognized unknown_source_policy value");
    }
}

NonComputationalModel NonComputationalModel::from_spec(
    LevelSet levels, std::vector<double> initial_state,
    const std::map<std::string, std::vector<std::vector<double>>>& transition_matrices,
    std::optional<ClassifierSpec> classifier_spec, NonComputationalPolicy policy) {
    // Build each component against the single `levels` table, so every
    // fingerprint matches by construction and the constructor's cross-object
    // checks pass without the caller ever touching a fingerprint or level id.
    std::map<std::string, TransitionInstrument> transitions;
    for (const auto& [gate, matrix] : transition_matrices) {
        transitions.emplace(gate, TransitionInstrument::from_matrix(matrix, levels));
    }

    std::optional<MeasurementClassifier> classifier;
    if (classifier_spec.has_value()) {
        classifier = MeasurementClassifier::from_matrix(std::move(classifier_spec->symbols),
                                                        std::move(classifier_spec->matrix), levels);
    }

    return NonComputationalModel(std::move(levels), std::move(initial_state),
                                 std::move(transitions), std::move(classifier), policy);
}

double NonComputationalModel::initial_probability(uint8_t level_id) const {
    if (level_id >= initial_state_.size()) {
        throw std::invalid_argument("NonComputationalModel::initial_probability: index " +
                                    std::to_string(level_id) + " out of range (num_levels " +
                                    std::to_string(initial_state_.size()) + ")");
    }
    return initial_state_[level_id];
}

const TransitionInstrument* NonComputationalModel::transition_for(GateType gate) const {
    const auto it = transitions_.find(gate);
    return it == transitions_.end() ? nullptr : &it->second;
}

const TransitionInstrument* NonComputationalModel::transition_for(std::string_view gate) const {
    const GateType type = parse_gate_name(gate);
    if (type == GateType::UNKNOWN) {
        return nullptr;
    }
    return transition_for(type);
}

}  // namespace clifft
