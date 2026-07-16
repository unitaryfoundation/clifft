#pragma once

// NonComputationalModel: the assembled, validated trajectory model.
//
// Ties together a per-level initial-state distribution, a name-keyed
// table of TransitionInstruments, an optional MeasurementClassifier,
// and the NonComputationalPolicy knobs.
//
// Construction runs the model-wide consistency checks (initial state is
// a probability vector over the levels; transition keys survive the
// LEVEL_TRANSITION tag syntax; policy values are recognized) and throws
// std::invalid_argument on failure.
//
// Transition keys are arbitrary names, stored verbatim; a circuit
// references any of them by exact key with a LEVEL_TRANSITION[key]
// annotation. A key that names a hookable physical gate (Stim aliases
// such as "CNOT" are accepted) additionally registers a gate hook,
// keyed by GateType so two spellings of one gate cannot register
// competing hooks.

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/transition_instrument.h"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace clifft {

class NonComputationalModel {
  public:
    // The one construction path: build every TransitionInstrument and
    // the classifier from raw matrices, then assemble and validate.
    // `transition_matrices` maps a transition key to its T[to][from]
    // matrix; `classifier_matrix` is optional. Throws
    // std::invalid_argument, naming the offending component, when a
    // matrix is malformed, the initial state is not a probability
    // vector over the five levels, a transition key cannot be
    // referenced by a LEVEL_TRANSITION tag, two hook-registering keys
    // resolve to the same gate, or the policy holds an unrecognized
    // value.
    static NonComputationalModel from_spec(
        std::vector<double> initial_state,
        const std::map<std::string, std::vector<std::vector<double>>>& transition_matrices,
        std::optional<std::vector<std::vector<double>>> classifier_matrix,
        NonComputationalPolicy policy);

    // P(initial level). The stored distribution sums to exactly 1.
    double initial_probability(Level level) const {
        return initial_state_[static_cast<size_t>(level)];
    }
    // Every declared transition by its original key. A key that names a
    // hookable gate additionally registers a gate hook (see
    // transition_hooks); any key can be referenced from a circuit by a
    // LEVEL_TRANSITION[key] annotation.
    const std::map<std::string, TransitionInstrument, std::less<>>& transitions() const {
        return transitions_;
    }

    // Gate hooks map each hooked gate to its transition key.
    // expand_transition_hooks() inserts the corresponding LEVEL_TRANSITION
    // annotations after each hooked operation.
    const std::map<GateType, std::string>& transition_hooks() const { return hooks_; }

    // Transition instrument by exact key, or nullptr if none. This is the
    // lookup a LEVEL_TRANSITION[name] annotation resolves through.
    const TransitionInstrument* transition_named(std::string_view name) const;

    // The classifier, or nullptr if the model has none.
    const MeasurementClassifier* classifier() const {
        return classifier_.has_value() ? &classifier_.value() : nullptr;
    }

    const NonComputationalPolicy& policy() const { return policy_; }

  private:
    NonComputationalModel(std::vector<double> initial_state,
                          std::map<std::string, TransitionInstrument> transitions,
                          std::optional<MeasurementClassifier> classifier,
                          NonComputationalPolicy policy);

    std::array<double, kNumLevels> initial_state_{};
    std::map<std::string, TransitionInstrument, std::less<>> transitions_;
    std::map<GateType, std::string> hooks_;
    std::optional<MeasurementClassifier> classifier_;
    NonComputationalPolicy policy_;
};

}  // namespace clifft
