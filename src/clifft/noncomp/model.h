#pragma once

// NonComputationalModel: the assembled, validated trajectory model.
//
// Ties together a LevelSet, a per-level initial-state distribution, a
// gate-keyed table of TransitionInstruments, an optional
// MeasurementClassifier, and the NonComputationalPolicy knobs.
//
// Construction runs the model-wide consistency checks (initial state is
// a probability vector over the levels; every instrument and the
// classifier were built against this model's level table; transition
// keys name hookable physical gates; policy values are recognized) and
// throws std::invalid_argument on failure. Whether a transition's
// source context is representable is a sample-time concern enforced by
// the sampler against the target qubit's QubitStatusKind, not here.
//
// Transition keys are supplied as gate-name strings (Stim aliases such
// as "CNOT" are accepted) but stored canonicalized to GateType, so the
// sampler resolves them against the parsed circuit's GateType directly
// and two spellings of the same gate cannot silently shadow each other.

#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/transition_instrument.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace clifft {

// Raw classifier spec for the spec-based builder: the symbol labels and the
// column-substochastic matrix P[symbol][level]. Bundled so the builder takes an
// optional classifier without a separate presence flag.
struct ClassifierSpec {
    std::vector<std::string> symbols;
    std::vector<std::vector<double>> matrix;
};

class NonComputationalModel {
  public:
    // Validates that:
    //   - initial_state has one entry per level, each finite and in
    //     [0, 1]; the distribution is normalized to sum to exactly 1
    //     after checking it sums to 1 within tolerance;
    //   - every transition key names a hookable physical gate (no
    //     annotations, identity no-ops, noise channels, or synthetic
    //     gates), and no two keys canonicalize to the same gate;
    //   - every TransitionInstrument and the classifier, if present,
    //     were built against this model's level table (fingerprint
    //     match);
    //   - the policy holds recognized enum values.
    // The level table itself is already validated by LevelSet, and each
    // instrument's entry / column-sum bounds are validated at its own
    // construction; this constructor checks only the cross-object
    // consistency that no single component can see on its own.
    NonComputationalModel(LevelSet levels, std::vector<double> initial_state,
                          std::map<std::string, TransitionInstrument> transitions,
                          std::optional<MeasurementClassifier> classifier,
                          NonComputationalPolicy policy);

    // Spec-based construction: build every TransitionInstrument and the
    // classifier against `levels` from raw matrices, then assemble. Because all
    // components are built against the one LevelSet, callers never construct
    // those objects or deal with level fingerprints. `transition_matrices` maps
    // a gate-name string to its T[to][from] matrix; `classifier_spec` is
    // optional. Validation and throwing match the component from_matrix
    // factories and the constructor.
    static NonComputationalModel from_spec(
        LevelSet levels, std::vector<double> initial_state,
        const std::map<std::string, std::vector<std::vector<double>>>& transition_matrices,
        std::optional<ClassifierSpec> classifier_spec, NonComputationalPolicy policy);

    const LevelSet& levels() const { return levels_; }
    size_t num_levels() const { return levels_.size(); }

    // P(initial level). The stored distribution sums to exactly 1.
    // Throws on an out-of-range level id.
    double initial_probability(uint8_t level_id) const;
    const std::vector<double>& initial_state() const { return initial_state_; }

    // Every declared transition by its original key. A key that names a
    // hookable gate additionally registers a gate hook (see
    // transition_hooks); any key can be referenced from a circuit by a
    // TRANSITION[key] annotation.
    const std::map<std::string, TransitionInstrument, std::less<>>& transitions() const {
        return transitions_;
    }

    // Gate hooks: the gate-named subset of the transition keys, mapping
    // each hooked gate to its key. The annotation layer expands these
    // into explicit TRANSITION annotations after each hooked operation.
    const std::map<GateType, std::string>& transition_hooks() const { return hooks_; }

    // Transition instrument hooked on a gate, or nullptr if none.
    const TransitionInstrument* transition_for(GateType gate) const;

    // Transition instrument by exact key, or nullptr if none. This is the
    // lookup a TRANSITION[name] annotation resolves through.
    const TransitionInstrument* transition_named(std::string_view name) const;

    // The classifier, or nullptr if the model has none.
    const MeasurementClassifier* classifier() const {
        return classifier_.has_value() ? &classifier_.value() : nullptr;
    }

    const NonComputationalPolicy& policy() const { return policy_; }

  private:
    LevelSet levels_;
    std::vector<double> initial_state_;
    std::map<std::string, TransitionInstrument, std::less<>> transitions_;
    std::map<GateType, std::string> hooks_;
    std::optional<MeasurementClassifier> classifier_;
    NonComputationalPolicy policy_;
};

}  // namespace clifft
