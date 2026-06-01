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

    const LevelSet& levels() const { return levels_; }
    size_t num_levels() const { return levels_.size(); }

    // P(initial level). The stored distribution sums to exactly 1.
    // Throws on an out-of-range level id.
    double initial_probability(uint8_t level_id) const;
    const std::vector<double>& initial_state() const { return initial_state_; }

    const std::map<GateType, TransitionInstrument>& transitions() const { return transitions_; }

    // Transition instrument declared for a gate, or nullptr if the model
    // declares none. The GateType overload is what the sampler uses; the
    // string overload canonicalizes the name first (returning nullptr
    // for an unrecognized name) and is a convenience for callers holding
    // a gate-name string.
    const TransitionInstrument* transition_for(GateType gate) const;
    const TransitionInstrument* transition_for(std::string_view gate) const;

    // The classifier, or nullptr if the model has none.
    const MeasurementClassifier* classifier() const {
        return classifier_.has_value() ? &classifier_.value() : nullptr;
    }

    const NonComputationalPolicy& policy() const { return policy_; }

  private:
    LevelSet levels_;
    std::vector<double> initial_state_;
    std::map<GateType, TransitionInstrument> transitions_;
    std::optional<MeasurementClassifier> classifier_;
    NonComputationalPolicy policy_;
};

}  // namespace clifft
