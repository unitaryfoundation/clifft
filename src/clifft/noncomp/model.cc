#include "clifft/noncomp/model.h"

#include "clifft/circuit/gate_data.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

// is_finite_robust below assumes IEEE 754 doubles (every Clifft
// target satisfies this). Make the assumption explicit.
static_assert(std::numeric_limits<double>::is_iec559,
              "NonComputationalModel requires IEEE 754 doubles");

namespace {

// Tolerance for the initial-state sum-to-one check. Matches the
// derived-quantity tolerance used by TransitionInstrument and
// MeasurementClassifier: raw entries are strict, the derived sum
// tolerates floating drift.
constexpr double kProbTolerance = 1e-12;

// Release builds use -ffast-math, which implies -ffinite-math-only.
// That folds away std::isfinite() and lets `v >= 0.0 && v <= 1.0`
// pass NaN through. Inspect the IEEE 754 bit pattern instead: a
// non-finite double has all exponent bits set.
bool is_finite_robust(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
}

// A transition instrument models the noncomputational side effects of a
// physical qubit operation, so it may only key a gate that represents
// one. Annotations, identity no-ops, the classical measurement pad, the
// expectation-value probe, and noise channels (whose composition with a
// level transition is deliberately left undefined in the MVP) are not
// hookable.
bool is_hookable_transition_gate(GateType g) {
    if (g == GateType::UNKNOWN) {
        return false;
    }
    if (gate_arity(g) == GateArity::ANNOTATION) {
        return false;
    }
    if (is_identity_noop(g) || is_noise_gate(g) || is_exp_val(g)) {
        return false;
    }
    if (g == GateType::MPAD) {
        return false;
    }
    return true;
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
        if (!is_hookable_transition_gate(gate)) {
            throw std::invalid_argument("NonComputationalModel: transition key '" + name + "' (" +
                                        std::string(gate_name(gate)) +
                                        ") is not a hookable physical gate");
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
