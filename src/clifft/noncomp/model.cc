#include "clifft/noncomp/model.h"

#include "clifft/circuit/gate_data.h"

#include <cstdint>
#include <cstring>
#include <limits>
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

}  // namespace

NonComputationalModel::NonComputationalModel(
    LevelSet levels, std::vector<double> initial_state,
    std::map<std::string, TransitionInstrument> transitions,
    std::optional<MeasurementClassifier> classifier, NonComputationalPolicy policy)
    : levels_(std::move(levels)),
      initial_state_(std::move(initial_state)),
      transitions_(std::move(transitions)),
      classifier_(std::move(classifier)),
      policy_(policy) {
    const size_t n = levels_.size();

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

    // Transition keys must name gates in clifft's vocabulary, and each
    // instrument must span the same level table as the model.
    for (const auto& [gate, instrument] : transitions_) {
        if (parse_gate_name(gate) == GateType::UNKNOWN) {
            throw std::invalid_argument("NonComputationalModel: transition key '" + gate +
                                        "' is not a recognized gate name");
        }
        if (instrument.num_levels() != n) {
            throw std::invalid_argument("NonComputationalModel: transition '" + gate + "' spans " +
                                        std::to_string(instrument.num_levels()) +
                                        " levels; expected " + std::to_string(n) +
                                        " to match the level table");
        }
    }

    // The classifier, if present, must span the same level table.
    if (classifier_.has_value() && classifier_->num_levels() != n) {
        throw std::invalid_argument(
            "NonComputationalModel: classifier spans " + std::to_string(classifier_->num_levels()) +
            " levels; expected " + std::to_string(n) + " to match the level table");
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

const TransitionInstrument* NonComputationalModel::transition_for(std::string_view gate) const {
    const auto it = transitions_.find(std::string(gate));
    return it == transitions_.end() ? nullptr : &it->second;
}

}  // namespace clifft
