#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/transition_instrument.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::LevelSet;
using clifft::MeasurementClassifier;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::TransitionInstrument;
using clifft::UnknownSourcePolicy;

namespace {

// Identity transition on the default 5-level set: every source stays at
// its own level (column sums all 1). Valid as an instrument.
TransitionInstrument identity_transition(const LevelSet& levels) {
    const size_t n = levels.size();
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        m[i][i] = 1.0;
    }
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// "Identity + reject" classifier for the default 5-level set.
MeasurementClassifier identity_classifier(const LevelSet& levels) {
    return MeasurementClassifier::from_matrix({"0", "1"},
                                              {
                                                  {1, 0, 0, 0, 0},
                                                  {0, 1, 0, 0, 0},
                                              },
                                              levels);
}

// A valid probability vector over the default 5 levels.
std::vector<double> default_initial_state() {
    return {0.5, 0.3, 0.1, 0.05, 0.05};
}

// A two-level set (g, e), distinct in size from the default 5-level set.
LevelSet two_level_set() {
    using clifft::BasisBit;
    using clifft::Level;
    using clifft::LevelCategory;
    return LevelSet({
        Level{"g", LevelCategory::Computational, BasisBit::Zero},
        Level{"e", LevelCategory::Computational, BasisBit::One},
    });
}

}  // namespace

// =========================================================================
// Construction: happy paths
// =========================================================================

TEST_CASE("NonComputationalModel: accepts a fully specified default model") {
    LevelSet levels = LevelSet::default_set();
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("DEPOLARIZE1", identity_transition(levels));
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(),
                                std::move(transitions), identity_classifier(levels),
                                NonComputationalPolicy{});
    REQUIRE(model.num_levels() == 5);
    REQUIRE(model.transitions().size() == 1);
    REQUIRE(model.classifier() != nullptr);
    REQUIRE(model.classifier()->num_levels() == 5);
}

TEST_CASE("NonComputationalModel: accepts a model with no classifier") {
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(), {}, std::nullopt,
                                NonComputationalPolicy{});
    REQUIRE(model.classifier() == nullptr);
    REQUIRE(model.transitions().empty());
}

TEST_CASE("NonComputationalModel: accepts an initial state at the sum tolerance boundary") {
    // Sum is 1 + 2^-52, representable and well within kProbTolerance.
    const double a = std::nextafter(std::nextafter(0.5, 1.0), 1.0);
    const double b = 0.5;
    REQUIRE(a + b > 1.0);  // guard: the inputs really do overshoot 1
    NonComputationalModel model(two_level_set(), {a, b}, {}, std::nullopt,
                                NonComputationalPolicy{});
    REQUIRE(model.num_levels() == 2);
}

// =========================================================================
// Construction: initial-state validation
// =========================================================================

TEST_CASE("NonComputationalModel: rejects initial state with wrong length") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel(LevelSet::default_set(), {0.5, 0.5}, {}, std::nullopt,
                              NonComputationalPolicy{}),
        ContainsSubstring("initial_state has 2 entries") && ContainsSubstring("expected 5"));
}

TEST_CASE("NonComputationalModel: rejects initial state entry out of [0, 1]") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel(LevelSet::default_set(), {1.5, -0.5, 0.0, 0.0, 0.0}, {}, std::nullopt,
                              NonComputationalPolicy{}),
        ContainsSubstring("initial_state entry 0") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("NonComputationalModel: rejects NaN initial state entry") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(NonComputationalModel(two_level_set(), {nan, 1.0}, {}, std::nullopt,
                                              NonComputationalPolicy{}),
                        ContainsSubstring("not finite"));
}

TEST_CASE("NonComputationalModel: rejects initial state that does not sum to 1") {
    REQUIRE_THROWS_WITH(NonComputationalModel(two_level_set(), {0.2, 0.2}, {}, std::nullopt,
                                              NonComputationalPolicy{}),
                        ContainsSubstring("sums to") && ContainsSubstring("must sum to 1"));
}

// =========================================================================
// Construction: transition validation
// =========================================================================

TEST_CASE("NonComputationalModel: rejects an unknown transition gate key") {
    LevelSet levels = LevelSet::default_set();
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("NOT_A_GATE", identity_transition(levels));
    REQUIRE_THROWS_WITH(
        NonComputationalModel(LevelSet::default_set(), default_initial_state(),
                              std::move(transitions), std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("NOT_A_GATE") && ContainsSubstring("not a recognized gate name"));
}

TEST_CASE("NonComputationalModel: rejects a transition spanning the wrong number of levels") {
    LevelSet small = two_level_set();
    std::map<std::string, TransitionInstrument> transitions;
    // Instrument built against the 2-level set, used in a 5-level model.
    transitions.emplace("DEPOLARIZE1", identity_transition(small));
    REQUIRE_THROWS_WITH(
        NonComputationalModel(LevelSet::default_set(), default_initial_state(),
                              std::move(transitions), std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("DEPOLARIZE1") && ContainsSubstring("spans 2 levels") &&
            ContainsSubstring("expected 5"));
}

// =========================================================================
// Construction: classifier validation
// =========================================================================

TEST_CASE("NonComputationalModel: rejects a classifier spanning the wrong number of levels") {
    LevelSet small = two_level_set();
    // Classifier built against the 2-level set, used in a 5-level model.
    MeasurementClassifier wrong = MeasurementClassifier::from_matrix({"0", "1"},
                                                                     {
                                                                         {1, 0},
                                                                         {0, 1},
                                                                     },
                                                                     small);
    REQUIRE_THROWS_WITH(
        NonComputationalModel(LevelSet::default_set(), default_initial_state(), {},
                              std::move(wrong), NonComputationalPolicy{}),
        ContainsSubstring("classifier spans 2 levels") && ContainsSubstring("expected 5"));
}

// =========================================================================
// Accessors
// =========================================================================

TEST_CASE("NonComputationalModel: initial_probability returns per-level values") {
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(), {}, std::nullopt,
                                NonComputationalPolicy{});
    REQUIRE_THAT(model.initial_probability(0), WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(model.initial_probability(4), WithinAbs(0.05, 1e-12));
}

TEST_CASE("NonComputationalModel: initial_probability throws on out-of-range level") {
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(), {}, std::nullopt,
                                NonComputationalPolicy{});
    REQUIRE_THROWS_WITH(model.initial_probability(5), ContainsSubstring("out of range"));
}

TEST_CASE("NonComputationalModel: transition_for resolves known gates and misses absent ones") {
    LevelSet levels = LevelSet::default_set();
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("DEPOLARIZE1", identity_transition(levels));
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(),
                                std::move(transitions), std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.transition_for("DEPOLARIZE1") != nullptr);
    REQUIRE(model.transition_for("DEPOLARIZE1")->num_levels() == 5);
    REQUIRE(model.transition_for("CX") == nullptr);
}

TEST_CASE("NonComputationalModel: policy accessor reflects the constructed policy") {
    NonComputationalPolicy policy;
    policy.reset_restores_lost = true;
    NonComputationalModel model(LevelSet::default_set(), default_initial_state(), {}, std::nullopt,
                                policy);
    REQUIRE(model.policy().reset_restores_lost == true);
    REQUIRE(model.policy().unknown_source_policy == UnknownSourcePolicy::Reject);
}
