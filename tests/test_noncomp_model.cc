#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::GateType;
using clifft::kAllLevels;
using clifft::Level;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::test::opaque_nan;

namespace {

// All-zero transition matrix: every source has no-jump weight 1, i.e.
// nothing happens. The honest no-op default.
std::vector<std::vector<double>> zero_matrix() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Identity readout on g/e; leak_g/lost read "0", leak_e reads "1".
std::vector<std::vector<double>> identity_classifier() {
    return {
        {1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0},
    };
}

// A valid probability vector over the 5 levels.
std::vector<double> default_initial_state() {
    return {0.5, 0.3, 0.1, 0.05, 0.05};
}

double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) {
        s += x;
    }
    return s;
}

}  // namespace

// =========================================================================
// Construction: happy paths
// =========================================================================

TEST_CASE("NonComputationalModel: accepts a fully specified model") {
    auto model = NonComputationalModel::from_spec(default_initial_state(), {{"H", zero_matrix()}},
                                                  std::make_optional(identity_classifier()),
                                                  NonComputationalPolicy{});
    REQUIRE(model.transitions().size() == 1);
    REQUIRE(model.classifier() != nullptr);
}

TEST_CASE("NonComputationalModel: accepts a model with no classifier") {
    auto model = NonComputationalModel::from_spec(default_initial_state(), {}, std::nullopt,
                                                  NonComputationalPolicy{});
    REQUIRE(model.classifier() == nullptr);
    REQUIRE(model.transitions().empty());
}

TEST_CASE("NonComputationalModel: accepts an initial state at the sum tolerance boundary") {
    // Sum is 1 + 2^-52, representable and well within kProbTolerance.
    const double a = std::nextafter(std::nextafter(0.5, 1.0), 1.0);
    const double b = 0.5;
    REQUIRE(a + b > 1.0);  // guard: the inputs really do overshoot 1
    REQUIRE_NOTHROW(NonComputationalModel::from_spec({a, b, 0.0, 0.0, 0.0}, {}, std::nullopt,
                                                     NonComputationalPolicy{}));
}

TEST_CASE("NonComputationalModel: normalizes the stored initial state") {
    // Sum is 1 + 1e-13: inside kProbTolerance but well above 1e-15, so a
    // normalized vector is distinguishable from the raw input.
    const std::vector<double> raw = {0.5 + 1e-13, 0.5, 0.0, 0.0, 0.0};
    REQUIRE(sum(raw) > 1.0 + 1e-15);  // guard: raw input is not already normalized
    auto model = NonComputationalModel::from_spec(raw, {}, std::nullopt, NonComputationalPolicy{});
    double stored_sum = 0.0;
    for (Level level : kAllLevels) {
        stored_sum += model.initial_probability(level);
    }
    REQUIRE_THAT(stored_sum, WithinAbs(1.0, 1e-15));
}

TEST_CASE("NonComputationalModel: alias key is stored verbatim and hooks the canonical gate") {
    auto model = NonComputationalModel::from_spec(
        default_initial_state(), {{"CNOT", zero_matrix()}}, std::nullopt, NonComputationalPolicy{});
    // Stored under the original key; the hook resolves the canonical gate.
    REQUIRE(model.transitions().count("CNOT") == 1);
    REQUIRE(model.transition_hooks().at(GateType::CX) == "CNOT");
    REQUIRE(model.transition_named("CNOT") != nullptr);
    // Named lookup is exact-key: the canonical spelling is not a key here.
    REQUIRE(model.transition_named("CX") == nullptr);
}

// =========================================================================
// Construction: initial-state validation
// =========================================================================

TEST_CASE("NonComputationalModel: rejects initial state with wrong length") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec({0.5, 0.5}, {}, std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("initial_state has 2 entries") && ContainsSubstring("expected 5"));
}

TEST_CASE("NonComputationalModel: rejects initial state entry out of [0, 1]") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec({1.5, -0.5, 0.0, 0.0, 0.0}, {}, std::nullopt,
                                         NonComputationalPolicy{}),
        ContainsSubstring("initial_state entry 0") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("NonComputationalModel: rejects NaN initial state entry") {
    const double nan = opaque_nan();
    REQUIRE_THROWS_WITH(NonComputationalModel::from_spec({nan, 1.0, 0.0, 0.0, 0.0}, {},
                                                         std::nullopt, NonComputationalPolicy{}),
                        ContainsSubstring("not finite"));
}

TEST_CASE("NonComputationalModel: rejects initial state that does not sum to 1") {
    REQUIRE_THROWS_WITH(NonComputationalModel::from_spec({0.2, 0.2, 0.0, 0.0, 0.0}, {},
                                                         std::nullopt, NonComputationalPolicy{}),
                        ContainsSubstring("sums to") && ContainsSubstring("must sum to 1"));
}

// =========================================================================
// Construction: transition validation
// =========================================================================

TEST_CASE("NonComputationalModel: a non-gate transition key is a named transition, not a hook") {
    auto model =
        NonComputationalModel::from_spec(default_initial_state(), {{"my_leak", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.transition_named("my_leak") != nullptr);
    REQUIRE(model.transition_hooks().empty());
}

TEST_CASE(
    "NonComputationalModel: rejects a transition key a LEVEL_TRANSITION tag cannot reference") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"bad]key", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("bad]key") && ContainsSubstring("cannot be referenced"));
}

TEST_CASE("NonComputationalModel: a non-hookable gate-named key is a named-only transition") {
    // Keys naming non-hookable instructions (noise channels, annotations,
    // LOSS itself) register no hook, but stay referenceable from a
    // LEVEL_TRANSITION[key] annotation like any other name.
    auto model = NonComputationalModel::from_spec(
        default_initial_state(),
        {{"DEPOLARIZE1", zero_matrix()}, {"TICK", zero_matrix()}, {"LOSS", zero_matrix()}},
        std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.transition_named("DEPOLARIZE1") != nullptr);
    REQUIRE(model.transition_named("TICK") != nullptr);
    REQUIRE(model.transition_named("LOSS") != nullptr);
    REQUIRE(model.transition_hooks().empty());
}

TEST_CASE("NonComputationalModel: rejects two keys resolving to the same gate") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(),
                                         {{"CX", zero_matrix()}, {"CNOT", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("both resolve to gate 'CX'"));
}

TEST_CASE("NonComputationalModel: a malformed transition matrix rejects, naming the component") {
    auto bad = zero_matrix();
    bad[0][0] = 1.5;
    REQUIRE_THROWS_WITH(NonComputationalModel::from_spec(default_initial_state(), {{"H", bad}},
                                                         std::nullopt, NonComputationalPolicy{}),
                        ContainsSubstring("TransitionInstrument") &&
                            ContainsSubstring("out of [0, 1]") && ContainsSubstring("'H'"));
}

// =========================================================================
// Accessors
// =========================================================================

TEST_CASE("NonComputationalModel: initial_probability returns per-level values") {
    auto model = NonComputationalModel::from_spec(default_initial_state(), {}, std::nullopt,
                                                  NonComputationalPolicy{});
    REQUIRE_THAT(model.initial_probability(Level::G), WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(model.initial_probability(Level::Lost), WithinAbs(0.05, 1e-12));
}

TEST_CASE("NonComputationalModel: transition hooks resolve known gates and miss absent ones") {
    auto model = NonComputationalModel::from_spec(default_initial_state(), {{"H", zero_matrix()}},
                                                  std::nullopt, NonComputationalPolicy{});
    REQUIRE(model.transition_hooks().at(GateType::H) == "H");
    REQUIRE(model.transition_named("H") != nullptr);
    REQUIRE(model.transition_hooks().count(GateType::CX) == 0);
    // Named lookup misses absent keys rather than throwing.
    REQUIRE(model.transition_named("NOT_A_KEY") == nullptr);
}

TEST_CASE("NonComputationalModel: policy accessor reflects the constructed policy") {
    NonComputationalPolicy policy;
    policy.reset_restores_lost = true;
    auto model =
        NonComputationalModel::from_spec(default_initial_state(), {}, std::nullopt, policy);
    REQUIRE(model.policy().reset_restores_lost == true);
}
