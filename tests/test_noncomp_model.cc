#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/noncomp/annotate.h"
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
#include <string_view>
#include <unordered_map>
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
        ContainsSubstring("bad]key") &&
            ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
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

// =========================================================================
// Construction: dead-hook key rejection
// =========================================================================

TEST_CASE("NonComputationalModel: rejects MXX as a hook key (desugars to MPP)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"MXX", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("desugars to MPP"));
}

TEST_CASE("NonComputationalModel: rejects MYY as a hook key (desugars to MPP)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"MYY", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("desugars to MPP"));
}

TEST_CASE("NonComputationalModel: rejects MZZ as a hook key (desugars to MPP)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"MZZ", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("desugars to MPP"));
}

TEST_CASE("NonComputationalModel: rejects CH as a hook key (parser-only rewrite)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"CH", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("parser decomposes"));
}

TEST_CASE("NonComputationalModel: rejects CCX as a hook key (parser-only rewrite)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"CCX", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("parser decomposes"));
}

TEST_CASE("NonComputationalModel: rejects CCZ as a hook key (parser-only rewrite)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"CCZ", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("parser decomposes"));
}

TEST_CASE("NonComputationalModel: rejects I as a hook key (identity no-op)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"I", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("identity no-ops emit no circuit nodes"));
}

TEST_CASE("NonComputationalModel: rejects II as a hook key (identity no-op)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"II", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("identity no-ops emit no circuit nodes"));
}

TEST_CASE("NonComputationalModel: rejects I_ERROR as a hook key (identity no-op)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"I_ERROR", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("identity no-ops emit no circuit nodes"));
}

TEST_CASE("NonComputationalModel: rejects II_ERROR as a hook key (identity no-op)") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"II_ERROR", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("identity no-ops emit no circuit nodes"));
}

// =========================================================================
// Construction: tag grammar validation
// =========================================================================

TEST_CASE("NonComputationalModel: rejects empty transition key") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys with leading whitespace") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{" padded", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys with trailing whitespace") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"padded ", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys that are only whitespace") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{" ", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys containing ']'") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"a]b", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys containing '#'") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"a#b", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: rejects keys containing a newline") {
    REQUIRE_THROWS_WITH(
        NonComputationalModel::from_spec(default_initial_state(), {{"a\nb", zero_matrix()}},
                                         std::nullopt, NonComputationalPolicy{}),
        ContainsSubstring("cannot be written as a LEVEL_TRANSITION tag"));
}

TEST_CASE("NonComputationalModel: accepts keys with internal spaces") {
    REQUIRE_NOTHROW(NonComputationalModel::from_spec(
        default_initial_state(), {{"a b", zero_matrix()}}, std::nullopt, NonComputationalPolicy{}));
}

TEST_CASE("NonComputationalModel: accepts keys with '[' (only ']' is banned)") {
    REQUIRE_NOTHROW(NonComputationalModel::from_spec(
        default_initial_state(), {{"a[b", zero_matrix()}}, std::nullopt, NonComputationalPolicy{}));
}

TEST_CASE("NonComputationalModel: accepts keys with hyphens and digits") {
    REQUIRE_NOTHROW(NonComputationalModel::from_spec(default_initial_state(),
                                                     {{"T1-decay", zero_matrix()}}, std::nullopt,
                                                     NonComputationalPolicy{}));
}

TEST_CASE("NonComputationalModel: accepted weird keys survive LEVEL_TRANSITION tag round-trip") {
    // Each key must parse back unchanged from a LEVEL_TRANSITION[key] circuit line.
    const std::vector<std::string> weird_keys = {"a b", "a[b", "T1-decay"};
    for (const auto& key : weird_keys) {
        INFO("key: '" + key + "'");
        auto model =
            NonComputationalModel::from_spec(default_initial_state(), {{key, zero_matrix()}},
                                             std::nullopt, NonComputationalPolicy{});
        std::string circuit_text = "LEVEL_TRANSITION[" + key + "] 0\n";
        auto circuit = clifft::parse(circuit_text);
        REQUIRE(circuit.nodes.size() == 1);
        REQUIRE(circuit.nodes[0].tag == key);
        REQUIRE(model.transition_named(key) != nullptr);
    }
}

// =========================================================================
// Hook-fires invariant: every registered hook names a gate the parser emits
// =========================================================================

TEST_CASE("model: every registered gate hook names a gate the parser can produce") {
    // A gate hook can fire only if its gate type appears in a parsed
    // circuit. For every key the model registers as a hook, parse a sample
    // line for that gate and require both a node of that type and the
    // expanded annotation after it. The table must cover exactly the
    // hookable gates: a missing or stale entry fails below.
    const std::unordered_map<std::string_view, std::string_view> kSampleLines = {
        {"H", "H 0"},
        {"S", "S 0"},
        {"S_DAG", "S_DAG 0"},
        {"X", "X 0"},
        {"Y", "Y 0"},
        {"Z", "Z 0"},
        {"SQRT_X", "SQRT_X 0"},
        {"SQRT_X_DAG", "SQRT_X_DAG 0"},
        {"SQRT_Y", "SQRT_Y 0"},
        {"SQRT_Y_DAG", "SQRT_Y_DAG 0"},
        {"H_XY", "H_XY 0"},
        {"H_YZ", "H_YZ 0"},
        {"H_NXY", "H_NXY 0"},
        {"H_NXZ", "H_NXZ 0"},
        {"H_NYZ", "H_NYZ 0"},
        {"C_XYZ", "C_XYZ 0"},
        {"C_ZYX", "C_ZYX 0"},
        {"C_NXYZ", "C_NXYZ 0"},
        {"C_NZYX", "C_NZYX 0"},
        {"C_XNYZ", "C_XNYZ 0"},
        {"C_XYNZ", "C_XYNZ 0"},
        {"C_ZNYX", "C_ZNYX 0"},
        {"C_ZYNX", "C_ZYNX 0"},
        {"T", "T 0"},
        {"T_DAG", "T_DAG 0"},
        {"R_X", "R_X(0.25) 0"},
        {"R_Y", "R_Y(0.25) 0"},
        {"R_Z", "R_Z(0.25) 0"},
        {"U3", "U3(0.1,0.2,0.3) 0"},
        {"R_XX", "R_XX(0.25) 0 1"},
        {"R_YY", "R_YY(0.25) 0 1"},
        {"R_ZZ", "R_ZZ(0.25) 0 1"},
        {"R_PAULI", "R_PAULI(0.1) X0*Y1"},
        {"CX", "CX 0 1"},
        {"CY", "CY 0 1"},
        {"CZ", "CZ 0 1"},
        {"SWAP", "SWAP 0 1"},
        {"ISWAP", "ISWAP 0 1"},
        {"ISWAP_DAG", "ISWAP_DAG 0 1"},
        {"SQRT_XX", "SQRT_XX 0 1"},
        {"SQRT_XX_DAG", "SQRT_XX_DAG 0 1"},
        {"SQRT_YY", "SQRT_YY 0 1"},
        {"SQRT_YY_DAG", "SQRT_YY_DAG 0 1"},
        {"SQRT_ZZ", "SQRT_ZZ 0 1"},
        {"SQRT_ZZ_DAG", "SQRT_ZZ_DAG 0 1"},
        {"CXSWAP", "CXSWAP 0 1"},
        {"CZSWAP", "CZSWAP 0 1"},
        {"SWAPCX", "SWAPCX 0 1"},
        {"XCX", "XCX 0 1"},
        {"XCY", "XCY 0 1"},
        {"XCZ", "XCZ 0 1"},
        {"YCX", "YCX 0 1"},
        {"YCY", "YCY 0 1"},
        {"YCZ", "YCZ 0 1"},
        {"M", "M 0"},
        {"MX", "MX 0"},
        {"MY", "MY 0"},
        {"MR", "MR 0"},
        {"MRX", "MRX 0"},
        {"MRY", "MRY 0"},
        {"MPP", "MPP X0*X1"},
        {"R", "R 0"},
        {"RX", "RX 0"},
        {"RY", "RY 0"},
    };

    size_t hooked_gates = 0;
    for (size_t i = 0; i < static_cast<size_t>(clifft::GateType::UNKNOWN); ++i) {
        const auto g = static_cast<clifft::GateType>(i);
        const std::string name{clifft::gate_name(g)};
        // Build a model with a single hook for this gate; skip if the model
        // constructor rejects the key (rejection tests cover those cases).
        std::optional<NonComputationalModel> maybe_model;
        try {
            maybe_model =
                NonComputationalModel::from_spec(default_initial_state(), {{name, zero_matrix()}},
                                                 std::nullopt, NonComputationalPolicy{});
        } catch (const std::invalid_argument&) {
            continue;
        }
        const auto& model = *maybe_model;
        // Only check gates that registered as hooks.
        if (model.transition_hooks().find(g) == model.transition_hooks().end()) {
            continue;
        }
        hooked_gates++;
        // The hook must have a sample parse line; missing entries mean a future
        // author added a hookable gate without adding parse coverage here.
        auto it = kSampleLines.find(clifft::gate_name(g));
        INFO("No sample parse line for hookable gate " + name +
             ": add an entry to kSampleLines so the hook-fires invariant is covered");
        REQUIRE(it != kSampleLines.end());
        // Parse the sample line; at least one node must have gate == g.
        auto circuit = clifft::parse(std::string(it->second) + "\n");
        bool found = false;
        for (const auto& node : circuit.nodes) {
            if (node.gate == g) {
                found = true;
                break;
            }
        }
        REQUIRE(found);
        // annotate() must insert a LEVEL_TRANSITION node after the gate.
        auto annotated = clifft::annotate(circuit, model);
        bool found_lt = false;
        for (const auto& node : annotated.nodes) {
            if (node.gate == clifft::GateType::LEVEL_TRANSITION && node.tag == name) {
                found_lt = true;
                break;
            }
        }
        REQUIRE(found_lt);
    }
    // Both directions: zero hooked gates would make the loop vacuous, and
    // an unvisited table entry names a gate that is no longer hookable.
    REQUIRE(hooked_gates == kSampleLines.size());
}
