// Tests for the static pre-sampling validation (validate_static), exercised
// end-to-end through sample_noncomputational.
//
// Rejection cases: all must throw std::invalid_argument before any shot is
// drawn (shots=1 with any seed; the static check fires before the first shot).
// Precision cases: false-positive guards -- these must sample cleanly.

#include "clifft/circuit/parser.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

// Level indices matching the fixed five-level set.
constexpr uint8_t kG = 0;
constexpr uint8_t kE = 1;
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLeakE = 3;
constexpr uint8_t kLost = 4;

// 5x5 zero matrix.
std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Transition matrix where both g and e leak to leak_e at rate `p`.
std::vector<std::vector<double>> leak_to_leak_e(double p) {
    auto m = zeros5();
    m[kLeakE][kG] = p;
    m[kLeakE][kE] = p;
    return m;
}

// Binary classifier: slot 0 reads symbol 0, all others read symbol 0;
// computational columns read out faithfully; noncomp level `level` reads `col`.
ClassifierSpec binary_classifier(uint8_t level, std::vector<double> col) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (int l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][kE] = 0.0;
    m[1][kE] = 1.0;
    m[0][level] = col[0];
    m[1][level] = col[1];
    return ClassifierSpec{2, std::move(m)};
}

NonComputationalModel model_with_leak_transition(const std::string& key, double rate,
                                                 bool with_classifier = true) {
    std::map<std::string, std::vector<std::vector<double>>> transitions{
        {key, leak_to_leak_e(rate)}};
    std::optional<ClassifierSpec> classifier;
    if (with_classifier) {
        classifier = binary_classifier(kLeakE, {0.0, 1.0});
    }
    return NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, transitions, classifier,
                                            NonComputationalPolicy{});
}

}  // namespace

// =========================================================================
// Rejection cases
// =========================================================================

TEST_CASE("static_check: MX after a low-rate leak always rejects") {
    // A 0.01-rate leak to leak_e means leak_e is reachable; MX on a
    // leak_e qubit is not representable.  Before this check, the circuit
    // sampled cleanly on most seeds (the low rate rarely fires on shot 0).
    auto model = model_with_leak_transition("S", 0.01);
    auto circuit = parse("S 0\nMX 0");

    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 1, 1),
                        ContainsSubstring("MX") && ContainsSubstring("not representable") &&
                            ContainsSubstring("before sampling"));
}

TEST_CASE("static_check: MPP parity measurement after a low-rate leak always rejects") {
    // Same reachability: leak_e is reachable on qubit 0 after the S
    // transition; MPP over qubit 0 is not representable.
    auto model = model_with_leak_transition("S", 0.01);
    auto circuit = parse("S 0\nMPP X0*X1");

    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 1, 1),
                        ContainsSubstring("MPP") && ContainsSubstring("not representable") &&
                            ContainsSubstring("before sampling"));
}

TEST_CASE("static_check: measurement without a classifier always rejects") {
    // A model with no classifier: any shot that finds qubit 0 leaked
    // after S would need a classifier for the M record.  The static
    // check catches this before drawing any shot.
    auto model = model_with_leak_transition("S", 0.01, /*with_classifier=*/false);
    auto circuit = parse("S 0\nM 0");

    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 1, 1),
                        ContainsSubstring("requires a classifier, but the model has none") &&
                            ContainsSubstring("before sampling"));
}

// =========================================================================
// Precision cases (false-positive guards)
// =========================================================================

TEST_CASE("static_check: R restores a leaked qubit before MX -- no false positive") {
    // A leaked qubit is always restored by R (leaked always restores),
    // so MX after R meets only a Computational qubit.  Sampling must run.
    auto model = model_with_leak_transition("S", 0.01);
    auto circuit = parse("S 0\nR 0\nMX 0");

    auto result = sample_noncomputational(circuit, model, 8, 1);
    REQUIRE(result.shots == 8);
    REQUIRE(result.measurements.size() == 8);
}

TEST_CASE("static_check: reset_restores_lost controls whether Lost survives R") {
    // LOSS(0.5) on qubit 0: Lost is reachable.  M requires a classifier
    // (no classifier provided); behavior depends on reset_restores_lost.

    // Classifier-less model, no reset_restores_lost: Lost survives R, so
    // M can meet a Lost qubit without a classifier -- rejects.
    {
        NonComputationalPolicy policy;
        policy.reset_restores_lost = false;
        auto model =
            NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {}, std::nullopt, policy);
        auto circuit = parse("H 0\nLOSS(0.5) 0\nR 0\nM 0");
        REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 1, 1),
                            ContainsSubstring("requires a classifier, but the model has none"));
    }

    // With reset_restores_lost=true: R restores Lost to Computational,
    // so M meets only a Computational qubit -- no classifier needed, samples.
    {
        NonComputationalPolicy policy;
        policy.reset_restores_lost = true;
        auto model =
            NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {}, std::nullopt, policy);
        auto circuit = parse("H 0\nLOSS(0.5) 0\nR 0\nM 0");
        auto result = sample_noncomputational(circuit, model, 8, 1);
        REQUIRE(result.shots == 8);
        REQUIRE(result.measurements.size() == 8);
    }
}

TEST_CASE("static_check: a zero-fire transition never adds leak to the reachable set") {
    // A seepage-only transition (both computational columns zero) cannot
    // fire on a Computational qubit; leak_e is never reachable from the
    // initial state.  MX after S must sample cleanly.
    std::vector<std::vector<double>> seep = zeros5();
    seep[kE][kLeakE] = 1.0;  // leak_e -> e; computational columns zero
    std::map<std::string, std::vector<std::vector<double>>> transitions{{"S", seep}};
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, transitions,
                                                  std::nullopt, NonComputationalPolicy{});
    auto circuit = parse("S 0\nMX 0");

    auto result = sample_noncomputational(circuit, model, 8, 1);
    REQUIRE(result.shots == 8);
    REQUIRE(result.measurements.size() == 8);
}

TEST_CASE("static_check: a leak on qubit 0 does not affect qubit 1's reachable set") {
    // The 0.01-rate leak transition on S annotates qubit 0 only; qubit 1
    // is a spectator.  MX on qubit 1 must sample cleanly.
    auto model = model_with_leak_transition("S", 0.01);
    auto circuit = parse("S 0\nMX 1");

    auto result = sample_noncomputational(circuit, model, 8, 1);
    REQUIRE(result.shots == 8);
    REQUIRE(result.measurements.size() == 8);
}

TEST_CASE("static check: a certain recapture retires the noncomputational member") {
    // The qubit starts entirely on leak_e; the S hook's leak_e column
    // recaptures to g with certainty, so MX always meets a computational
    // qubit. The no-event branch of the leak_e source is unreachable, so
    // the member must not survive the transition in the abstract walk.
    auto m = zeros5();
    m[kLeakE][kG] = 0.1;
    m[kLeakE][kE] = 0.1;
    m[kG][kLeakE] = 1.0;
    auto model = NonComputationalModel::from_spec({0.0, 0.0, 0.0, 1.0, 0.0}, {{"S", m}},
                                                  std::nullopt, NonComputationalPolicy{});
    auto circuit = parse("S 0\nMX 0");

    auto result = sample_noncomputational(circuit, model, 4, 1);
    REQUIRE(result.shots == 4);
    for (uint32_t shot = 0; shot < 4; ++shot) {
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
}

TEST_CASE("static check: a re-leak after the certain recapture still rejects") {
    // Same model, but a second S: after the certain recapture the
    // computational columns leak again (rate 0.1), so at MX the leaked
    // status is genuinely reachable and the pair must reject.
    auto m = zeros5();
    m[kLeakE][kG] = 0.1;
    m[kLeakE][kE] = 0.1;
    m[kG][kLeakE] = 1.0;
    auto model = NonComputationalModel::from_spec({0.0, 0.0, 0.0, 1.0, 0.0}, {{"S", m}},
                                                  std::nullopt, NonComputationalPolicy{});
    auto circuit = parse("S 0\nS 0\nMX 0");

    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 1, 1),
                        ContainsSubstring("MX") && ContainsSubstring("not representable") &&
                            ContainsSubstring("before sampling"));
}
