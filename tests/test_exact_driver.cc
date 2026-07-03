// End-to-end tests for exact-mode sampling: the driver loop, continuation
// cache, frame-preloaded initials, and trap resolution behind
// sample_noncomputational with unknown_source_policy = Exact.
//
// Deterministic pins use certain (p = 1) channels; the exact-vs-AOT
// agreement checks use source-independent rates, where both paths are
// exact, with generous statistical margins (the full distributional
// campaign is a later step).

#include "clifft/circuit/parser.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLeak = 3;
constexpr uint8_t kLost = 4;

struct ModelSpec {
    double leak_from_g = 0.0;
    double leak_from_e = 0.0;
    double seep_to_e = 0.0;
    std::vector<double> initial = {1.0, 0.0, 0.0, 0.0, 0.0};
    DampingPolicy damping = DampingPolicy::Exact;
    UnknownSourcePolicy source_policy = UnknownSourcePolicy::Exact;
};

NonComputationalModel make_model(const ModelSpec& spec) {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][0] = spec.leak_from_g;
    leak[kLeak][1] = spec.leak_from_e;
    leak[1][kLeak] = spec.seep_to_e;

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1"};
    classifier.matrix = {{1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.unknown_source_policy = spec.source_policy;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    policy.damping = spec.damping;
    return NonComputationalModel::from_spec(levels, spec.initial, {{"leak", leak}}, classifier,
                                            policy);
}

double mean_of(const std::vector<uint8_t>& bits, uint32_t stride, uint32_t index) {
    double sum = 0.0;
    size_t n = 0;
    for (size_t i = index; i < bits.size(); i += stride) {
        sum += bits[i];
        ++n;
    }
    return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

}  // namespace

TEST_CASE("exact: an untrapped run reproduces plain sampling behavior") {
    // No annotations at all: the exact path is one shared module executed
    // per shot. A Bell pair's measurements must be perfectly correlated.
    ModelSpec spec;
    auto model = make_model(spec);
    auto circuit = parse("H 0\nCX 0 1\nM 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 200, 7);
    REQUIRE(result.measurements.size() == 400);
    int ones = 0;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        REQUIRE(result.measurements[shot * 2] == result.measurements[shot * 2 + 1]);
        ones += result.measurements[shot * 2];
    }
    REQUIRE(ones > 50);
    REQUIRE(ones < 150);
    for (const QubitStatus& s : result.final_status) {
        REQUIRE(s.kind() != QubitStatusKind::Leaked);
        REQUIRE(s.kind() != QubitStatusKind::Lost);
    }
}

TEST_CASE("exact: a certain leak traps, classifies, and reports the status") {
    // From |1> (X prep), the site fires every shot; the leaked qubit's
    // measurement classifies (leaked column reads 1) and the sidecar
    // carries the leaked status. Qubit 1 is untouched.
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    auto model = make_model(spec);
    auto circuit = parse("X 0\nLEVEL_TRANSITION[leak] 0\nM 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 25, 3);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);      // classifier: leaked reads 1
        REQUIRE(result.measurements[shot * 2 + 1] == 0);  // spectator
        REQUIRE(result.final_status[shot * 2].kind() == QubitStatusKind::Leaked);
        REQUIRE(result.final_status[shot * 2 + 1].kind() == QubitStatusKind::ComputationalKnown);
    }
}

TEST_CASE("exact: a known |1> initial preloads the frame without a distinct module") {
    // Initial state entirely on e: every shot must measure 1 through the
    // frame preload alone.
    ModelSpec spec;
    spec.initial = {0.0, 1.0, 0.0, 0.0, 0.0};
    auto model = make_model(spec);
    auto circuit = parse("LEVEL_TRANSITION[leak] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 25, 11);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot] == 1);
    }
}

TEST_CASE("exact: a noncomputational initial compiles its own continuation") {
    // Initial state entirely on the lost level: no trap ever fires (the
    // annotation is a classical consult), the measurement classifies from
    // the lost column, and the final status stays Lost.
    ModelSpec spec;
    spec.initial = {0.0, 0.0, 0.0, 0.0, 1.0};
    auto model = make_model(spec);
    auto circuit = parse("LEVEL_TRANSITION[leak] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 10, 5);
    for (uint32_t shot = 0; shot < 10; ++shot) {
        REQUIRE(result.measurements[shot] == 0);  // the fixture's lost column reads 0
        REQUIRE(result.final_status[shot].kind() == QubitStatusKind::Lost);
    }
}

TEST_CASE("exact: seepage after a trap recaptures through a classical consult") {
    // Certain leak, then certain seepage back to e at the next site, then
    // a measurement: every shot reads 1 with a computational final
    // status. The recapture path exercises pre-drawn classical outcomes
    // and the multi-annotation continuation.
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    spec.seep_to_e = 1.0;
    auto model = make_model(spec);
    auto circuit = parse("X 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 13);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);
        REQUIRE(result.final_status[shot].kind() == QubitStatusKind::ComputationalKnown);
    }
}

TEST_CASE("exact: agreement with the AOT path where both are exact") {
    // A source-independent rate (p_g = p_e) is exact under the AOT
    // sampler too. Compare the leaked-measurement marginal over many
    // shots with a generous margin: p(leak) = 0.3, and a leaked qubit
    // reads 1 while a computational |0> reads 0.
    const uint32_t shots = 4000;
    auto circuit = parse("LEVEL_TRANSITION[leak] 0\nM 0");

    ModelSpec exact_spec;
    exact_spec.leak_from_g = 0.3;
    exact_spec.leak_from_e = 0.3;
    auto exact_result = sample_noncomputational(circuit, make_model(exact_spec), shots, 17);

    ModelSpec aot_spec = exact_spec;
    aot_spec.source_policy = UnknownSourcePolicy::Reject;  // AOT path, exact here
    auto aot_result = sample_noncomputational(circuit, make_model(aot_spec), shots, 17);

    const double exact_mean = mean_of(exact_result.measurements, 1, 0);
    const double aot_mean = mean_of(aot_result.measurements, 1, 0);
    // Binomial std at p = 0.3 over 4000 shots is ~0.007; allow 5 sigma
    // between each mean and the true rate.
    REQUIRE(std::abs(exact_mean - 0.3) < 0.04);
    REQUIRE(std::abs(aot_mean - 0.3) < 0.04);
}

TEST_CASE("exact: same seed reproduces identical runs") {
    ModelSpec spec;
    spec.leak_from_e = 0.5;
    spec.initial = {0.5, 0.5, 0.0, 0.0, 0.0};
    auto model = make_model(spec);
    auto circuit = parse("H 1\nCX 1 0\nLEVEL_TRANSITION[leak] 0\nM 0\nM 1");

    auto a = sample_noncomputational(circuit, model, 100, 23);
    auto b = sample_noncomputational(circuit, model, 100, 23);
    REQUIRE(a.measurements == b.measurements);
    REQUIRE(a.heralds == b.heralds);
    for (size_t i = 0; i < a.final_status.size(); ++i) {
        REQUIRE(a.final_status[i].kind() == b.final_status[i].kind());
    }
}

TEST_CASE("exact: max_rank rejects an over-budget compile naming the line") {
    // Each dormant-random exact site adds one to k: three H-prefixed
    // sites push the peak to 3, over a cap of 2.
    ModelSpec spec;
    spec.leak_from_e = 0.2;
    auto model = make_model(spec);
    auto circuit = parse(
        "H 0\nH 1\nH 2\n"
        "LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 1\nLEVEL_TRANSITION[leak] 2\n"
        "M 0\nM 1\nM 2");

    REQUIRE_THROWS_WITH(
        sample_noncomputational(circuit, model, 5, 1, /*max_rank=*/2),
        ContainsSubstring("exceeds max_rank 2") && ContainsSubstring("circuit line"));
}

TEST_CASE("exact: a neglect-form trap is guarded until the correlated continuation lands") {
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    spec.leak_from_g = 0.5;
    spec.damping = DampingPolicy::Neglect;
    auto model = make_model(spec);
    auto circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nM 0");

    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 20, 29),
                        ContainsSubstring("neglect-form site fired"));
}

TEST_CASE("exact: ternary heralds ride the cache key") {
    // A three-symbol classifier whose leaked column always heralds: every
    // trapped shot's classified slot reports a herald, and the record bit
    // stays roughly fair across shots.
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][1] = 1.0;

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1", "herald"};
    classifier.matrix = {
        {1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.unknown_source_policy = UnknownSourcePolicy::Exact;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    auto model = NonComputationalModel::from_spec(levels, {0.0, 1.0, 0.0, 0.0, 0.0},
                                                  {{"leak", leak}}, classifier, policy);

    auto circuit = parse("LEVEL_TRANSITION[leak] 0\nM 0");
    auto result = sample_noncomputational(circuit, model, 200, 31);

    int ones = 0;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        REQUIRE(result.heralds[shot] == 1);
        ones += result.measurements[shot];
    }
    // Heralded slots carry a uniform bit.
    REQUIRE(ones > 60);
    REQUIRE(ones < 140);
}
