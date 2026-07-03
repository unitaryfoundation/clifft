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

TEST_CASE("exact: a neglect-form trap keeps the fire-side correlation") {
    // The decisive pin for the forced trace-out. Qubit 0 is Bell-entangled
    // with qubit 1 and dormant-random at the site; under neglect the fire
    // traps with the carrier uncollapsed. The channel is certain but
    // source-dependent in its destination (g -> leak_g, e -> leak_e), and
    // the classifier maps those levels to 0 and 1 -- so the classified
    // M 0 *is* the reported source, and the partner's M 1 must equal it
    // on every shot, because the continuation's trace-out is forced to
    // that same source. Independent redraw would mismatch half the time.
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeakG][0] = 1.0;  // from g: leak_g, certainly
    leak[kLeak][1] = 1.0;   // from e: leak_e, certainly

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1"};
    classifier.matrix = {{1.0, 0.0, 1.0, 0.0, 1.0},   // leak_g reads 0
                         {0.0, 1.0, 0.0, 1.0, 0.0}};  // leak_e reads 1

    NonComputationalPolicy policy;
    policy.unknown_source_policy = UnknownSourcePolicy::Exact;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec(levels, {1.0, 0.0, 0.0, 0.0, 0.0},
                                                  {{"leak", leak}}, classifier, policy);

    auto circuit = parse("H 0\nCX 0 1\nLEVEL_TRANSITION[leak] 0\nM 0\nM 1");
    auto result = sample_noncomputational(circuit, model, 100, 29);

    bool saw_zero = false;
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 100; ++shot) {
        const uint8_t classified = result.measurements[shot * 2];
        const uint8_t partner = result.measurements[shot * 2 + 1];
        REQUIRE(classified == partner);
        (classified == 0 ? saw_zero : saw_one) = true;
    }
    REQUIRE(saw_zero);
    REQUIRE(saw_one);
}

TEST_CASE("exact: a trap may insert a classical consult between pre-drawn ones") {
    // Both qubits start leaked; R 0; X 0 recaptures q0 to a definite |1>,
    // so q0's first annotation is quantum and traps (certain leak from
    // e), while q1's later annotation was already pre-drawn as a
    // classical consult. The trap turns q0's *second* annotation
    // classical at a position before q1's recorded outcome -- the shape
    // that breaks an append-only outcome stream. Post-fix: both
    // measurements classify as leaked (reads 1) on every shot.
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    spec.initial = {0.0, 0.0, 0.0, 1.0, 0.0};  // all mass on the leaked level
    auto model = make_model(spec);
    auto circuit = parse(
        "R 0\nX 0\n"
        "LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 1\n"
        "M 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 25, 41);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);
        REQUIRE(result.measurements[shot * 2 + 1] == 1);
        REQUIRE(result.final_status[shot * 2].kind() == QubitStatusKind::Leaked);
        REQUIRE(result.final_status[shot * 2 + 1].kind() == QubitStatusKind::Leaked);
    }
}

TEST_CASE("exact: a chain of two forced traps keeps both correlations") {
    // Two independent Bell pairs, each with a certain source-dependent
    // leak under neglect: every shot traps twice, forcing two trace-outs
    // at two different hidden slots, and the second continuation's prefix
    // contains the first's forced instruction (exercising the
    // sampling/forced mask in the debug prefix comparison). Both
    // partner correlations must survive.
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeakG][0] = 1.0;
    leak[kLeak][1] = 1.0;

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1"};
    classifier.matrix = {{1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.unknown_source_policy = UnknownSourcePolicy::Exact;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec(levels, {1.0, 0.0, 0.0, 0.0, 0.0},
                                                  {{"leak", leak}}, classifier, policy);

    auto circuit = parse(
        "H 0\nCX 0 1\nH 2\nCX 2 3\n"
        "LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 2\n"
        "M 0\nM 1\nM 2\nM 3");
    auto result = sample_noncomputational(circuit, model, 60, 43);

    for (uint32_t shot = 0; shot < 60; ++shot) {
        const uint8_t* m = result.measurements.data() + shot * 4;
        REQUIRE(m[0] == m[1]);
        REQUIRE(m[2] == m[3]);
    }
}

TEST_CASE("exact: a neglect fire onto a computational destination stays correlated") {
    // Under neglect every fire traps, including computational
    // destinations. A certain source-swap channel (g -> e, e -> g) on a
    // Bell-entangled dormant-random qubit: the continuation's forced
    // materialization collapses the partner to the source while the
    // trapped qubit re-preps at the destination, so the two measurements
    // must anti-correlate on every shot.
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> swap_ge(5, std::vector<double>(5, 0.0));
    swap_ge[1][0] = 1.0;  // g -> e, certainly
    swap_ge[0][1] = 1.0;  // e -> g, certainly

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1"};
    classifier.matrix = {{1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.unknown_source_policy = UnknownSourcePolicy::Exact;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec(levels, {1.0, 0.0, 0.0, 0.0, 0.0},
                                                  {{"swap", swap_ge}}, classifier, policy);

    auto circuit = parse("H 0\nCX 0 1\nLEVEL_TRANSITION[swap] 0\nM 0\nM 1");
    auto result = sample_noncomputational(circuit, model, 60, 47);

    bool saw_zero = false;
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 60; ++shot) {
        const uint8_t trapped = result.measurements[shot * 2];
        const uint8_t partner = result.measurements[shot * 2 + 1];
        REQUIRE(trapped == 1 - partner);
        (partner == 0 ? saw_zero : saw_one) = true;
    }
    REQUIRE(saw_zero);
    REQUIRE(saw_one);
}

TEST_CASE("exact: herald flags drawn in one continuation are reused by the next") {
    // The reuse invariant, made observable: with p_herald = 0.5 and a
    // deterministic not-heralded bit (P(1 | leak, not heralded) = 1), a
    // slot whose sidecar flag says not-heralded must always record 1 --
    // the record was drawn under the same flag that the sidecar reports.
    // If the second continuation redrew the flag independently, its patch
    // could disagree with the sidecar and break the invariant half the
    // time. The two-trap chain makes slot 0's flag be drawn while
    // compiling continuation one and reused by continuation two, where
    // the measurement actually executes.
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][1] = 1.0;  // certain leak from e

    ClassifierSpec classifier;
    classifier.symbols = {"0", "1", "herald"};
    classifier.matrix = {
        {1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 0.5, 0.0}, {0.0, 0.0, 0.0, 0.5, 0.0}};

    NonComputationalPolicy policy;
    policy.unknown_source_policy = UnknownSourcePolicy::Exact;
    policy.lost_leaked_ops = LostLeakedOpsPolicy::Drop;
    auto model = NonComputationalModel::from_spec(levels, {1.0, 0.0, 0.0, 0.0, 0.0},
                                                  {{"leak", leak}}, classifier, policy);

    auto circuit = parse("X 0\nX 1\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 1\nM 0\nM 1");
    auto result = sample_noncomputational(circuit, model, 200, 53);

    int heralded = 0;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint8_t herald = result.heralds[shot * 2 + slot];
            const uint8_t bit = result.measurements[shot * 2 + slot];
            if (herald == 0) {
                REQUIRE(bit == 1);  // the not-heralded conditional is deterministic
            }
            heralded += herald;
        }
    }
    REQUIRE(heralded > 120);  // p_herald = 0.5 over 400 slots
    REQUIRE(heralded < 280);
}

TEST_CASE("exact: spectator noise between two traps fires exactly once") {
    // Certain X errors on a spectator qubit, one between the two traps
    // and one after: the noise-gap cursor must fire each exactly once
    // across the two re-anchors, flipping the spectator twice back to 0.
    // A cursor that refires or skips across either resume flips the
    // deterministic record.
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    auto model = make_model(spec);
    auto circuit = parse(
        "X 0\nX 1\n"
        "LEVEL_TRANSITION[leak] 0\nX_ERROR(1) 2\nLEVEL_TRANSITION[leak] 1\nX_ERROR(1) 2\n"
        "M 2");

    auto result = sample_noncomputational(circuit, model, 20, 59);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 0);
        REQUIRE(result.final_status[shot * 3].kind() == QubitStatusKind::Leaked);
        REQUIRE(result.final_status[shot * 3 + 1].kind() == QubitStatusKind::Leaked);
    }
}

TEST_CASE("exact: a detector and observable span the trap boundary") {
    // The detector compares a pre-trap measurement (executed on the main
    // line) with a post-trap classified one (written by the
    // continuation): both read 1, so the parity is 0 on every shot, and
    // the observable carries the classified bit.
    ModelSpec spec;
    spec.leak_from_e = 1.0;
    auto model = make_model(spec);
    auto circuit = parse(
        "X 0\nM 0\nLEVEL_TRANSITION[leak] 0\nM 0\n"
        "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]");

    auto result = sample_noncomputational(circuit, model, 20, 61);
    REQUIRE(result.num_detectors == 1);
    REQUIRE(result.num_observables == 1);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);      // pre-trap Born measurement
        REQUIRE(result.measurements[shot * 2 + 1] == 1);  // post-trap classified bit
        REQUIRE(result.detectors[shot] == 0);             // equal bits: parity 0
        REQUIRE(result.observables[shot] == 1);
    }
}

TEST_CASE("exact: a hand-written multi-target annotation traps on one target") {
    // A single LEVEL_TRANSITION node with two targets materializes one
    // site per target; the trap maps back through site_targets to the
    // right (op, qubit), and the continuation's split keeps the sibling
    // target's instrument live.
    ModelSpec spec;
    spec.leak_from_e = 1.0;  // fires from e only
    auto model = make_model(spec);
    auto circuit = parse("X 0\nLEVEL_TRANSITION[leak] 0 1\nM 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 20, 67);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);      // q0 leaked, classified
        REQUIRE(result.measurements[shot * 2 + 1] == 0);  // q1 in g: never fires
        REQUIRE(result.final_status[shot * 2].kind() == QubitStatusKind::Leaked);
        REQUIRE(result.final_status[shot * 2 + 1].kind() == QubitStatusKind::ComputationalKnown);
    }
}

TEST_CASE("exact: neglect keeps rank flat while the exact default expands") {
    ModelSpec spec;
    spec.leak_from_e = 0.3;  // source-dependent: the damp is non-scalar
    spec.damping = DampingPolicy::Neglect;
    auto model = make_model(spec);
    auto circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nH 0\nM 0");

    // max_rank 0 admits the neglect compile (no expansion) and would
    // reject the exact-damping one (which adds one at the site).
    auto result = sample_noncomputational(circuit, model, 50, 37, /*max_rank=*/0);
    REQUIRE(result.measurements.size() == 50);

    ModelSpec exact_spec = spec;
    exact_spec.damping = DampingPolicy::Exact;
    REQUIRE_THROWS_WITH(
        sample_noncomputational(circuit, make_model(exact_spec), 50, 37, /*max_rank=*/0),
        ContainsSubstring("exceeds max_rank 0"));
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
