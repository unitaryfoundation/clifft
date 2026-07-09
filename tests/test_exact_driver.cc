// End-to-end tests for exact-mode sampling: the driver loop, continuation
// cache, frame-preloaded initials, and trap resolution behind
// sample_noncomputational.
//
// Deterministic pins use certain (p = 1) channels; statistical checks
// use source-independent rates with closed forms and generous margins
// (the full distributional campaign lives in the Python enumerator
// suite).

#include "clifft/circuit/parser.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"
#include "clifft/noncomp/seed.h"
#include "clifft/util/xoshiro.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
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
    double flip_g_to_e = 0.0;
    std::vector<double> initial = {1.0, 0.0, 0.0, 0.0, 0.0};
    DampingPolicy damping = DampingPolicy::Exact;
};

NonComputationalModel make_model(const ModelSpec& spec) {
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][0] = spec.leak_from_g;
    leak[kLeak][1] = spec.leak_from_e;
    leak[1][kLeak] = spec.seep_to_e;
    std::map<std::string, std::vector<std::vector<double>>> transitions{{"leak", leak}};
    if (spec.flip_g_to_e > 0.0) {
        // A purely computational g -> e transition: its fires resolve
        // in-line inside the VM and never trap.
        std::vector<std::vector<double>> flip(5, std::vector<double>(5, 0.0));
        flip[1][0] = spec.flip_g_to_e;
        transitions.emplace("flip", std::move(flip));
    }

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.damping = spec.damping;
    return NonComputationalModel::from_spec(spec.initial, transitions,
                                            std::make_optional(classifier), policy);
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

// Source-independent: g and e both jump to lost with certainty.
// The classifier is faithful on the computational columns (g reads 0,
// e reads 1); the noncomputational columns read the parked-carrier
// convention (leak_g/lost read 0, leak_e reads 1).
NonComputationalModel make_lose_model() {
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;  // g -> lost, certainly
    lose[kLost][1] = 1.0;  // e -> lost, certainly

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    return NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                            std::make_optional(classifier), policy);
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
    for (const QubitStatus s : result.final_status) {
        REQUIRE(s == QubitStatus::Computational);
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
        REQUIRE(is_leaked(result.final_status[shot * 2]));
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Computational);
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
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
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
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
}

TEST_CASE("exact: an in-line computational fire is never reported as a known level") {
    // From g, the flip site fires g -> e inside the VM on ~half the
    // shots; the driver never learns which shots fired, so the sidecar
    // must not claim a known level for either population. The
    // measurement mixture pins that the fire really happens.
    ModelSpec spec;
    spec.flip_g_to_e = 0.5;
    auto model = make_model(spec);
    auto circuit = parse("LEVEL_TRANSITION[flip] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 200, 17);
    int ones = 0;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        ones += result.measurements[shot];
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
    REQUIRE(ones > 60);
    REQUIRE(ones < 140);
}

TEST_CASE("exact: a later trap composes with an earlier in-line fire") {
    // The flip site fires g -> e inside the VM; the leak site then fires
    // only from e (p = 1), so exactly the flipped shots trap. Per shot, a
    // leaked status must coincide with a classified 1 and a computational
    // status with a quantum 0: the trap resolves against the state's true
    // level, not the status walk's pre-fire bookkeeping.
    ModelSpec spec;
    spec.flip_g_to_e = 0.5;
    spec.leak_from_e = 1.0;
    auto model = make_model(spec);
    auto circuit = parse("LEVEL_TRANSITION[flip] 0\nLEVEL_TRANSITION[leak] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 200, 19);
    int leaked = 0;
    for (uint32_t shot = 0; shot < 200; ++shot) {
        if (is_leaked(result.final_status[shot])) {
            ++leaked;
            REQUIRE(result.measurements[shot] == 1);
        } else {
            REQUIRE(result.final_status[shot] == QubitStatus::Computational);
            REQUIRE(result.measurements[shot] == 0);
        }
    }
    REQUIRE(leaked > 60);
    REQUIRE(leaked < 140);
}

TEST_CASE("exact: a source-independent rate matches its closed form") {
    // A source-independent rate (p_g = p_e) has the closed form
    // p(leak) = 0.3 regardless of the carrier state; a leaked qubit
    // reads 1 while a computational |0> reads 0. (The distributional
    // reference for richer scenarios is the Python enumerator suite.)
    const uint32_t shots = 4000;
    auto circuit = parse("LEVEL_TRANSITION[leak] 0\nM 0");

    ModelSpec spec;
    spec.leak_from_g = 0.3;
    spec.leak_from_e = 0.3;
    auto result = sample_noncomputational(circuit, make_model(spec), shots, 17);

    const double mean = mean_of(result.measurements, 1, 0);
    // Binomial std at p = 0.3 over 4000 shots is ~0.007; allow 5 sigma.
    REQUIRE(std::abs(mean - 0.3) < 0.04);
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
        REQUIRE(a.final_status[i] == b.final_status[i]);
    }
}

TEST_CASE("exact: the driver and SVM seed streams are domain-separated") {
    // The host draws (initial levels, trap destinations, classifier consults)
    // and the in-VM Born draws run on independent streams; handing the same
    // per-shot seed to both would correlate them. Guard the documented domain
    // split at its source: for every shot, the same (global, shot) with
    // different domains yields unrelated seeds, and the streams they start
    // diverge on the very first draw.
    for (uint64_t global : {0ULL, 1ULL, 0x9E3779B97F4A7C15ULL}) {
        for (uint64_t shot = 0; shot < 16; ++shot) {
            const uint64_t host = derive_seed(global, shot, kExactDriverDomain);
            const uint64_t svm = derive_seed(global, shot, kExactSvmDomain);
            REQUIRE(host != svm);
            Xoshiro256PlusPlus host_rng(host);
            Xoshiro256PlusPlus svm_rng(svm);
            REQUIRE(host_rng.next_double() != svm_rng.next_double());
        }
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

TEST_CASE("exact: a trap-form fire keeps the fire-side correlation") {
    // The decisive pin for the forced trace-out. Qubit 0 is Bell-entangled
    // with qubit 1 and dormant-random at the site; under neglect the fire
    // traps with the carrier uncollapsed. The channel is certain but
    // source-dependent in its destination (g -> leak_g, e -> leak_e), and
    // the classifier maps those levels to 0 and 1 -- so the classified
    // M 0 *is* the reported source, and the partner's M 1 must equal it
    // on every shot, because the continuation's trace-out is forced to
    // that same source. Independent redraw would mismatch half the time.
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeakG][0] = 1.0;  // from g: leak_g, certainly
    leak[kLeak][1] = 1.0;   // from e: leak_e, certainly

    const std::vector<std::vector<double>> classifier = {
        {1.0, 0.0, 1.0, 0.0, 1.0},   // leak_g reads 0
        {0.0, 1.0, 0.0, 1.0, 0.0}};  // leak_e reads 1

    NonComputationalPolicy policy;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                                  std::make_optional(classifier), policy);

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
        REQUIRE(is_leaked(result.final_status[shot * 2]));
        REQUIRE(is_leaked(result.final_status[shot * 2 + 1]));
    }
}

TEST_CASE("exact: a chain of two forced traps keeps both correlations") {
    // Two independent Bell pairs, each with a certain source-dependent
    // leak under neglect: every shot traps twice, forcing two trace-outs
    // at two different hidden slots, and the second continuation's prefix
    // contains the first's forced instruction (exercising the
    // sampling/forced mask in the debug prefix comparison). Both
    // partner correlations must survive.
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeakG][0] = 1.0;
    leak[kLeak][1] = 1.0;

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                                  std::make_optional(classifier), policy);

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
    std::vector<std::vector<double>> swap_ge(5, std::vector<double>(5, 0.0));
    swap_ge[1][0] = 1.0;  // g -> e, certainly
    swap_ge[0][1] = 1.0;  // e -> g, certainly

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.damping = DampingPolicy::Neglect;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"swap", swap_ge}},
                                                  std::make_optional(classifier), policy);

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
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][1] = 1.0;  // certain leak from e

    const std::vector<std::vector<double>> classifier = {
        {1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 0.5, 0.0}, {0.0, 0.0, 0.0, 0.5, 0.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                                  std::make_optional(classifier), policy);

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
        REQUIRE(is_leaked(result.final_status[shot * 3]));
        REQUIRE(is_leaked(result.final_status[shot * 3 + 1]));
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
        REQUIRE(is_leaked(result.final_status[shot * 2]));
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Computational);
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
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][1] = 1.0;

    const std::vector<std::vector<double>> classifier = {
        {1.0, 0.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({0.0, 1.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                                  std::make_optional(classifier), policy);

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

TEST_CASE("exact: a hand-built malformed LOSS rejects up front") {
    // A live LOSS site rides through the continuation rewrite verbatim,
    // and trace() must never be the first to look at its arguments (a
    // missing one would read as probability zero). Every annotation
    // validates before the first compile instead. The parser guarantees
    // LOSS(p) for parsed circuits; these are programmatically built.
    ModelSpec spec;
    auto model = make_model(spec);
    Circuit circuit = parse("H 0\nM 0");
    SECTION("missing the probability argument") {
        circuit.nodes.insert(circuit.nodes.begin() + 1,
                             AstNode{GateType::LOSS, {Target::qubit(0)}, {}, 0});
        REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 4, 1),
                            ContainsSubstring("exactly one argument"));
    }
    SECTION("probability outside [0, 1]") {
        circuit.nodes.insert(circuit.nodes.begin() + 1,
                             AstNode{GateType::LOSS, {Target::qubit(0)}, {7.0}, 0});
        REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 4, 1),
                            ContainsSubstring("out of [0, 1]"));
    }
}

TEST_CASE("exact: a smaller starting module must not shrink the reused state") {
    // A leaked initial compiles a from-the-top continuation with more
    // hidden record slots (the MR restore) but a smaller peak rank than
    // the main line, whose expand_damp site needs the array. A shot
    // sequence interleaving both starting modules used to rebuild the
    // state to the smaller module's rank while the tracker kept the
    // maximum; the next main-line shot then overran its allocation --
    // caught by the Debug kernel assert, an out-of-bounds write in
    // Release.
    ModelSpec spec;
    spec.leak_from_e = 3e-3;  // source-dependent: the dormant site damp-expands
    spec.leak_from_g = 3e-4;
    spec.initial = {0.5, 0.0, 0.5, 0.0, 0.0};  // both starting modules occur
    auto model = make_model(spec);
    Circuit circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nMR 0\nM 0");
    auto result = sample_noncomputational(circuit, model, 64, 5);
    REQUIRE(result.shots == 64);
}

TEST_CASE("exact: a zero-fire LOSS(0) before a firing LOSS does not shift site ids") {
    // LOSS(0) can never fire from a computational qubit; trace() skips it.
    // The rewriter must skip it too, so LOSS(1) gets site id 0 and the
    // driver maps the trap correctly.
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 1.0, 1.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LOSS(0) 0\nLOSS(1) 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 71);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);  // lost: classifier's lost column reads 1
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: a seepage-only transition before a firing site does not shift site ids") {
    // A LEVEL_TRANSITION whose computational columns are both zero (seepage
    // from leak_e to e only) cannot fire on a computational qubit; trace()
    // skips it. The rewriter must skip it too so the firing LOSS(1) that
    // follows gets site id 0 and the driver maps the trap correctly.
    std::vector<std::vector<double>> seep(5, std::vector<double>(5, 0.0));
    seep[1][kLeak] = 1.0;  // leak_e -> e, prob 1; computational columns zero

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 1.0, 1.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"seep", seep}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LEVEL_TRANSITION[seep] 0\nLOSS(1) 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 73);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);  // lost: classifier's lost column reads 1
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: a seepage-only transition on q0 does not corrupt q1's site id") {
    // Cross-qubit arrangement: the seepage-only LEVEL_TRANSITION on q0 must
    // not occupy a site slot, so the firing LOSS(1) on q1 keeps site id 0.
    // In Release, a stale site id would read the wrong trap record and
    // silently produce wrong results; in Debug, the site-table lookup
    // overruns or mismatches.
    std::vector<std::vector<double>> seep(5, std::vector<double>(5, 0.0));
    seep[1][kLeak] = 1.0;  // leak_e -> e, prob 1; computational columns zero

    // q0: stays computational (no loss); q1: lost reads 1.
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 1.0, 1.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"seep", seep}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LEVEL_TRANSITION[seep] 0\nLOSS(1) 1\nM 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 20, 79);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 0);      // q0: computational, M reads 0
        REQUIRE(result.measurements[shot * 2 + 1] == 1);  // q1: lost, classifier reads 1
        REQUIRE(result.final_status[shot * 2] == QubitStatus::Computational);
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: a seepage-only transition still seeps a noncomputational qubit") {
    // A zero-fire annotation is skipped for a computational pre-status, but
    // must still execute its classical consult for a noncomputational qubit.
    // Here leak_e is the starting level; the seep fires (classical consult
    // with destination e) and the qubit recaptures to |1>, so M reads 1.
    std::vector<std::vector<double>> seep(5, std::vector<double>(5, 0.0));
    seep[1][kLeak] = 1.0;  // leak_e -> e, prob 1; computational columns zero

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    // All initial mass on leak_e (index 3).
    auto model = NonComputationalModel::from_spec({0.0, 0.0, 0.0, 1.0, 0.0}, {{"seep", seep}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LEVEL_TRANSITION[seep] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 83);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);  // recaptured to |e>=|1>, M reads 1
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
}

// =========================================================================
// Correlated-chain passthrough on noncomputational operands
// =========================================================================

TEST_CASE("exact: a correlated-chain head with a lost operand does not orphan the ELSE") {
    // q0 is lost before the E node: the head must keep its slot in the
    // else-conditioning rather than being dropped. E(1) fires with
    // certainty, so the ELSE never fires. q0 final status is Lost; its
    // record reads 0 (identity classifier). q1 record reads 1 (X1 from the
    // fired head).
    auto model = make_lose_model();
    auto circuit = parse(
        "LEVEL_TRANSITION[lose] 0\n"
        "E(1) X0 X1\n"
        "ELSE_CORRELATED_ERROR(1) X1\n"
        "M 0 1\n");

    auto result = sample_noncomputational(circuit, model, 25, 71);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 0);      // lost reads 0
        REQUIRE(result.measurements[shot * 2 + 1] == 1);  // X1 from fired head
        REQUIRE(result.final_status[shot * 2] == QubitStatus::Lost);
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Computational);
    }
}

TEST_CASE("exact: a fired head with a lost operand prevents the ELSE from firing") {
    // Conditioning pin: E(1) fires (head always fires, operating on the
    // vacated q0 carrier), so the ELSE must NOT fire -- if the head were
    // dropped the ELSE would become the new head and fire, flipping q1.
    // q1 record must read 0 every shot.
    auto model = make_lose_model();
    auto circuit = parse(
        "LEVEL_TRANSITION[lose] 0\n"
        "E(1) X0\n"
        "ELSE_CORRELATED_ERROR(1) X1\n"
        "M 0 1\n");

    auto result = sample_noncomputational(circuit, model, 25, 73);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2 + 1] == 0);  // ELSE did not fire
    }
}

TEST_CASE("exact: a mixed-operand chain member keeps the healthy qubit's Pauli") {
    // E(1) X0 X1 with only q1 lost: q0 is computational and its X must
    // land; dropping the mixed-operand node whole would suppress the X on q0.
    // q0 record reads 1 (X flipped it); q1 record reads 0 (identity classifier).
    auto model = make_lose_model();
    auto circuit = parse(
        "LEVEL_TRANSITION[lose] 1\n"
        "E(1) X0 X1\n"
        "M 0 1\n");

    auto result = sample_noncomputational(circuit, model, 25, 79);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);      // X landed on q0
        REQUIRE(result.measurements[shot * 2 + 1] == 0);  // lost reads 0
    }
}

TEST_CASE("exact: E(1) X0 X1 with no loss applies X to both qubits") {
    // Baseline: all computational, E(1) fires with certainty applying X0 X1.
    // Both qubits start at |0> so both measure 1 after X.
    ModelSpec spec;  // all computational, no loss
    auto model = make_model(spec);
    auto circuit = parse("E(1) X0 X1\nM 0\nM 1\n");
    auto result = sample_noncomputational(circuit, model, 5, 99);
    for (uint32_t shot = 0; shot < 5; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 1);      // X0 applied
        REQUIRE(result.measurements[shot * 2 + 1] == 1);  // X1 applied
    }
}

// =========================================================================
// MX / MY classify on vacated carriers
// =========================================================================

namespace {

// Classifier whose lost column is col (binary), g=0, e=1 faithful,
// leak columns deterministic symbol 0.
NonComputationalModel make_classify_lost_model(std::vector<double> lost_col,
                                               bool with_hook = false) {
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("lose", lose);

    // g=0, e=1, leak_g=0, leak_e=0, lost=lost_col
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 1.0, lost_col[0]},
                                                         {0.0, 1.0, 0.0, 0.0, lost_col[1]}};
    (void)with_hook;
    NonComputationalPolicy policy;
    return NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, transitions,
                                            std::make_optional(classifier), policy);
}

}  // namespace

TEST_CASE("exact: MX on a certainly-lost qubit reads the classifier bit") {
    // lose column [0, 1]: the lost qubit reads 1 every shot; final status Lost.
    auto model = make_classify_lost_model({0.0, 1.0});
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nMX 0");
    auto result = sample_noncomputational(circuit, model, 20, 101);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: MY on a certainly-lost qubit reads the classifier bit") {
    // Identical semantics to MX: the readout basis is incidental on a vacated carrier.
    auto model = make_classify_lost_model({0.0, 1.0});
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nMY 0");
    auto result = sample_noncomputational(circuit, model, 20, 103);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 1);
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: inverted MX !0 on a lost qubit complements the classifier bit") {
    // lose column [0, 1]: classifier says 1; inversion flips to 0 every shot.
    auto model = make_classify_lost_model({0.0, 1.0});
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nMX !0");
    auto result = sample_noncomputational(circuit, model, 20, 105);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 0);  // inverted: 1 -> 0
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: ternary herald column on MX sets sidecar flag and patches record") {
    // Three-symbol classifier for the lost level: {0, 0, 1} always heralds.
    // The herald flag must be 1 on every shot; the record carries an
    // unbiased bit (matches the existing M-herald test in make_lose_model).
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;

    // lost column = {0, 0, 1}: always heralds. g/e/leak columns symbol 0.
    const std::vector<std::vector<double>> classifier = {
        {1.0, 0.0, 1.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 1.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nMX 0");
    auto result = sample_noncomputational(circuit, model, 40, 107);
    for (uint32_t shot = 0; shot < 40; ++shot) {
        REQUIRE(result.heralds[shot] == 1);
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: computational X-basis behavior is untouched by MX classify change") {
    // RX prepares |+>; MX measures in the X basis and records 0 deterministically.
    // No noncomputational model capability: the classifier path must never fire.
    ModelSpec spec;  // all computational
    auto model = make_model(spec);
    auto circuit = parse("RX 0\nMX 0");
    auto result = sample_noncomputational(circuit, model, 20, 109);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 0);
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
}

TEST_CASE("exact: memory-X smoke: two qubits, low-rate leak hook, MX measures both") {
    // The motivating use case: a stim-style memory-X circuit that ends in MX
    // on data qubits runs cleanly under a leakage model.
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLost][0] = 0.01;  // low-rate loss from g
    leak[kLost][1] = 0.01;

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 1.0, 1.0},
                                                         {0.0, 1.0, 0.0, 0.0, 0.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"leak", leak}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("RX 0 1\nLEVEL_TRANSITION[leak] 0 1\nMX 0 1");
    auto result = sample_noncomputational(circuit, model, 50, 111);
    REQUIRE(result.shots == 50);
    REQUIRE(result.num_measurements == 2);
    // Shape check: result exists with the right dimensions.
    REQUIRE(result.measurements.size() == 100u);
    REQUIRE(result.final_status.size() == 100u);
}

// =========================================================================
// Up-front capability contract (gate A and gate B)
// =========================================================================

TEST_CASE("exact: gate A: capable model + MPP rejects before sampling") {
    // A capable model (non-zero loss) plus an MPP measurement must throw
    // before any shots are drawn, with a message naming 'MPP', 'not supported',
    // and 'ancilla'.
    auto model = make_classify_lost_model({0.5, 0.5});
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nMPP X0*X1");
    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 4, 1),
                        ContainsSubstring("MPP") && ContainsSubstring("not supported") &&
                            ContainsSubstring("ancilla"));
}

TEST_CASE("exact: gate A: non-capable model + MPP samples fine") {
    // A model with no capability (initial all-g, no leak/loss transitions)
    // does not trigger gate A, so MPP is accepted.
    ModelSpec spec;  // all computational
    auto model = make_model(spec);
    auto circuit = parse("H 0\nCX 0 1\nMPP Z0*Z1");
    auto result = sample_noncomputational(circuit, model, 10, 1);
    REQUIRE(result.shots == 10);
    // ZZ stabilizer on Bell state: all 0.
    for (uint32_t shot = 0; shot < 10; ++shot) {
        REQUIRE(result.measurements[shot] == 0);
    }
}

TEST_CASE("exact: gate B: capable model + measurement + no classifier throws") {
    // A model with a LEVEL_TRANSITION annotation that can fire into a
    // noncomputational level, paired with a measurement, must throw when
    // no classifier is present.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;
    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                                  std::nullopt, policy);
    // The annotation makes the circuit capable.
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nM 0");
    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 4, 1),
                        ContainsSubstring("classifier is required"));
}

TEST_CASE("exact: gate B: capable model + measurement-free circuit + no classifier samples") {
    // A capable model without a classifier is fine if the circuit has no measurements.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;
    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                                  std::nullopt, policy);
    // Circuit has no measurement nodes.
    auto circuit = parse("H 0\nLEVEL_TRANSITION[lose] 0");
    auto result = sample_noncomputational(circuit, model, 10, 1);
    REQUIRE(result.shots == 10);
}

TEST_CASE("exact: gate B: non-capable model + MX + no classifier samples") {
    // A non-capable model (no loss/leak transitions, all-g initial) plus
    // MX requires no classifier: gate B does not fire.
    ModelSpec spec;  // all computational, no loss
    auto model = make_model(spec);
    auto circuit = parse("RX 0\nMX 0");
    auto result = sample_noncomputational(circuit, model, 10, 1);
    REQUIRE(result.shots == 10);
    for (uint32_t shot = 0; shot < 10; ++shot) {
        REQUIRE(result.measurements[shot] == 0);
    }
}

TEST_CASE(
    "exact: bluntness pin -- model leaks only q0 via annotation, circuit measures only q1, "
    "no classifier throws") {
    // The contract is a capability boundary, not per-qubit reachability.
    // The annotation on q0 makes the model capable; q1 is measured but
    // never touches a vacated carrier in any reachable shot. Gate B still
    // fires because the capability boundary is coarse: capable model +
    // any measurement = classifier required.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;  // q0 loses certainly from g
    lose[kLost][1] = 1.0;
    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                                  std::nullopt, policy);
    auto circuit = parse("LEVEL_TRANSITION[lose] 0\nM 1");
    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 4, 1),
                        ContainsSubstring("classifier is required"));
}

TEST_CASE("exact: the model contract is validated even for zero shots") {
    // Validation is shot-count independent: a zero-shot call still checks
    // the circuit/model contract, and a valid pair returns empty results.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;
    auto no_classifier = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"S", lose}},
                                                          std::nullopt, NonComputationalPolicy{});
    REQUIRE_THROWS_WITH(sample_noncomputational(parse("S 0\nM 0"), no_classifier, 0, 1),
                        ContainsSubstring("classifier is required"));

    auto valid = make_lose_model();
    auto result = sample_noncomputational(parse("S 0\nM 0"), valid, 0, 1);
    REQUIRE(result.shots == 0);
    REQUIRE(result.measurements.empty());
}

// =========================================================================
// Same-qubit re-trap (composed-continuation path)
// =========================================================================

namespace {

// Certain e->leak_e leak and certain leak_e->e seep; g initial.
// Circuit must prep |e> (e.g. X 0) before the first annotation fires.
NonComputationalModel make_leak_seep_model() {
    std::vector<std::vector<double>> leak(5, std::vector<double>(5, 0.0));
    leak[kLeak][1] = 1.0;  // e -> leak_e, certainly

    std::vector<std::vector<double>> seep(5, std::vector<double>(5, 0.0));
    seep[1][kLeak] = 1.0;  // leak_e -> e, certainly

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    return NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0},
                                            {{"leak", leak}, {"seep", seep}},
                                            std::make_optional(classifier), policy);
}

}  // namespace

TEST_CASE("exact: same-qubit re-trap: leak then recapture then leak again") {
    // X 0 preps |e>. LEVEL_TRANSITION[leak] traps (e->leak_e, certainly).
    // LEVEL_TRANSITION[seep] recaptures back to e (classical consult).
    // The second LEVEL_TRANSITION[leak] traps again (e->leak_e).
    // The composed-continuation path exercises trap->recapture-consult->trap on one qubit.
    // Expect: every shot reads 1 (leak_e column) and final status LeakE.
    auto model = make_leak_seep_model();
    auto circuit = parse(
        "X 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[seep] 0\n"
        "LEVEL_TRANSITION[leak] 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 30, 131);
    for (uint32_t shot = 0; shot < 30; ++shot) {
        REQUIRE(result.measurements[shot] == 1);  // leak_e column reads 1
        REQUIRE(result.final_status[shot] == QubitStatus::LeakE);
    }
}

TEST_CASE("exact: multi-target annotation jumps both targets in one shot") {
    // LEVEL_TRANSITION[lose] on both q0 and q1 in a single annotation; the
    // lose channel is source-independent (g->lost p=1, e->lost p=1), so both
    // qubits lose in every shot. Both records read 0 (identity classifier's
    // lost column), both statuses are Lost.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;  // g -> lost, certainly
    lose[kLost][1] = 1.0;  // e -> lost, certainly

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"lose", lose}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("LEVEL_TRANSITION[lose] 0 1\nM 0\nM 1");

    auto result = sample_noncomputational(circuit, model, 20, 133);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot * 2] == 0);      // lost column reads 0
        REQUIRE(result.measurements[shot * 2 + 1] == 0);  // lost column reads 0
        REQUIRE(result.final_status[shot * 2] == QubitStatus::Lost);
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Lost);
    }
}

// =========================================================================
// LOSS on already-leaked and already-lost qubits
// =========================================================================

TEST_CASE("exact: LOSS(1) on a pure leak_g initial vacates the carrier") {
    // Initial mass entirely on leak_g (index 2): the qubit is already
    // noncomputational. LOSS(1) on a leaked qubit still vacates it (the
    // channel fires from any non-lost level). Every shot: status Lost,
    // record 0 (identity classifier's lost column reads 0).
    ModelSpec spec;
    spec.initial = {0.0, 0.0, 1.0, 0.0, 0.0};  // pure leak_g
    auto model = make_model(spec);
    auto circuit = parse("LOSS(1) 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 141);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 0);  // identity classifier, lost column reads 0
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: LOSS(1) on a pure lost initial is a no-op") {
    // Initial mass entirely on lost (index 4): LOSS is a no-op on an
    // already-lost qubit. Status Lost, record 0 on every shot.
    ModelSpec spec;
    spec.initial = {0.0, 0.0, 0.0, 0.0, 1.0};  // pure lost
    auto model = make_model(spec);
    auto circuit = parse("LOSS(1) 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 20, 143);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        REQUIRE(result.measurements[shot] == 0);  // identity classifier, lost column reads 0
        REQUIRE(result.final_status[shot] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: LOSS on a lost qubit spends no draw -- a later consult sees the same stream") {
    // A LOSS(1) on an already-lost qubit records its no-op without
    // consuming any driver randomness. The discriminator is a stochastic
    // consult AFTER it: the recover channel returns the lost qubit to e
    // with probability one half, drawing from the driver stream. If the
    // preceding LOSS spent a draw, every later recover outcome would
    // shift; instead the two runs must agree shot by shot on records and
    // statuses alike.
    const uint64_t seed = 147;
    std::vector<std::vector<double>> recover(5, std::vector<double>(5, 0.0));
    recover[1][kLost] = 0.5;  // lost -> e, probability one half
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};
    auto model =
        NonComputationalModel::from_spec({0.0, 0.0, 0.0, 0.0, 1.0}, {{"recover", recover}},
                                         std::make_optional(classifier), NonComputationalPolicy{});

    auto r_plain =
        sample_noncomputational(parse("LEVEL_TRANSITION[recover] 0\nM 0"), model, 128, seed);
    auto r_loss = sample_noncomputational(parse("LOSS(1) 0\nLEVEL_TRANSITION[recover] 0\nM 0"),
                                          model, 128, seed);
    REQUIRE(r_plain.measurements == r_loss.measurements);
    REQUIRE(r_plain.final_status == r_loss.final_status);

    // Vacuity guard: the recover consult really is stochastic -- both
    // outcomes occur across the shots.
    bool saw_zero = false;
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 128; ++shot) {
        (r_plain.measurements[shot] == 0 ? saw_zero : saw_one) = true;
    }
    REQUIRE(saw_zero);
    REQUIRE(saw_one);
}

TEST_CASE("exact: a non-restoring lost MRX spends no hidden draw -- M and MRX read identically") {
    // Both circuits lose qubit 0 with certainty and classify its record
    // slot from the same lost column, so they compile to the same module
    // -- unless MRX's X-basis reset half were emitted on the vacated
    // carrier, whose hidden MX would spend an SVM draw and shift every
    // later outcome on qubit 1's stream.
    const uint64_t seed = 61;
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 1.0, 1.0}};
    auto model = NonComputationalModel::from_spec(
        {1.0, 0.0, 0.0, 0.0, 0.0}, {}, std::make_optional(classifier), NonComputationalPolicy{});

    auto r_m = sample_noncomputational(parse("LOSS(1) 0\nM 0\nH 1\nM 1"), model, 128, seed);
    auto r_mrx = sample_noncomputational(parse("LOSS(1) 0\nMRX 0\nH 1\nM 1"), model, 128, seed);
    REQUIRE(r_m.measurements == r_mrx.measurements);
    REQUIRE(r_m.final_status == r_mrx.final_status);

    bool saw_zero = false;
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 128; ++shot) {
        // Slot 0 is the classifier's lost column: 1 with certainty.
        REQUIRE(r_m.measurements[shot * 2] == 1);
        // Vacuity guard: qubit 1's H-then-measure really draws -- both
        // outcomes occur across the shots.
        (r_m.measurements[shot * 2 + 1] == 0 ? saw_zero : saw_one) = true;
    }
    REQUIRE(saw_zero);
    REQUIRE(saw_one);
}

// =========================================================================
// Classified records driving feedback
// =========================================================================

TEST_CASE("exact: classified record drives a CX feedback gate") {
    // "lose" is hooked on S: g and e both jump to lost with certainty.
    // Classifier: classified symbol 0 always reads 0, classified symbol 1
    // always reads 1; the lost column has weight on symbol 1, so rec[-1]
    // after the S-annotated M 0 is 1 on every shot.
    // CX rec[-1] 1 flips q1 conditioned on that classified record.
    // Final M 1 must read 1 on every shot; q1 stays Computational.
    //
    // Note: a feedback-onto-leaked observability test would be a vacuous
    // pass -- a dropped virtual correction on a vacated carrier is
    // unobservable by design (the carrier is gone).
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;

    // g=0, e=1, leak_g=0, leak_e=1, lost=1
    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 1.0, 1.0}};

    NonComputationalPolicy policy;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"S", lose}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("S 0\nM 0\nCX rec[-1] 1\nM 1");

    auto result = sample_noncomputational(circuit, model, 25, 151);
    for (uint32_t shot = 0; shot < 25; ++shot) {
        REQUIRE(result.measurements[shot * 2 + 1] == 1);  // classified record drove the CX
        REQUIRE(result.final_status[shot * 2 + 1] == QubitStatus::Computational);
    }
}

// =========================================================================
// Carrier re-preparation on restore
// =========================================================================

TEST_CASE("exact: a reset truly re-prepares a restored-lost qubit") {
    // "lose" is hooked on S (g and e both -> lost with certainty).
    // reset_restores_lost=true means R 0 reloads the lost carrier.
    // H 0 before S scrambles the pre-loss state: if R did NOT re-prepare
    // the carrier the record would be random (the lost carrier's parked
    // |+> state would read 0 or 1 at random). A true re-prepare resets
    // to |0>, so M 0 must read 0 and the final status must be Computational
    // on every shot.
    std::vector<std::vector<double>> lose(5, std::vector<double>(5, 0.0));
    lose[kLost][0] = 1.0;
    lose[kLost][1] = 1.0;

    const std::vector<std::vector<double>> classifier = {{1.0, 0.0, 1.0, 0.0, 1.0},
                                                         {0.0, 1.0, 0.0, 1.0, 0.0}};

    NonComputationalPolicy policy;
    policy.reset_restores_lost = true;
    auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"S", lose}},
                                                  std::make_optional(classifier), policy);
    auto circuit = parse("H 0\nS 0\nR 0\nM 0");

    auto result = sample_noncomputational(circuit, model, 30, 161);
    for (uint32_t shot = 0; shot < 30; ++shot) {
        REQUIRE(result.measurements[shot] == 0);  // re-prepared to |0>
        REQUIRE(result.final_status[shot] == QubitStatus::Computational);
    }
}

// =========================================================================
// max_rank boundary behavior
// =========================================================================

TEST_CASE("exact: max_rank succeeds exactly at the required rank") {
    // Reuses the over-cap circuit from the rejection test: three H-prefixed
    // source-dependent sites push peak rank to 3 (over a cap of 2). The
    // same circuit must succeed with max_rank = 3 (the actual required rank).
    ModelSpec spec;
    spec.leak_from_e = 0.2;
    auto model = make_model(spec);
    auto circuit = parse(
        "H 0\nH 1\nH 2\n"
        "LEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 1\nLEVEL_TRANSITION[leak] 2\n"
        "M 0\nM 1\nM 2");

    // max_rank = 3 is exactly the peak; must succeed.
    auto result = sample_noncomputational(circuit, model, 5, 1, /*max_rank=*/3);
    REQUIRE(result.shots == 5);
    REQUIRE(result.num_measurements == 3);
}

// =========================================================================
// Seed sensitivity
// =========================================================================

TEST_CASE("exact: different seeds produce different records") {
    // A stochastic config (leak p=0.35 from e after H) sampled at two
    // different seeds must differ: a constant-output implementation that
    // ignores the seed would collide here with probability 2^(-measurements).
    // This is the C++ companion to the Python same-seed/different-seed pins.
    ModelSpec spec;
    spec.leak_from_e = 0.35;
    spec.initial = {0.5, 0.5, 0.0, 0.0, 0.0};
    auto model = make_model(spec);
    auto circuit = parse("H 0\nLEVEL_TRANSITION[leak] 0\nM 0");

    auto a = sample_noncomputational(circuit, model, 64, 97);
    auto b = sample_noncomputational(circuit, model, 64, 98);
    // 64 coin-flip measurements collide with probability 2^-64; this is
    // astronomically unlikely for any two distinct seeds.
    REQUIRE(a.measurements != b.measurements);
}

TEST_CASE("exact: max_rank is not checked against the unreachable all-computational module") {
    // When every qubit starts lost, the no-event module is never used;
    // its rank must not trigger a max_rank rejection.
    // The lost column reads symbol 1 with certainty: a raw readout of the
    // dropped-everything |0> carriers would give 0, so all-1 records pin
    // that the classifier wrote them.
    const std::vector<std::vector<double>> classifier_matrix = {{1.0, 0.0, 1.0, 0.0, 0.0},
                                                                {0.0, 1.0, 0.0, 1.0, 1.0}};
    // Lost model: all qubits start lost.
    const NonComputationalModel lost_model = NonComputationalModel::from_spec(
        {0.0, 0.0, 0.0, 0.0, 1.0}, {}, std::make_optional(classifier_matrix),
        NonComputationalPolicy{});
    // Ground model: all qubits start in g -- the no-event module IS compiled.
    const NonComputationalModel ground_model = NonComputationalModel::from_spec(
        {1.0, 0.0, 0.0, 0.0, 0.0}, {}, std::make_optional(classifier_matrix),
        NonComputationalPolicy{});

    // Circuit with T gates -- non-trivial rank.
    const std::string circuit_str =
        "H 0\n"
        "CX 0 1\nCX 0 2\nCX 0 3\nCX 0 4\nCX 0 5\n"
        "T 0\nH 0\nT 1\nH 1\nT 2\nH 2\nT 3\nH 3\nT 4\nH 4\nT 5\nH 5\n"
        "M 0\nM 1\nM 2\nM 3\nM 4\nM 5\n";
    const Circuit circuit = parse(circuit_str);

    // Ground model at max_rank=0 must throw (the no-event module exceeds it).
    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, ground_model, 4, 1, 0),
                        ContainsSubstring("max_rank"));

    // Lost model at max_rank=0 must run (the no-event module is lazy).
    const NonComputationalSample r = sample_noncomputational(circuit, lost_model, 16, 1, 0);
    for (const uint8_t bit : r.measurements) {
        REQUIRE(bit == 1);
    }
    for (size_t i = 0; i < r.final_status.size(); ++i) {
        REQUIRE(r.final_status[i] == QubitStatus::Lost);
    }
}

// =========================================================================
// Classifier and herald end-to-end
// =========================================================================

namespace {

// Helpers for the classifier/herald/confusion test group below.
// Make an all-zero 5x5 matrix.
std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-independent: g and e both jump to leak_g with certainty.
std::vector<std::vector<double>> certain_leak() {
    auto m = zeros5();
    m[kLeakG][0] = 1.0;
    m[kLeakG][1] = 1.0;
    return m;
}

// Source-independent: g and e both jump to lost with certainty.
std::vector<std::vector<double>> certain_loss() {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return m;
}

// Source-independent: g and e both jump to the computational g level.
std::vector<std::vector<double>> certain_jump_to_g() {
    auto m = zeros5();
    m[0][0] = 1.0;
    m[0][1] = 1.0;
    return m;
}

// Source-independent: g and e both jump to the computational e level.
std::vector<std::vector<double>> certain_jump_to_e() {
    auto m = zeros5();
    m[1][0] = 1.0;
    m[1][1] = 1.0;
    return m;
}

// Two-symbol classifier whose column for `level` is `col`; computational
// levels read out faithfully (no readout confusion) and other
// noncomputational levels read a deterministic symbol 0.
std::vector<std::vector<double>> classifier_with(uint8_t level, std::vector<double> col) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;  // symbol "0"
    }
    // Computational levels read out faithfully (identity columns) so the
    // classifier adds no readout confusion here.
    m[0][1] = 0.0;
    m[1][1] = 1.0;
    m[0][level] = col[0];
    m[1][level] = col[1];
    return m;
}

// The common case: classify the leak_g column.
std::vector<std::vector<double>> leakg_classifier(std::vector<double> leakg) {
    return classifier_with(kLeakG, std::move(leakg));
}

// Build a NonComputationalModel directly from raw vectors.
NonComputationalModel make_direct_model(
    std::vector<double> initial_state,
    std::map<std::string, std::vector<std::vector<double>>> transitions,
    std::optional<std::vector<std::vector<double>>> classifier = std::nullopt,
    NonComputationalPolicy policy = {}) {
    return NonComputationalModel::from_spec(std::move(initial_state), transitions,
                                            std::move(classifier), policy);
}

std::vector<double> all_g() {
    return {1.0, 0.0, 0.0, 0.0, 0.0};
}
std::vector<double> all_e() {
    return {0.0, 1.0, 0.0, 0.0, 0.0};
}

// Three-symbol classifier whose column for `level` is `col`; computational
// levels read out faithfully and other levels default to symbol 0.
std::vector<std::vector<double>> ternary_classifier_with(uint8_t level, std::vector<double> col) {
    std::vector<std::vector<double>> m(3, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][1] = 0.0;
    m[1][1] = 1.0;
    m[0][level] = col[0];
    m[1][level] = col[1];
    m[2][level] = col[2];
    return m;
}

// Classifier with confused computational columns; noncomputational levels
// deterministically read symbol 0.
std::vector<std::vector<double>> comp_confusion(double p01, double p10) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][0] = 1.0 - p01;
    m[1][0] = p01;
    m[0][1] = p10;
    m[1][1] = 1.0 - p10;
    return m;
}

}  // namespace

TEST_CASE("exact: a partial classifier bit matches its frequency") {
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    NonComputationalModel model =
        make_direct_model(all_g(), std::move(transitions), leakg_classifier({0.5, 0.5}));

    NonComputationalSample s = sample_noncomputational(c, model, 2000, 3);
    size_t ones = 0;
    for (uint8_t d : s.detectors) {
        ones += d;
    }
    REQUIRE(ones > 850);  // expected 1000; generous band
    REQUIRE(ones < 1150);
}

TEST_CASE("exact: the herald symbol fills the sidecar, not the record") {
    // Qubit 1 leaks and its column always heralds; qubit 0 stays
    // computational. The heralded slot's visible record bit is a uniform
    // draw, so the record layout is unchanged and both values occur.
    Circuit c = parse("H 1\nS 1\nM 1\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    auto cl = ternary_classifier_with(kLeakG, {0.0, 0.0, 1.0});
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions), std::move(cl));

    constexpr uint32_t kShots = 256;
    NonComputationalSample r = sample_noncomputational(c, model, kShots, 5);
    REQUIRE(r.heralds.size() == kShots * 2);
    size_t ones = 0;
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        REQUIRE(r.heralds[shot * 2 + 0] == 1);  // leaked slot heralds
        REQUIRE(r.heralds[shot * 2 + 1] == 0);  // computational slot does not
        ones += r.measurements[shot * 2 + 0];
    }
    REQUIRE(ones > kShots * 30 / 100);  // heralded record bit is uniform
    REQUIRE(ones < kShots * 70 / 100);
}

TEST_CASE("exact: a partial herald column matches its frequency") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    auto cl = ternary_classifier_with(kLeakG, {0.3, 0.0, 0.7});
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions), std::move(cl));

    constexpr uint32_t kShots = 1000;
    NonComputationalSample r = sample_noncomputational(c, model, kShots, 6);
    size_t heralded = 0;
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        heralded += r.heralds[shot];
    }
    REQUIRE(heralded > 630);
    REQUIRE(heralded < 770);
}

TEST_CASE("exact: the record bit is uniform given a herald, pinned without") {
    // Column {0.5, 0, 0.5}: a non-heralded draw is always symbol 0, while a
    // heralded slot's bit is uniform. This pins the (herald, bit) joint: if
    // heralded slots kept the not-heralded flip probability (0), their bits
    // would all read 0.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    auto cl = ternary_classifier_with(kLeakG, {0.5, 0.0, 0.5});
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions), std::move(cl));

    constexpr uint32_t kShots = 2000;
    NonComputationalSample r = sample_noncomputational(c, model, kShots, 13);
    size_t heralded = 0;
    size_t heralded_ones = 0;
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        if (r.heralds[shot]) {
            ++heralded;
            heralded_ones += r.measurements[shot];
        } else {
            REQUIRE(r.measurements[shot] == 0);  // not-heralded bit is symbol 0
        }
    }
    REQUIRE(heralded > 850);  // expected 1000; generous band
    REQUIRE(heralded < 1150);
    // Heralded bits are uniform: expected heralded/2, ~6 sigma band.
    REQUIRE(heralded_ones > heralded * 35 / 100);
    REQUIRE(heralded_ones < heralded * 65 / 100);
}

TEST_CASE("exact: a two-symbol classifier leaves the herald sidecar zero") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    auto cl = leakg_classifier({0.5, 0.5});
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions), std::move(cl));

    constexpr uint32_t kShots = 64;
    NonComputationalSample r = sample_noncomputational(c, model, kShots, 7);
    REQUIRE(r.heralds.size() == kShots);
    for (uint8_t h : r.heralds) {
        REQUIRE(h == 0);
    }
}

TEST_CASE("exact: a circuit with EXP_VAL probes rejects") {
    Circuit c;
    c.num_qubits = 1;
    c.num_exp_vals = 1;  // EXP_VAL output is not carried by the noncomp sidecar
    NonComputationalModel model = make_direct_model(all_g(), {});
    REQUIRE_THROWS_WITH(sample_noncomputational(c, model, 4, 1), ContainsSubstring("EXP_VAL"));
}

TEST_CASE("exact: a measure-and-reset on a leaked qubit injects and restores") {
    // MR on the leaked qubit records the classifier bit AND resets the site to
    // |0>; the following M then deterministically reads 0 -- proving the reset
    // actually ran in the SVM, not just in the trajectory bookkeeping.
    Circuit c = parse("H 0\nS 0\nMR 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());
    NonComputationalModel model =
        make_direct_model(all_g(), std::move(transitions), leakg_classifier({0.0, 1.0}));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 5);
    REQUIRE(s.num_measurements == 2);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 1);  // MR slot: forced classifier bit
        REQUIRE(s.measurements[shot * 2 + 1] == 0);  // M slot: restored |0>
    }
}

TEST_CASE("exact: a lost-qubit measurement feeds the detector the classifier bit") {
    // A lost qubit (vacated site) is still measured; the classifier supplies
    // the record bit just as for a leaked qubit, so the detector reads it.
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_loss());

    // Force the lost column -> symbol 1 -> record bit 1.
    NonComputationalModel model =
        make_direct_model(all_g(), std::move(transitions), classifier_with(kLost, {0.0, 1.0}));
    NonComputationalSample s = sample_noncomputational(c, model, 200, 9);
    REQUIRE(s.num_detectors == 1);
    REQUIRE(s.detectors.size() == 200);
    for (uint8_t d : s.detectors) {
        REQUIRE(d == 1);  // detector saw the lost-column classifier bit
    }
}

TEST_CASE("exact: a leaked measurement feeds the observable the classifier bit") {
    Circuit c = parse("H 0\nS 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_leak());

    // Force leak_g -> symbol 1 -> record bit 1; the observable must see it,
    // not the residual |0>.
    NonComputationalModel model =
        make_direct_model(all_g(), std::move(transitions), leakg_classifier({0.0, 1.0}));
    NonComputationalSample s = sample_noncomputational(c, model, 200, 11);
    REQUIRE(s.num_observables == 1);
    REQUIRE(s.observables.size() == 200);
    for (uint8_t o : s.observables) {
        REQUIRE(o == 1);
    }
}

TEST_CASE("exact: a jump to the ground level forces the measurement to 0") {
    // The S transition collapses the H-prepared |+> to the g level; without
    // the materializing collapse the M would read 1 on ~half the shots. The
    // fire resolves entirely inside the VM; the sidecar reports the bare
    // computational kind, never a level claim.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_jump_to_g());
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions));
    NonComputationalSample s = sample_noncomputational(c, model, 200, 1);
    for (uint8_t bit : s.measurements) {
        REQUIRE(bit == 0);
    }
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.final_status[shot] == QubitStatus::Computational);
    }
}

TEST_CASE("exact: a jump to the excited level forces the measurement to 1") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_jump_to_e());
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 200, 1);
    for (uint8_t bit : s.measurements) {
        REQUIRE(bit == 1);
    }
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.final_status[shot] == QubitStatus::Computational);
    }
}

TEST_CASE("exact: a partial jump to ground matches the analytic mixture") {
    // With probability p the S transition collapses the H-prepared |+> to g;
    // otherwise the carrier stays coherent. P(M = 1) = (1 - p) / 2 = 0.35.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    auto m = zeros5();
    m[0][0] = 0.3;
    m[0][1] = 0.3;
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", std::move(m));
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 4000, 3);
    size_t ones = 0;
    for (uint8_t bit : s.measurements) {
        ones += bit;
    }
    REQUIRE(ones > 1220);  // expected 1400; generous band
    REQUIRE(ones < 1580);
}

TEST_CASE("exact: a measurement records the pre-relaxation bit") {
    // The transition's source column is read at op entry and its jump applies
    // after the base op: the first M reads the original |1>, the relaxation
    // then materializes g, and the second M reads the relaxed 0.
    Circuit c = parse("M 0\nM 0\n");
    auto m = zeros5();
    m[0][1] = 1.0;  // e relaxes to g with certainty; g stays
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("M", std::move(m));
    NonComputationalModel model = make_direct_model(all_e(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 5);
    REQUIRE(s.num_measurements == 2);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 1);  // pre-relaxation bit
        REQUIRE(s.measurements[shot * 2 + 1] == 0);  // relaxed |0>
    }
}

TEST_CASE("exact: recapturing a lost qubit clears the stale residual") {
    // The first M loses the |1> qubit (its trace-out reset clears the
    // carrier). The second M is classified (lost at entry) and its
    // attached recapture jump materializes g; the third M must read the
    // recaptured 0.
    Circuit c = parse("M 0\nM 0\nM 0\n");
    auto m = zeros5();
    m[kLost][1] = 1.0;  // e is lost with certainty
    m[0][kLost] = 1.0;  // a lost qubit is recaptured at g with certainty
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("M", std::move(m));
    NonComputationalModel model =
        make_direct_model(all_e(), std::move(transitions), classifier_with(kLost, {1.0, 0.0}));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 7);
    REQUIRE(s.num_measurements == 3);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 3 + 0] == 1);  // real pre-loss bit
        REQUIRE(s.measurements[shot * 3 + 1] == 0);  // classifier bit for the lost site
        REQUIRE(s.measurements[shot * 3 + 2] == 0);  // recaptured |0>, residual cleared
    }
}

TEST_CASE("exact: a multi-round circuit runs through loss") {
    // The data qubit is lost up front; both syndrome CXs drop (identity on
    // the ancilla, which keeps reading 0) and the final data measurement
    // reads the classifier's lost bit -- dropping ops on a vacated site is
    // the (only) op policy.
    Circuit c = parse("H 0\nS 0\nCX 0 1\nMR 1\nCX 0 1\nMR 1\nM 0\n");
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("S", certain_loss());

    NonComputationalModel model =
        make_direct_model(all_g(), std::move(transitions), classifier_with(kLost, {0.0, 1.0}));
    NonComputationalSample r = sample_noncomputational(c, model, 16, 3);
    REQUIRE(r.num_measurements == 3);
    for (uint32_t shot = 0; shot < 16; ++shot) {
        REQUIRE(r.measurements[shot * 3 + 0] == 0);  // ancilla round 1
        REQUIRE(r.measurements[shot * 3 + 1] == 0);  // ancilla round 2
        REQUIRE(r.measurements[shot * 3 + 2] == 1);  // lost data, classifier bit
        REQUIRE(r.final_status[shot * 2 + 0] == QubitStatus::Lost);
    }
}

TEST_CASE("exact: computational confusion misreports into the detector") {
    // A certain 0->1 misreport: the record and the detector read 1 on every
    // shot even though the qubit is |0> -- the flip is in-circuit, not
    // postprocessing.
    Circuit c = parse("M 0\nDETECTOR rec[-1]\n");
    NonComputationalModel model = make_direct_model(all_g(), {}, comp_confusion(1.0, 0.0));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 3);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot] == 1);
        REQUIRE(s.detectors[shot] == 1);
    }
}

TEST_CASE("exact: asymmetric confusion matches its rates") {
    // True bit is 1 (X-prepared); it is misread as 0 with probability 0.2.
    Circuit c = parse("X 0\nM 0\n");
    NonComputationalModel model = make_direct_model(all_g(), {}, comp_confusion(0.0, 0.2));

    constexpr uint32_t kShots = 4000;
    NonComputationalSample s = sample_noncomputational(c, model, kShots, 9);
    size_t zeros = 0;
    for (uint8_t bit : s.measurements) {
        zeros += bit == 0 ? 1 : 0;
    }
    REQUIRE(zeros > 650);  // expected 800; ~6 sigma band
    REQUIRE(zeros < 950);
}

TEST_CASE("exact: hand-written LOSS and LEVEL_TRANSITION run end to end") {
    // Qubit 0 is lost by a local LOSS; its measurement takes the classifier's
    // lost bit. Qubit 1 leaks via a local named LEVEL_TRANSITION; its measurement
    // takes the leak_g bit.
    Circuit c = parse("H 0\nH 1\nLOSS(1) 0\nLEVEL_TRANSITION[leak] 1\nM 0\nM 1\n");
    auto leak = zeros5();
    leak[kLeakG][0] = 1.0;
    leak[kLeakG][1] = 1.0;
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    transitions.emplace("leak", std::move(leak));
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    m[1][kLeakG] = 1.0;  // leaked reads 1
    m[0][kLost] = 1.0;   // lost reads 0
    m[0][3] = 1.0;
    NonComputationalModel model = make_direct_model(all_g(), std::move(transitions), std::move(m));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 5);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 0);  // lost slot: classifier bit 0
        REQUIRE(s.measurements[shot * 2 + 1] == 1);  // leaked slot: classifier bit 1
        REQUIRE(s.final_status[shot * 2 + 0] == QubitStatus::Lost);
        REQUIRE(s.final_status[shot * 2 + 1] == QubitStatus::LeakG);
    }
}
