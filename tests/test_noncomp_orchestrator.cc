#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"
#include "clifft/noncomp/transition_instrument.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::Circuit;
using clifft::LevelSet;
using clifft::MeasurementClassifier;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::NonComputationalSample;
using clifft::parse;
using clifft::sample_noncomputational;
using clifft::TransitionInstrument;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kLeakG = 2;

std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-independent: g and e both jump to leak_g with certainty.
TransitionInstrument always_leaked(const LevelSet& levels) {
    auto m = zeros5();
    m[kLeakG][0] = 1.0;
    m[kLeakG][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Two-symbol classifier. The leak_g column gets `leakg`; every other level is
// a deterministic symbol 0, which is never consulted (only noncomputational
// qubits are classified, and these tests only leak to leak_g).
MeasurementClassifier make_classifier(const LevelSet& levels, std::vector<double> leakg) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t level = 0; level < 5; ++level) {
        m[0][level] = 1.0;  // symbol "0"
    }
    m[0][kLeakG] = leakg[0];
    m[1][kLeakG] = leakg[1];
    return MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
}

NonComputationalModel make_model(std::vector<double> initial_state,
                                 std::map<std::string, TransitionInstrument> transitions,
                                 std::optional<MeasurementClassifier> classifier = std::nullopt,
                                 NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), std::move(initial_state),
                                 std::move(transitions), std::move(classifier), policy);
}

std::vector<double> all_g() {
    return {1.0, 0.0, 0.0, 0.0, 0.0};
}

}  // namespace

TEST_CASE("sample_noncomputational: a lossless model returns the plain record shape") {
    Circuit c = parse("H 0\nM 0\n");
    NonComputationalModel model = make_model(all_g(), {});

    NonComputationalSample s = sample_noncomputational(c, model, 1000, 7);
    REQUIRE(s.shots == 1000);
    REQUIRE(s.num_measurements == 1);
    REQUIRE(s.measurements.size() == 1000);
    REQUIRE(s.final_status.size() == 1000);  // one qubit per shot

    size_t ones = 0;
    for (uint8_t bit : s.measurements) {
        ones += bit;
    }
    REQUIRE(ones > 400);  // H 0 then M 0 is ~50/50; generous band
    REQUIRE(ones < 600);
}

TEST_CASE(
    "sample_noncomputational: a leaked measurement feeds the detector with the classifier bit") {
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));

    // Classifier forces leak_g -> symbol 1 -> record bit 1.
    NonComputationalModel one =
        make_model(all_g(), transitions, make_classifier(LevelSet::default_set(), {0.0, 1.0}));
    NonComputationalSample s1 = sample_noncomputational(c, one, 200, 1);
    REQUIRE(s1.num_detectors == 1);
    REQUIRE(s1.detectors.size() == 200);
    for (uint8_t d : s1.detectors) {
        REQUIRE(d == 1);  // detector saw the forced classifier bit, not the residual |0>
    }

    // Classifier forces leak_g -> symbol 0 -> record bit 0.
    NonComputationalModel zero = make_model(all_g(), std::move(transitions),
                                            make_classifier(LevelSet::default_set(), {1.0, 0.0}));
    NonComputationalSample s0 = sample_noncomputational(c, zero, 200, 1);
    for (uint8_t d : s0.detectors) {
        REQUIRE(d == 0);
    }
}

TEST_CASE("sample_noncomputational: a partial classifier bit matches its frequency") {
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions),
                                             make_classifier(LevelSet::default_set(), {0.5, 0.5}));

    NonComputationalSample s = sample_noncomputational(c, model, 2000, 3);
    size_t ones = 0;
    for (uint8_t d : s.detectors) {
        ones += d;
    }
    REQUIRE(ones > 850);  // expected 1000; generous band
    REQUIRE(ones < 1150);
}

TEST_CASE("sample_noncomputational: a rejecting classifier raises") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    // leak_g column sums to 0 -> reject probability 1.
    NonComputationalModel model = make_model(all_g(), std::move(transitions),
                                             make_classifier(LevelSet::default_set(), {0.0, 0.0}));

    REQUIRE_THROWS_WITH(sample_noncomputational(c, model, 16, 1),
                        ContainsSubstring("classifier rejected"));
}

TEST_CASE("sample_noncomputational: a measurement on a leaked qubit without a classifier raises") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));  // no classifier

    REQUIRE_THROWS_WITH(sample_noncomputational(c, model, 16, 1),
                        ContainsSubstring("requires a classifier"));
}

TEST_CASE("sample_noncomputational: a non-binary classifier rejects on injection") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));

    // Three symbols: no defined symbol-to-record-bit mapping for the binary record.
    std::vector<std::vector<double>> m(3, std::vector<double>(5, 0.0));
    for (size_t level = 0; level < 5; ++level) {
        m[0][level] = 1.0;
    }
    m[0][kLeakG] = 0.5;
    m[1][kLeakG] = 0.3;
    m[2][kLeakG] = 0.2;
    MeasurementClassifier three =
        MeasurementClassifier::from_matrix({"0", "1", "2"}, std::move(m), LevelSet::default_set());
    NonComputationalModel model = make_model(all_g(), std::move(transitions), std::move(three));

    REQUIRE_THROWS_WITH(sample_noncomputational(c, model, 16, 1),
                        ContainsSubstring("two-symbol classifier"));
}

TEST_CASE("sample_noncomputational: a circuit with EXP_VAL probes rejects") {
    Circuit c;
    c.num_qubits = 1;
    c.num_exp_vals = 1;  // EXP_VAL output is not carried by the noncomp sidecar
    NonComputationalModel model = make_model(all_g(), {});
    REQUIRE_THROWS_WITH(sample_noncomputational(c, model, 4, 1), ContainsSubstring("EXP_VAL"));
}

TEST_CASE("sample_noncomputational: zero shots still reports the record shape") {
    Circuit c = parse("H 0\nM 0\nDETECTOR rec[-1]\n");
    NonComputationalModel model = make_model(all_g(), {});

    NonComputationalSample s = sample_noncomputational(c, model, 0, 1);
    REQUIRE(s.shots == 0);
    REQUIRE(s.num_measurements == 1);  // record widths hold even with no shots
    REQUIRE(s.num_detectors == 1);
    REQUIRE(s.measurements.empty());
    REQUIRE(s.detectors.empty());
}

TEST_CASE("sample_noncomputational: a measure-and-reset on a leaked qubit injects and restores") {
    // MR on the leaked qubit records the classifier bit AND resets the site to
    // |0>; the following M then deterministically reads 0 -- proving the reset
    // actually ran in the SVM, not just in the trajectory bookkeeping.
    Circuit c = parse("H 0\nS 0\nMR 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions),
                                             make_classifier(LevelSet::default_set(), {0.0, 1.0}));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 5);
    REQUIRE(s.num_measurements == 2);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 1);  // MR slot: forced classifier bit
        REQUIRE(s.measurements[shot * 2 + 1] == 0);  // M slot: restored |0>
    }
}

TEST_CASE("sample_noncomputational: deterministic in the seed") {
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions),
                                             make_classifier(LevelSet::default_set(), {0.5, 0.5}));

    NonComputationalSample a = sample_noncomputational(c, model, 128, 42);
    NonComputationalSample b = sample_noncomputational(c, model, 128, 42);
    REQUIRE(a.measurements == b.measurements);
    REQUIRE(a.detectors == b.detectors);
}
