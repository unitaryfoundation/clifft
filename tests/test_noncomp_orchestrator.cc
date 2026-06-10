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
using clifft::QubitStatusKind;
using clifft::sample_noncomputational;
using clifft::TransitionInstrument;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLost = 4;

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

// Source-independent: g and e both jump to lost with certainty.
TransitionInstrument always_lost(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent: g and e both jump to the computational g level.
TransitionInstrument always_to_g(const LevelSet& levels) {
    auto m = zeros5();
    m[0][0] = 1.0;
    m[0][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent: g and e both jump to the computational e level.
TransitionInstrument always_to_e(const LevelSet& levels) {
    auto m = zeros5();
    m[1][0] = 1.0;
    m[1][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Two-symbol classifier whose column for `level` is `col`; every other level
// is a deterministic symbol 0. Only noncomputational qubits are classified, so
// the computational columns are never consulted in these tests.
MeasurementClassifier classifier_with(const LevelSet& levels, uint8_t level,
                                      std::vector<double> col) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;  // symbol "0"
    }
    m[0][level] = col[0];
    m[1][level] = col[1];
    return MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
}

// The common case: classify the leak_g column.
MeasurementClassifier make_classifier(const LevelSet& levels, std::vector<double> leakg) {
    return classifier_with(levels, kLeakG, std::move(leakg));
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
std::vector<double> all_e() {
    return {0.0, 1.0, 0.0, 0.0, 0.0};
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

TEST_CASE("sample_noncomputational: a substochastic classifier column is unsupported and raises") {
    Circuit c = parse("H 0\nS 0\nM 0\n");

    // A partially substochastic leak_g column (sums to 0.8) reserves reject
    // probability, which this entry point does not support.
    std::map<std::string, TransitionInstrument> partial;
    partial.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel partial_model = make_model(
        all_g(), std::move(partial), make_classifier(LevelSet::default_set(), {0.5, 0.3}));
    REQUIRE_THROWS_WITH(sample_noncomputational(c, partial_model, 16, 1),
                        ContainsSubstring("classifier reject columns are not supported"));

    // A fully reject column (sums to 0) is refused the same way, not sampled.
    std::map<std::string, TransitionInstrument> empty;
    empty.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel empty_model =
        make_model(all_g(), std::move(empty), make_classifier(LevelSet::default_set(), {0.0, 0.0}));
    REQUIRE_THROWS_WITH(sample_noncomputational(c, empty_model, 16, 1),
                        ContainsSubstring("classifier reject columns are not supported"));
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

TEST_CASE(
    "sample_noncomputational: a lost-qubit measurement feeds the detector the classifier bit") {
    // A lost qubit (vacated site) is still measured; the classifier supplies
    // the record bit just as for a leaked qubit, so the detector reads it.
    Circuit c = parse("H 0\nS 0\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));

    // Force the lost column -> symbol 1 -> record bit 1.
    NonComputationalModel model =
        make_model(all_g(), std::move(transitions),
                   classifier_with(LevelSet::default_set(), kLost, {0.0, 1.0}));
    NonComputationalSample s = sample_noncomputational(c, model, 200, 9);
    REQUIRE(s.num_detectors == 1);
    REQUIRE(s.detectors.size() == 200);
    for (uint8_t d : s.detectors) {
        REQUIRE(d == 1);  // detector saw the lost-column classifier bit
    }
}

TEST_CASE("sample_noncomputational: a leaked measurement feeds the observable the classifier bit") {
    Circuit c = parse("H 0\nS 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));

    // Force leak_g -> symbol 1 -> record bit 1; the observable must see it,
    // not the residual |0>.
    NonComputationalModel model = make_model(all_g(), std::move(transitions),
                                             make_classifier(LevelSet::default_set(), {0.0, 1.0}));
    NonComputationalSample s = sample_noncomputational(c, model, 200, 11);
    REQUIRE(s.num_observables == 1);
    REQUIRE(s.observables.size() == 200);
    for (uint8_t o : s.observables) {
        REQUIRE(o == 1);
    }
}

TEST_CASE("sample_noncomputational: a jump to the ground level forces the measurement to 0") {
    // The S transition collapses the H-prepared |+> to the g level; without
    // the materializing carrier edit the M would read 1 on ~half the shots.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_to_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 200, 1);
    for (uint8_t bit : s.measurements) {
        REQUIRE(bit == 0);
    }
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.final_status[shot].kind() == QubitStatusKind::ComputationalKnown);
        REQUIRE(s.final_status[shot].level_id() == 0);
    }
}

TEST_CASE("sample_noncomputational: a jump to the excited level forces the measurement to 1") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_to_e(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 200, 1);
    for (uint8_t bit : s.measurements) {
        REQUIRE(bit == 1);
    }
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.final_status[shot].kind() == QubitStatusKind::ComputationalKnown);
        REQUIRE(s.final_status[shot].level_id() == 1);
    }
}

TEST_CASE("sample_noncomputational: a partial jump to ground matches the analytic mixture") {
    // With probability p the S transition collapses the H-prepared |+> to g;
    // otherwise the carrier stays coherent. P(M = 1) = (1 - p) / 2 = 0.35.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    auto m = zeros5();
    m[0][0] = 0.3;
    m[0][1] = 0.3;
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S",
                        TransitionInstrument::from_matrix(std::move(m), LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 4000, 3);
    size_t ones = 0;
    for (uint8_t bit : s.measurements) {
        ones += bit;
    }
    REQUIRE(ones > 1220);  // expected 1400; generous band
    REQUIRE(ones < 1580);
}

TEST_CASE("sample_noncomputational: a measurement records the pre-relaxation bit") {
    // The transition's source column is read at op entry and its jump applies
    // after the base op: the first M reads the original |1>, the relaxation
    // then materializes g, and the second M reads the relaxed 0.
    Circuit c = parse("M 0\nM 0\n");
    auto m = zeros5();
    m[0][1] = 1.0;  // e relaxes to g with certainty; g stays
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("M",
                        TransitionInstrument::from_matrix(std::move(m), LevelSet::default_set()));
    NonComputationalModel model = make_model(all_e(), std::move(transitions));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 5);
    REQUIRE(s.num_measurements == 2);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 1);  // pre-relaxation bit
        REQUIRE(s.measurements[shot * 2 + 1] == 0);  // relaxed |0>
    }
}

TEST_CASE("sample_noncomputational: recapturing a lost qubit clears the stale residual") {
    // The first M loses the known |1> qubit with no trace-out R (a definite
    // atom needs no unraveling), so a stale |1> residual stays in the SVM.
    // The second M is classified (lost at entry) and its attached recapture
    // jump materializes g; the third M must read the recaptured 0, not the
    // residual.
    Circuit c = parse("M 0\nM 0\nM 0\n");
    auto m = zeros5();
    m[kLost][1] = 1.0;  // e is lost with certainty
    m[0][kLost] = 1.0;  // a lost qubit is recaptured at g with certainty
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("M",
                        TransitionInstrument::from_matrix(std::move(m), LevelSet::default_set()));
    NonComputationalModel model =
        make_model(all_e(), std::move(transitions),
                   classifier_with(LevelSet::default_set(), kLost, {1.0, 0.0}));

    NonComputationalSample s = sample_noncomputational(c, model, 64, 7);
    REQUIRE(s.num_measurements == 3);
    for (uint32_t shot = 0; shot < s.shots; ++shot) {
        REQUIRE(s.measurements[shot * 3 + 0] == 1);  // real pre-loss bit
        REQUIRE(s.measurements[shot * 3 + 1] == 0);  // classifier bit for the lost site
        REQUIRE(s.measurements[shot * 3 + 2] == 0);  // recaptured |0>, residual cleared
    }
}
