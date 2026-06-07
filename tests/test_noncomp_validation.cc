// Physics validation for the noncomputational pipeline: a Bell pair with one
// qubit lost. This is the headline check that the rewriter's hidden trace-out
// (Z-basis unraveling + reset) reproduces a partial trace over the lost qubit.
//
// A Bell pair (|00> + |11>)/sqrt(2) has perfectly Z-correlated halves. Tracing
// out one half must leave the survivor maximally mixed and destroy that
// correlation. These tests drive sample_noncomputational end to end and compare
// the lossless control (correlated) against the lossy run (decorrelated, 50/50
// survivor).
//
// Scope: this validates the observable statistics on the accessible qubits --
// correlation present without loss, gone with it, survivor maximally mixed. A
// single survivor's reduced state is I/2 in either case, so survivor marginals
// alone cannot distinguish a correct trace-out from a hypothetical no-op; the
// correlation contrast is what carries the signal here.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/transition_instrument.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

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
constexpr uint8_t kLost = 4;

std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-independent: g and e both jump to lost with certainty.
TransitionInstrument always_lost(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Two-symbol classifier whose lost column is `col` (the only column a lost
// qubit consults here); every other level deterministically reads symbol 0.
MeasurementClassifier lost_classifier(const LevelSet& levels, std::vector<double> col) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][kLost] = col[0];
    m[1][kLost] = col[1];
    return MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
}

NonComputationalModel make_model(std::map<std::string, TransitionInstrument> transitions,
                                 std::optional<MeasurementClassifier> classifier = std::nullopt) {
    // Both halves start in g (|0>); no leading X-prep is needed.
    return NonComputationalModel(LevelSet::default_set(), {1.0, 0.0, 0.0, 0.0, 0.0},
                                 std::move(transitions), std::move(classifier),
                                 NonComputationalPolicy{});
}

}  // namespace

TEST_CASE("validation: a lossless Bell pair measures as perfectly Z-correlated halves") {
    Circuit c = parse("H 0\nCX 0 1\nM 0\nM 1\n");
    NonComputationalModel model = make_model({});  // no loss

    const uint32_t shots = 1024;
    NonComputationalSample s = sample_noncomputational(c, model, shots, 1);
    REQUIRE(s.num_measurements == 2);

    size_t ones0 = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        const uint8_t m0 = s.measurements[shot * 2 + 0];
        const uint8_t m1 = s.measurements[shot * 2 + 1];
        REQUIRE(m0 == m1);  // ZZ stabilizer: the halves always agree
        ones0 += m0;
    }
    // Both outcomes occur (the correlation is not the trivial "always 0").
    REQUIRE(ones0 > 400);
    REQUIRE(ones0 < 624);
}

TEST_CASE("validation: losing one Bell-pair qubit leaves the survivor maximally mixed") {
    // S 0 is the carrier of the loss event: its transition sends the (now
    // entangled, ComputationalUnknown) qubit 0 to the lost level, which makes
    // the rewriter insert the hidden trace-out R. The classifier's lost column
    // is 50/50, so the lost qubit's own record bit is an independent coin.
    Circuit c = parse("H 0\nCX 0 1\nS 0\nM 0\nM 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model =
        make_model(std::move(transitions), lost_classifier(LevelSet::default_set(), {0.5, 0.5}));

    const uint32_t shots = 4096;
    NonComputationalSample s = sample_noncomputational(c, model, shots, 7);
    REQUIRE(s.num_measurements == 2);

    std::array<size_t, 4> joint{0, 0, 0, 0};  // index = m0 * 2 + m1
    size_t ones1 = 0;
    size_t mismatches = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        const uint8_t m0 = s.measurements[shot * 2 + 0];
        const uint8_t m1 = s.measurements[shot * 2 + 1];
        ++joint[m0 * 2 + m1];
        ones1 += m1;
        mismatches += (m0 != m1) ? 1 : 0;
    }

    // Survivor (qubit 1) is maximally mixed: ~50/50. Expected 2048; ~6 sigma.
    REQUIRE(ones1 > 1843);
    REQUIRE(ones1 < 2253);
    // Correlation is destroyed: the halves disagree ~half the time (the
    // lossless run above never disagrees). Expected 2048.
    REQUIRE(mismatches > 1843);
    REQUIRE(mismatches < 2253);
    // All four (m0, m1) combinations occur with comparable weight (expected
    // 1024 each), i.e. the two records are independent. Generous band.
    for (size_t cell = 0; cell < 4; ++cell) {
        REQUIRE(joint[cell] > 800);
        REQUIRE(joint[cell] < 1248);
    }
}

TEST_CASE(
    "validation: a deterministic classifier pins the lost record while the survivor stays free") {
    // Lost column [1, 0]: the lost qubit's record is deterministically 0, yet
    // the survivor is still an independent 50/50 -- the classifier governs the
    // vacated site's bit, not the surviving qubit.
    Circuit c = parse("H 0\nCX 0 1\nS 0\nM 0\nM 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model =
        make_model(std::move(transitions), lost_classifier(LevelSet::default_set(), {1.0, 0.0}));

    const uint32_t shots = 2048;
    NonComputationalSample s = sample_noncomputational(c, model, shots, 5);

    size_t ones1 = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        REQUIRE(s.measurements[shot * 2 + 0] == 0);  // classifier pins the lost record
        ones1 += s.measurements[shot * 2 + 1];
    }
    // Survivor unaffected: still ~50/50. Expected 1024; ~6 sigma.
    REQUIRE(ones1 > 879);
    REQUIRE(ones1 < 1169);
}
