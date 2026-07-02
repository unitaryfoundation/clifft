// Validation for the noncomputational lost-measurement path, on a Bell pair.
//
// Two different things are checked, because statistics and structure validate
// different claims:
//
//   * Statistics (lossless correlation, lost-record independence, survivor
//     marginal) exercise classifier injection and the end-to-end pipeline.
//     They do NOT, on their own, validate the hidden trace-out: once a lost
//     qubit's record is replaced by MPAD(classifier_bit) and the qubit never
//     re-enters the circuit, a ghost-entangled carrier is observationally
//     identical to a traced-out one on the surviving qubit (a Bell-pair half is
//     I/2 either way). Distinguishing the two would need the carrier to
//     re-enter -- out of scope here.
//   * Structure: rewriting the Bell loss circuit inserts exactly one hidden
//     trace-out R at the loss site, lowering to one surviving hidden
//     measurement. This is the check that the partial-trace unraveling is
//     actually emitted.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

using clifft::annotate;
using clifft::Circuit;
using clifft::default_hir_pass_manager;
using clifft::GateType;
using clifft::HirModule;
using clifft::HistorySample;
using clifft::LevelSet;
using clifft::MeasurementClassifier;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::NonComputationalSample;
using clifft::parse;
using clifft::rewrite;
using clifft::sample_history;
using clifft::sample_noncomputational;
using clifft::trace;
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

// Two-symbol classifier whose lost column is `col`; computational levels
// read out faithfully and leaked levels deterministically read symbol 0.
MeasurementClassifier lost_classifier(const LevelSet& levels, std::vector<double> col) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    // Computational levels read out faithfully (identity columns) so the
    // classifier adds no readout confusion here.
    m[0][1] = 0.0;
    m[1][1] = 1.0;
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

size_t count_gate(const Circuit& c, GateType gate) {
    size_t n = 0;
    for (const auto& node : c.nodes) {
        if (node.gate == gate) {
            ++n;
        }
    }
    return n;
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

TEST_CASE("validation: a lost Bell-pair qubit's classifier record is independent of the survivor") {
    // This exercises classifier injection, NOT the trace-out: the lost qubit's
    // record is supplied by the classifier (a 50/50 coin here), and the
    // survivor's record is an independent 50/50. The decorrelation is a
    // consequence of the M0 -> MPAD(classifier_bit) substitution, not of the
    // hidden R (see the file header and the structural test below).
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

    // Survivor (qubit 1) record is ~50/50. Expected 2048; ~6 sigma band.
    REQUIRE(ones1 > 1843);
    REQUIRE(ones1 < 2253);
    // The two records are independent: they disagree ~half the time (the
    // lossless run above never disagrees). Expected 2048.
    REQUIRE(mismatches > 1843);
    REQUIRE(mismatches < 2253);
    // All four (m0, m1) combinations occur with comparable weight (expected
    // 1024 each). Generous band.
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
    // Survivor unaffected: still ~50/50. Expected 1024; ~6 sigma band.
    REQUIRE(ones1 > 879);
    REQUIRE(ones1 < 1169);
}

TEST_CASE("validation: losing a Bell-pair qubit inserts the hidden trace-out R at the loss site") {
    // The actual trace-out check: structurally, the loss rewrite must emit the
    // hidden Z-basis unraveling for the coherent (entangled) carrier that jumps
    // to lost. Statistics cannot see this; gate counts and the lowered HIR can.
    Circuit c = parse("H 0\nCX 0 1\nS 0\nM 0\nM 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    // The lost qubit's later M needs a classifier column for its record bit.
    NonComputationalModel model =
        make_model(std::move(transitions), lost_classifier(LevelSet::default_set(), {1.0, 0.0}));

    Circuit annotated = annotate(c, model);
    HistorySample hs =
        sample_history(annotated, model, 1);  // always_lost: qubit 0 is deterministically lost
    Circuit rw = rewrite(annotated, hs.history, model).circuit;

    // The original circuit has no reset; the loss rewrite adds exactly one
    // trace-out R (and no X-prep, since both halves start in |0>).
    REQUIRE(count_gate(c, GateType::R) == 0);
    REQUIRE(count_gate(rw, GateType::R) == 1);

    // The trace-out R lowers to exactly one hidden measurement that survives
    // the default HIR passes -- the partial-trace unraveling reaching the SVM.
    // The visible measurement count is unchanged (the record layout is stable).
    HirModule base = trace(c);
    HirModule hir = trace(rw);
    default_hir_pass_manager().run(hir);
    REQUIRE(hir.num_hidden_measurements == base.num_hidden_measurements + 1);
    REQUIRE(hir.num_measurements == base.num_measurements);
}
