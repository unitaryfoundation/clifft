// Per-node rewriter semantics, exercised through rewrite_continuation --
// the one rewrite entry: the policy scan (keep / drop / reject), classifier
// record writes and their slot targeting, computational readout confusion,
// carrier edits for recorded jumps, and the annotate() hook expansion the
// rewriter consumes. Noncomputational statuses come from initial statuses
// or recorded events; a coherent qubit's annotation stays a runtime
// instrument site (the driver's territory, tested in test_exact_driver).

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::annotate;
using clifft::AstNode;
using clifft::Circuit;
using clifft::ContinuationRewrite;
using clifft::default_hir_pass_manager;
using clifft::ExactShotEvents;
using clifft::GateType;
using clifft::HirModule;
using clifft::LevelSet;
using clifft::MeasurementClassifier;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::parse;
using clifft::QubitStatus;
using clifft::rewrite_continuation;
using clifft::trace;
using clifft::TransitionInstrument;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kG = 0;
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLost = 4;

std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-independent: always jumps to lost.
TransitionInstrument always_lost(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

NonComputationalModel make_model(std::map<std::string, TransitionInstrument> transitions,
                                 NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), {1.0, 0.0, 0.0, 0.0, 0.0},
                                 std::move(transitions), std::nullopt, policy);
}

// Classifier whose column for `level` is `col` (two or three symbols by
// col's length); computational levels read out faithfully and other levels
// default to a deterministic symbol 0.
MeasurementClassifier classifier_with(uint8_t level, std::vector<double> col) {
    std::vector<std::vector<double>> m(col.size(), std::vector<double>(5, 0.0));
    std::vector<std::string> symbols;
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][1] = 0.0;
    m[1][1] = 1.0;
    for (size_t s = 0; s < col.size(); ++s) {
        m[s][level] = col[s];
        symbols.push_back(std::to_string(s));
    }
    return MeasurementClassifier::from_matrix(std::move(symbols), std::move(m),
                                              LevelSet::default_set());
}

NonComputationalModel make_model_with_classifier(
    std::map<std::string, TransitionInstrument> transitions, MeasurementClassifier classifier,
    NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), {1.0, 0.0, 0.0, 0.0, 0.0},
                                 std::move(transitions), std::move(classifier), policy);
}

// Per-qubit initial statuses from level ids.
std::vector<QubitStatus> initials(const std::vector<uint8_t>& levels) {
    std::vector<QubitStatus> out;
    out.reserve(levels.size());
    for (uint8_t l : levels) {
        out.push_back(LevelSet::default_set().status_for(l));
    }
    return out;
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

// Expand hooks and rewrite under the given initial statuses and events.
ContinuationRewrite rewritten(const Circuit& c, const NonComputationalModel& model,
                              const std::vector<uint8_t>& initial_levels,
                              ExactShotEvents events = {}) {
    Circuit annotated = annotate(c, model);
    events.initial_status = initials(initial_levels);
    return rewrite_continuation(annotated, events, false, model);
}

}  // namespace

// =========================================================================
// Carrier edits for recorded jumps
// =========================================================================

TEST_CASE("rewrite: a coherent qubit's recorded jump to lost gets a trace-out R") {
    Circuit c = parse("H 0\nLEVEL_TRANSITION[lk] 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("lk", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    ExactShotEvents events;
    events.jumps.push_back({/*op_index=*/1, /*qubit=*/0, /*destination_level=*/kLost});
    ContinuationRewrite rw = rewritten(c, model, {kG}, events);
    // H and the site kept; one trace-out R follows the site.
    REQUIRE(rw.circuit.nodes.size() == 3);
    REQUIRE(rw.circuit.nodes[1].gate == GateType::LEVEL_TRANSITION);
    REQUIRE(rw.circuit.nodes[2].gate == GateType::R);
    REQUIRE(rw.circuit.nodes[2].targets[0].value() == 0);
}

TEST_CASE("rewrite: a recorded jump to the |0> level inserts an R, no X") {
    Circuit c = parse("H 0\nLEVEL_TRANSITION[lk] 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("lk", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    ExactShotEvents events;
    events.jumps.push_back({1, 0, /*destination_level=*/kG});
    ContinuationRewrite rw = rewritten(c, model, {kG}, events);
    REQUIRE(count_gate(rw.circuit, GateType::R) == 1);  // materialize at |0>
    REQUIRE(count_gate(rw.circuit, GateType::X) == 0);
}

TEST_CASE("rewrite: an inserted trace-out R survives compilation as one hidden measurement") {
    Circuit c = parse("H 0\nLEVEL_TRANSITION[lk] 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("lk", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    ExactShotEvents events;
    events.jumps.push_back({1, 0, kLost});
    ContinuationRewrite rw = rewritten(c, model, {kG}, events);

    // Baseline: the same rewrite with no jump recorded (the site runs live).
    ContinuationRewrite base = rewritten(c, model, {kG});
    clifft::InstrumentTraceOptions options = clifft::instrument_trace_options(model);
    HirModule base_hir = trace(base.circuit, &options);
    HirModule hir = trace(rw.circuit, &options);
    default_hir_pass_manager().run(hir);

    REQUIRE(hir.num_hidden_measurements == base_hir.num_hidden_measurements + 1);
    REQUIRE(hir.num_measurements == base_hir.num_measurements);
}

TEST_CASE("rewrite: an inserted R does not shift visible measurements or detectors") {
    Circuit c =
        parse("M 0\nDETECTOR rec[-1]\nH 1\nLEVEL_TRANSITION[lk] 1\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("lk", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    ExactShotEvents events;
    events.jumps.push_back({3, 1, kLost});
    ContinuationRewrite rw = rewritten(c, model, {kG, kG}, events);
    ContinuationRewrite base = rewritten(c, model, {kG, kG});

    clifft::InstrumentTraceOptions options = clifft::instrument_trace_options(model);
    HirModule base_hir = trace(base.circuit, &options);
    HirModule hir = trace(rw.circuit, &options);
    default_hir_pass_manager().run(hir);

    REQUIRE(hir.num_measurements == base_hir.num_measurements);
    REQUIRE(hir.num_detectors == base_hir.num_detectors);
    REQUIRE(hir.detector_targets == base_hir.detector_targets);
    REQUIRE(hir.num_hidden_measurements == base_hir.num_hidden_measurements + 1);
}

// =========================================================================
// Policy scan: keep / drop / reject
// =========================================================================

TEST_CASE("rewrite: a two-qubit gate on a lost operand drops whole") {
    Circuit c = parse("CZ 0 1\nM 1\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLost, kG});
    REQUIRE(count_gate(rw.circuit, GateType::CZ) == 0);  // identity on the survivor
    REQUIRE(count_gate(rw.circuit, GateType::M) == 1);   // record slot preserved
}

TEST_CASE("rewrite: a dropped gate leaves the surviving operand's status untouched") {
    // The CZ drops whole (lost operand), so qubit 1 keeps its
    // instruction-known g status through to the end of the walk.
    Circuit c = parse("CZ 0 1\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLost, kG});
    REQUIRE(rw.final_status[1].kind() == clifft::QubitStatusKind::ComputationalKnown);
    REQUIRE(rw.final_status[1].level_id() == kG);
}

TEST_CASE("rewrite: a single-qubit gate on a leaked qubit drops") {
    Circuit c = parse("X 0\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLeakG});
    REQUIRE(count_gate(rw.circuit, GateType::X) == 0);
}

TEST_CASE("rewrite: a single-qubit gate on a lost qubit drops") {
    Circuit c = parse("X 0\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLost});
    REQUIRE(count_gate(rw.circuit, GateType::X) == 0);
}

TEST_CASE("rewrite: classical feedback onto a lost qubit drops") {
    Circuit c = parse("H 1\nM 1\nCX rec[-1] 0\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLost, kG});
    REQUIRE(count_gate(rw.circuit, GateType::CX) == 0);
}

TEST_CASE("rewrite: a two-qubit noise channel on a lost operand drops") {
    Circuit c = parse("DEPOLARIZE2(0.1) 0 1\n");
    NonComputationalModel model = make_model({});
    ContinuationRewrite rw = rewritten(c, model, {kLost, kG});
    REQUIRE(count_gate(rw.circuit, GateType::DEPOLARIZE2) == 0);
}

TEST_CASE("rewrite: a non-restoring lost reset drops; reset_restores_lost keeps it") {
    Circuit c = parse("R 0\n");

    NonComputationalModel dropped = make_model({});
    ContinuationRewrite rw_drop = rewritten(c, dropped, {kLost});
    REQUIRE(count_gate(rw_drop.circuit, GateType::R) == 0);
    REQUIRE(rw_drop.final_status[0].kind() == clifft::QubitStatusKind::Lost);  // not restored

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload = make_model({}, restore);
    ContinuationRewrite rw_keep = rewritten(c, reload, {kLost});
    REQUIRE(count_gate(rw_keep.circuit, GateType::R) == 1);
    REQUIRE(rw_keep.final_status[0].kind() == clifft::QubitStatusKind::ComputationalKnown);
}

TEST_CASE("rewrite: an X/Y-basis measurement of a noncomputational qubit rejects") {
    // No faithful single-bit form on a leaked/lost operand: a
    // representability limit, not a policy choice, so it rejects even though
    // drop is the only op policy.
    Circuit c = parse("MX 0\n");
    NonComputationalModel model = make_model({});
    REQUIRE_THROWS_WITH(rewritten(c, model, {kLeakG}),
                        ContainsSubstring("MX") && ContainsSubstring("representable"));
}

// =========================================================================
// Classifier record writes
// =========================================================================

TEST_CASE("rewrite: a lost-qubit measurement becomes a classifier record write") {
    Circuit c = parse("M 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLost, {0.5, 0.5}));

    ContinuationRewrite rw = rewritten(c, model, {kLost});
    REQUIRE(count_gate(rw.circuit, GateType::M) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 1);
    REQUIRE(rw.circuit.num_measurements == 1);  // visible record preserved

    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].slot == 0);
    REQUIRE(rw.classified_measurements[0].level == kLost);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.targets[0].is_rec());
    REQUIRE(noise.targets[0].value() == 0);
    REQUIRE(noise.args[0] == 0.5);
}

TEST_CASE("rewrite: a deterministic classifier column pads the literal bit, no draw") {
    Circuit c = parse("M 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLost, {0.0, 1.0}));

    ContinuationRewrite rw = rewritten(c, model, {kLost});
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);
    for (const AstNode& node : rw.circuit.nodes) {
        if (node.gate == GateType::MPAD) {
            REQUIRE(node.targets[0].value() == 1);  // the bit is the padding literal
        }
    }
    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].noise_node == SIZE_MAX);
}

TEST_CASE("rewrite: the record write flips its own slot, not an earlier one") {
    // Slot 0 is a computational measurement on qubit 1; the lost qubit's
    // measurement is slot 1 and its READOUT_NOISE must target slot 1.
    Circuit c = parse("M 1\nM 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLost, {0.5, 0.5}));

    ContinuationRewrite rw = rewritten(c, model, {kLost, kG});
    REQUIRE(count_gate(rw.circuit, GateType::M) == 1);  // the computational one
    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].slot == 1);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.targets[0].value() == 1);
}

TEST_CASE("rewrite: a ternary column emits the not-heralded conditional flip") {
    // Column {0.3, 0.1, 0.6}: conditioned on not heralding, the bit is
    // 0.1 / (1 - 0.6) = 0.25. A ternary slot always gets a READOUT_NOISE so
    // the driver's herald patching has a node to re-point at one half.
    Circuit c = parse("M 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLeakG, {0.3, 0.1, 0.6}));

    ContinuationRewrite rw = rewritten(c, model, {kLeakG});
    REQUIRE(rw.classified_measurements.size() == 1);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.args[0] == Catch::Approx(0.25));
}

TEST_CASE("rewrite: an always-herald ternary column still emits a patchable node") {
    // {0, 0, 1} has no not-heralded conditional; one half stands in, and the
    // herald patching overwrites it on every (always-heralding) draw.
    Circuit c = parse("M 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLeakG, {0.0, 0.0, 1.0}));

    ContinuationRewrite rw = rewritten(c, model, {kLeakG});
    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].noise_node != SIZE_MAX);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.args[0] == 0.5);
}

TEST_CASE("rewrite: a noncomputational measurement without a classifier rejects") {
    Circuit c = parse("M 0\n");
    NonComputationalModel model = make_model({});
    REQUIRE_THROWS_WITH(rewritten(c, model, {kLost}), ContainsSubstring("requires a classifier"));
}

TEST_CASE("rewrite: an X/Y-basis or multi-qubit measurement on a lost qubit rejects") {
    NonComputationalModel model = make_model({});
    REQUIRE_THROWS_WITH(rewritten(parse("MX 0\n"), model, {kLost}),
                        ContainsSubstring("MX") && ContainsSubstring("Lost"));
    REQUIRE_THROWS_WITH(rewritten(parse("MPP X0\n"), model, {kLost}),
                        ContainsSubstring("MPP") && ContainsSubstring("Lost"));
}

TEST_CASE("rewrite: reset_restores_lost restores a measure-and-reset's lost qubit") {
    // MR on a lost qubit is kept: the classifier supplies the record bit and
    // the reset runs. With reset_restores_lost the reset returns the qubit to
    // a computational state.
    Circuit c = parse("MR 0\n");
    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload =
        make_model_with_classifier({}, classifier_with(kLost, {1.0, 0.0}), restore);
    ContinuationRewrite rw = rewritten(c, reload, {kLost});
    // The MR splits into the classifier record write plus its kept reset.
    REQUIRE(count_gate(rw.circuit, GateType::MR) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);
    REQUIRE(count_gate(rw.circuit, GateType::R) == 1);
    REQUIRE(rw.circuit.num_measurements == 1);  // visible record preserved
    REQUIRE(rw.final_status[0].kind() == clifft::QubitStatusKind::ComputationalKnown);
}

TEST_CASE("rewrite: a measure-and-reset on a leaked qubit records and resets") {
    Circuit c = parse("MR 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLeakG, {0.5, 0.5}));

    ContinuationRewrite rw = rewritten(c, model, {kLeakG});
    REQUIRE(count_gate(rw.circuit, GateType::MR) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 1);
    REQUIRE(count_gate(rw.circuit, GateType::R) == 1);  // the MR's kept reset
    REQUIRE(rw.final_status[0].kind() == clifft::QubitStatusKind::ComputationalKnown);
}

TEST_CASE("rewrite: a measure-and-reset on a non-restoring lost qubit is kept") {
    Circuit c = parse("MR 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLost, {1.0, 0.0}));

    ContinuationRewrite rw = rewritten(c, model, {kLost});
    REQUIRE(count_gate(rw.circuit, GateType::MR) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);  // record slot preserved
    REQUIRE(rw.circuit.num_measurements == 1);
    REQUIRE(rw.final_status[0].kind() == clifft::QubitStatusKind::Lost);  // not restored
}

// =========================================================================
// Computational readout confusion
// =========================================================================

namespace {

// Classifier with confused computational columns: a true 0 is misread as 1
// with probability p01, a true 1 as 0 with probability p10; noncomputational
// levels deterministically read symbol 0.
MeasurementClassifier confused_classifier(double p01, double p10) {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][0] = 1.0 - p01;
    m[1][0] = p01;
    m[0][1] = p10;
    m[1][1] = 1.0 - p10;
    return MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), LevelSet::default_set());
}

}  // namespace

TEST_CASE("rewrite: computational readout confusion appends an asymmetric flip") {
    Circuit c = parse("H 0\nM 0\n");
    NonComputationalModel model = make_model_with_classifier({}, confused_classifier(0.1, 0.2));

    ContinuationRewrite rw = rewritten(c, model, {kG});
    REQUIRE(count_gate(rw.circuit, GateType::M) == 1);  // measurement kept
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 1);
    const AstNode& noise = rw.circuit.nodes.back();
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.targets[0].is_rec());
    REQUIRE(noise.targets[0].value() == 0);
    REQUIRE(noise.args.size() == 2);
    REQUIRE(noise.args[0] == 0.1);                // P(symbol 1 | zero level)
    REQUIRE(noise.args[1] == 0.2);                // P(symbol 0 | one level)
    REQUIRE(rw.classified_measurements.empty());  // not a noncomp record write
}

TEST_CASE("rewrite: identity computational columns add no readout confusion") {
    Circuit c = parse("H 0\nM 0\n");
    NonComputationalModel model =
        make_model_with_classifier({}, classifier_with(kLost, {0.5, 0.5}));

    ContinuationRewrite rw = rewritten(c, model, {kG});
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 0);
}

TEST_CASE("rewrite: an inverted measurement swaps the confusion probabilities") {
    // Confusion physically precedes the reporting convention, and
    // invert-after-flip(p01, p10) equals flip(p10, p01)-after-invert.
    Circuit c = parse("H 0\nM !0\n");
    NonComputationalModel model = make_model_with_classifier({}, confused_classifier(0.1, 0.2));

    ContinuationRewrite rw = rewritten(c, model, {kG});
    const AstNode& noise = rw.circuit.nodes.back();
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.args[0] == 0.2);
    REQUIRE(noise.args[1] == 0.1);
}

TEST_CASE("rewrite: confusion applies to MR but not X-basis measurements") {
    Circuit c = parse("MR 0\nMX 0\n");
    NonComputationalModel model = make_model_with_classifier({}, confused_classifier(0.1, 0.2));

    ContinuationRewrite rw = rewritten(c, model, {kG});
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 1);
    for (const AstNode& node : rw.circuit.nodes) {
        if (node.gate == GateType::READOUT_NOISE) {
            REQUIRE(node.targets[0].value() == 0);  // the MR slot, not the MX slot
        }
    }
}

TEST_CASE("rewrite: a substochastic computational column rejects") {
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][0] = 0.6;  // g column sums to 0.8: reject mass on a computational level
    m[1][0] = 0.2;
    m[0][1] = 0.0;
    m[1][1] = 1.0;
    MeasurementClassifier cl =
        MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), LevelSet::default_set());
    NonComputationalModel model = make_model_with_classifier({}, std::move(cl));

    REQUIRE_THROWS_WITH(rewritten(parse("M 0\n"), model, {kG}),
                        ContainsSubstring("computational level"));
}

TEST_CASE("rewrite: a computational column with herald mass rejects") {
    std::vector<std::vector<double>> m(3, std::vector<double>(5, 0.0));
    for (size_t l = 0; l < 5; ++l) {
        m[0][l] = 1.0;
    }
    m[0][0] = 0.9;  // g column puts 0.1 on the herald symbol
    m[2][0] = 0.1;
    m[0][1] = 0.0;
    m[1][1] = 1.0;
    MeasurementClassifier cl =
        MeasurementClassifier::from_matrix({"0", "1", "2"}, std::move(m), LevelSet::default_set());
    NonComputationalModel model = make_model_with_classifier({}, std::move(cl));

    REQUIRE_THROWS_WITH(rewritten(parse("M 0\n"), model, {kG}),
                        ContainsSubstring("beyond the bit"));
}

// =========================================================================
// annotate(): hook expansion
// =========================================================================

TEST_CASE("annotate: gate hooks expand to per-operand LEVEL_TRANSITION annotations") {
    Circuit c = parse("H 0\nCZ 0 1\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    Circuit ann = annotate(c, model);
    // H, CZ, LEVEL_TRANSITION(0), LEVEL_TRANSITION(1), M.
    REQUIRE(ann.nodes.size() == 5);
    REQUIRE(ann.nodes[2].gate == GateType::LEVEL_TRANSITION);
    REQUIRE(ann.nodes[2].tag == "CZ");
    REQUIRE(ann.nodes[2].targets[0].value() == 0);
    REQUIRE(ann.nodes[3].gate == GateType::LEVEL_TRANSITION);
    REQUIRE(ann.nodes[3].targets[0].value() == 1);
    REQUIRE(ann.num_measurements == c.num_measurements);  // layout untouched
}

TEST_CASE("annotate: feedback operands get no annotation") {
    Circuit c = parse("M 0\nCX rec[-1] 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CX", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(std::move(transitions));

    Circuit ann = annotate(c, model);
    REQUIRE(ann.nodes.size() == c.nodes.size());  // virtual correction: no consult point
}

TEST_CASE("annotate: an unhooked model leaves the circuit unchanged") {
    Circuit c = parse("H 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("my_leak", always_lost(LevelSet::default_set()));  // named, no hook
    NonComputationalModel model = make_model(std::move(transitions));

    Circuit ann = annotate(c, model);
    REQUIRE(ann.nodes.size() == c.nodes.size());
}
