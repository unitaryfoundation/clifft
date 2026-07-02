#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/annotate.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
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
using clifft::default_hir_pass_manager;
using clifft::GateType;
using clifft::HirModule;
using clifft::HistorySample;
using clifft::LevelSet;
using clifft::MeasurementClassifier;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::parse;
using clifft::rewrite;
using clifft::RewriteResult;
using clifft::sample_history;
using clifft::trace;
using clifft::TransitionInstrument;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLost = 4;

std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-dependent (g and e columns differ): g certainly jumps to lost, e
// does nothing. Allowed only on a known/leaked/lost source.
TransitionInstrument lose_from_g(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent (identical g and e columns): always jumps to lost.
TransitionInstrument always_lost(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    m[kLost][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent: always jumps to leak_g.
TransitionInstrument always_leaked(const LevelSet& levels) {
    auto m = zeros5();
    m[kLeakG][0] = 1.0;
    m[kLeakG][1] = 1.0;
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

// Source-dependent relaxation: e decays to g with certainty, g stays.
TransitionInstrument relax_e_to_g(const LevelSet& levels) {
    auto m = zeros5();
    m[0][1] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent: all columns zero, so a consult never fires.
TransitionInstrument never_fires(const LevelSet& levels) {
    return TransitionInstrument::from_matrix(zeros5(), levels);
}

// Source-independent on computational (both columns zero): only a lost
// qubit jumps, back to the computational g level (recapture).
TransitionInstrument recapture_lost_to_g(const LevelSet& levels) {
    auto m = zeros5();
    m[0][kLost] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

NonComputationalModel make_model(std::vector<double> initial_state,
                                 std::map<std::string, TransitionInstrument> transitions,
                                 NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), std::move(initial_state),
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
    // Computational levels read out faithfully (identity columns) so the
    // classifier adds no readout confusion here.
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
    std::vector<double> initial_state, std::map<std::string, TransitionInstrument> transitions,
    MeasurementClassifier classifier, NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), std::move(initial_state),
                                 std::move(transitions), std::move(classifier), policy);
}

std::vector<double> all_g() {
    return {1.0, 0.0, 0.0, 0.0, 0.0};
}
std::vector<double> all_e() {
    return {0.0, 1.0, 0.0, 0.0, 0.0};
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

// Expand the model's gate hooks, sample, and rewrite: the pipeline the
// orchestrator runs, for tests written against gate-hooked models.
Circuit rewritten(const Circuit& c, const NonComputationalModel& model, uint64_t seed) {
    Circuit annotated = annotate(c, model);
    HistorySample s = sample_history(annotated, model, seed);
    return rewrite(annotated, s.history, model).circuit;
}

RewriteResult rewritten_result(const Circuit& c, const NonComputationalModel& model,
                               uint64_t seed) {
    Circuit annotated = annotate(c, model);
    HistorySample s = sample_history(annotated, model, seed);
    return rewrite(annotated, s.history, model);
}

}  // namespace

TEST_CASE("rewrite: a known |1> initial sample prepends an X, |0> does not") {
    Circuit c;
    c.num_qubits = 1;

    Circuit prep_e = rewritten(c, make_model(all_e(), {}), 1);
    REQUIRE(prep_e.nodes.size() == 1);
    REQUIRE(prep_e.nodes[0].gate == GateType::X);
    REQUIRE(prep_e.nodes[0].targets[0].value() == 0);

    Circuit prep_g = rewritten(c, make_model(all_g(), {}), 1);
    REQUIRE(prep_g.nodes.empty());
}

TEST_CASE("rewrite: a coherent qubit that jumps to lost gets a trace-out R") {
    Circuit c = parse("H 0\nS 0\n");  // H makes qubit 0 coherent, then S leaks it
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    // H, S kept; one trace-out R appended after S.
    REQUIRE(rw.nodes.size() == 3);
    REQUIRE(rw.nodes[2].gate == GateType::R);
    REQUIRE(rw.nodes[2].targets[0].value() == 0);
}

TEST_CASE("rewrite: a source-dependent hook on a just-demoted qubit rejects") {
    // Qubit 0 enters H as Known(g), but the expanded annotation consults the
    // state at its own position -- after the H -- where the qubit is an
    // unknown superposition. A source-dependent matrix therefore rejects
    // under the default policy: the hooked gate's entry status no longer
    // selects the column.
    Circuit c = parse("H 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("H", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    REQUIRE_THROWS_WITH(rewritten(c, model, 1), ContainsSubstring("ComputationalUnknown"));
}

TEST_CASE("rewrite: a definite atom lost at a diagonal gate needs no trace-out") {
    // S preserves Known(g), so the qubit is a definite atom where the
    // annotation loses it: no entanglement to unravel, no hidden R.
    Circuit c = parse("S 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::R) == 0);
    REQUIRE(count_gate(rw, GateType::S) == 1);
}

TEST_CASE("rewrite: a still-known atom that is lost gets no trace-out R") {
    // M preserves Known(g): the qubit is a definite |0> atom at jump time, so
    // losing it needs no unraveling.
    Circuit c = parse("M 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("M", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::R) == 0);
    REQUIRE(count_gate(rw, GateType::M) == 1);  // measurement kept
}

TEST_CASE("rewrite: an inserted R survives compilation as one hidden measurement") {
    Circuit c = parse("H 0\nS 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);

    HirModule base = trace(c);
    HirModule hir = trace(rw);
    default_hir_pass_manager().run(hir);

    REQUIRE(hir.num_hidden_measurements == base.num_hidden_measurements + 1);
    REQUIRE(hir.num_measurements == base.num_measurements);  // no visible measurement added
}

TEST_CASE("rewrite: an inserted R does not shift visible measurements or detectors") {
    // Detectors reference the measurement record on both sides of the loss
    // point; the hidden trace-out R must not renumber them.
    Circuit c = parse("M 0\nDETECTOR rec[-1]\nH 1\nS 1\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);

    HirModule base = trace(c);
    HirModule hir = trace(rw);
    default_hir_pass_manager().run(hir);

    REQUIRE(hir.num_measurements == base.num_measurements);  // visible record unchanged
    REQUIRE(hir.num_detectors == base.num_detectors);
    // The detectors resolve to the same measurement-record indices: the
    // inserted hidden R renumbered nothing they point at.
    REQUIRE(hir.detector_targets == base.detector_targets);
    REQUIRE(hir.num_hidden_measurements == base.num_hidden_measurements + 1);
}

TEST_CASE("rewrite: a two-qubit gate on a lost qubit rejects by default") {
    Circuit c = parse("H 0\nS 0\nCZ 0 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);  // sampler does not enforce policy
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, model), ContainsSubstring("CZ") &&
                                                              ContainsSubstring("Lost") &&
                                                              ContainsSubstring("qubit 0"));
}

TEST_CASE("rewrite: a single-qubit gate on a leaked qubit rejects by default") {
    Circuit c = parse("H 0\nS 0\nX 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, model),
                        ContainsSubstring("Leaked") && ContainsSubstring("qubit 0"));
}

TEST_CASE("rewrite: a single-qubit gate on a lost qubit is dropped") {
    Circuit c = parse("H 0\nS 0\nX 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    // The X on the lost qubit is dropped; only the trace-out R remains.
    REQUIRE(count_gate(rw, GateType::X) == 0);
    REQUIRE(count_gate(rw, GateType::R) == 1);
}

TEST_CASE("rewrite: a lost-qubit reset rejects by default and restores under policy") {
    Circuit c = parse("H 0\nS 0\nR 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));

    NonComputationalModel reject = make_model(all_g(), transitions);
    Circuit c_ann = annotate(c, reject);
    HistorySample s = sample_history(c_ann, reject, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, reject),
                        ContainsSubstring("Lost") && ContainsSubstring("qubit 0"));

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload = make_model(all_g(), std::move(transitions), restore);
    Circuit rw = rewritten(c, reload, 1);
    // The trace-out R plus the kept original reset.
    REQUIRE(count_gate(rw, GateType::R) == 2);
}

TEST_CASE("rewrite: a lost-qubit measurement becomes a classifier record write") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(all_g(), std::move(transitions),
                                                             classifier_with(kLost, {0.5, 0.5}));

    RewriteResult rw = rewritten_result(c, model, 1);
    // The M is replaced by an MPAD padding the same visible slot, plus a
    // READOUT_NOISE drawing the stochastic classifier bit at sample time.
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
    REQUIRE(noise.targets[0].value() == 0);  // flips the padded slot
    REQUIRE(noise.args[0] == 0.5);
}

TEST_CASE("rewrite: a deterministic classifier column pads the literal bit, no draw") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(all_g(), std::move(transitions),
                                                             classifier_with(kLost, {0.0, 1.0}));

    RewriteResult rw = rewritten_result(c, model, 1);
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
    // Slot 0 is a computational measurement; the lost qubit's measurement is
    // slot 1, and its READOUT_NOISE must target slot 1.
    Circuit c = parse("M 1\nH 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(all_g(), std::move(transitions),
                                                             classifier_with(kLost, {0.5, 0.5}));

    RewriteResult rw = rewritten_result(c, model, 1);
    REQUIRE(count_gate(rw.circuit, GateType::M) == 1);  // the computational one
    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].slot == 1);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.targets[0].value() == 1);
}

TEST_CASE("rewrite: a ternary column emits the not-heralded conditional flip") {
    // Column {0.3, 0.1, 0.6}: conditioned on not heralding, the bit is
    // 0.1 / (1 - 0.6) = 0.25. A ternary slot always gets a READOUT_NOISE so
    // the herald pass has a node to re-point at one half.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(
        all_g(), std::move(transitions), classifier_with(kLeakG, {0.3, 0.1, 0.6}));

    RewriteResult rw = rewritten_result(c, model, 1);
    REQUIRE(rw.classified_measurements.size() == 1);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.args[0] == Catch::Approx(0.25));
}

TEST_CASE("rewrite: an always-herald ternary column still emits a patchable node") {
    // {0, 0, 1} has no not-heralded conditional; one half stands in, and the
    // herald pass overwrites it on every (always-heralding) draw.
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(
        all_g(), std::move(transitions), classifier_with(kLeakG, {0.0, 0.0, 1.0}));

    RewriteResult rw = rewritten_result(c, model, 1);
    REQUIRE(rw.classified_measurements.size() == 1);
    REQUIRE(rw.classified_measurements[0].noise_node != SIZE_MAX);
    const AstNode& noise = rw.circuit.nodes[rw.classified_measurements[0].noise_node];
    REQUIRE(noise.args[0] == 0.5);
}

TEST_CASE("rewrite: a noncomputational measurement without a classifier rejects") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, model),
                        ContainsSubstring("requires a classifier"));
}

TEST_CASE("rewrite: an X/Y-basis or multi-qubit measurement on a lost qubit rejects") {
    std::map<std::string, TransitionInstrument> mx;
    mx.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel m_mx = make_model(all_g(), std::move(mx));
    Circuit cx = parse("H 0\nS 0\nMX 0\n");
    Circuit cx_ann = annotate(cx, m_mx);
    HistorySample sx = sample_history(cx_ann, m_mx, 1);
    REQUIRE_THROWS_WITH(rewrite(cx_ann, sx.history, m_mx),
                        ContainsSubstring("MX") && ContainsSubstring("Lost"));

    std::map<std::string, TransitionInstrument> mp;
    mp.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel m_mp = make_model(all_g(), std::move(mp));
    Circuit cp = parse("H 0\nS 0\nMPP X0\n");
    Circuit cp_ann = annotate(cp, m_mp);
    HistorySample sp = sample_history(cp_ann, m_mp, 1);
    REQUIRE_THROWS_WITH(rewrite(cp_ann, sp.history, m_mp),
                        ContainsSubstring("MPP") && ContainsSubstring("Lost"));
}

TEST_CASE("rewrite: a measure-and-reset on a lost qubit rejects by default, kept under policy") {
    Circuit c = parse("H 0\nS 0\nMR 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));

    NonComputationalModel reject = make_model(all_g(), transitions);
    Circuit c_ann = annotate(c, reject);
    HistorySample s = sample_history(c_ann, reject, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, reject),
                        ContainsSubstring("MR") && ContainsSubstring("Lost"));

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload = make_model_with_classifier(
        all_g(), std::move(transitions), classifier_with(kLost, {1.0, 0.0}), restore);
    Circuit rw = rewritten(c, reload, 1);
    // The MR splits into the classifier record write plus its reset; with the
    // trace-out of the lost coherent qubit that makes two Rs.
    REQUIRE(count_gate(rw, GateType::MR) == 0);
    REQUIRE(count_gate(rw, GateType::MPAD) == 1);
    REQUIRE(count_gate(rw, GateType::R) == 2);
    REQUIRE(rw.num_measurements == 1);  // visible record preserved
}

TEST_CASE("rewrite: a measure-and-reset on a leaked qubit records and resets") {
    Circuit c = parse("H 0\nS 0\nMR 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model_with_classifier(all_g(), std::move(transitions),
                                                             classifier_with(kLeakG, {0.5, 0.5}));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::MR) == 0);
    REQUIRE(count_gate(rw, GateType::MPAD) == 1);
    REQUIRE(count_gate(rw, GateType::READOUT_NOISE) == 1);
    REQUIRE(count_gate(rw, GateType::R) == 2);  // trace-out plus the MR's reset
}

TEST_CASE("rewrite: a jump to the |0> computational level inserts an R, no X") {
    Circuit c = parse("H 0\nS 0\n");  // H makes qubit 0 coherent at the jump
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_to_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    // H, S kept; one materializing R appended after S, no destination-prep X.
    REQUIRE(rw.nodes.size() == 3);
    REQUIRE(rw.nodes[2].gate == GateType::R);
    REQUIRE(rw.nodes[2].targets[0].value() == 0);
    REQUIRE(count_gate(rw, GateType::X) == 0);
}

TEST_CASE("rewrite: a jump to the |1> computational level inserts an R then an X") {
    Circuit c = parse("H 0\nS 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_to_e(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(rw.nodes.size() == 4);
    REQUIRE(rw.nodes[2].gate == GateType::R);
    REQUIRE(rw.nodes[2].targets[0].value() == 0);
    REQUIRE(rw.nodes[3].gate == GateType::X);
    REQUIRE(rw.nodes[3].targets[0].value() == 0);
}

TEST_CASE("rewrite: a known carrier that relaxes is re-prepared at the destination") {
    // Qubit 0 enters S as Known(e) (initial X-prep), and the attached
    // relaxation sends it to g. Even though the carrier is definite, the
    // materializing R is inserted so the SVM state is re-prepared at |0>.
    Circuit c = parse("S 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", relax_e_to_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_e(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    // X (initial |1> prep), S, R (materialize at g).
    REQUIRE(rw.nodes.size() == 3);
    REQUIRE(rw.nodes[0].gate == GateType::X);
    REQUIRE(rw.nodes[1].gate == GateType::S);
    REQUIRE(rw.nodes[2].gate == GateType::R);
}

TEST_CASE("rewrite: a materializing R/X does not shift visible measurements or detectors") {
    Circuit c = parse("M 0\nDETECTOR rec[-1]\nH 1\nS 1\nM 0\nDETECTOR rec[-1]\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_to_e(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);

    HirModule base = trace(c);
    HirModule hir = trace(rw);
    default_hir_pass_manager().run(hir);

    REQUIRE(hir.num_measurements == base.num_measurements);  // visible record unchanged
    REQUIRE(hir.num_detectors == base.num_detectors);
    REQUIRE(hir.detector_targets == base.detector_targets);
    REQUIRE(hir.num_hidden_measurements == base.num_hidden_measurements + 1);
}

TEST_CASE("rewrite: recapturing a lost qubit rezeros its stale residual") {
    // H entangles nothing here but leaves qubit 0 coherent; S loses it (with a
    // trace-out R that may collapse to |1>), and the X's attached transition
    // recaptures it at g. The X itself is dropped (lost operand at entry) but
    // the recapture still materializes the carrier with a second R, clearing
    // whatever residual the trace-out left behind.
    Circuit c = parse("H 0\nS 0\nX 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    transitions.emplace("X", recapture_lost_to_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::X) == 0);  // base X dropped, no |1> prep
    REQUIRE(count_gate(rw, GateType::R) == 2);  // trace-out, then materialization
}

TEST_CASE("rewrite: drop policy excises a two-qubit gate on a lost operand whole") {
    Circuit c = parse("H 0\nS 0\nCZ 0 1\nM 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::CZ) == 0);  // identity on the survivor
    REQUIRE(count_gate(rw, GateType::M) == 1);   // record slot preserved
    REQUIRE(count_gate(rw, GateType::R) == 1);   // trace-out of the lost qubit
}

TEST_CASE("rewrite: a dropped gate leaves the surviving operand's status untouched") {
    // Qubit 0 is lost, so the CZ drops whole and qubit 1 keeps its
    // instruction-known g status. The later source-dependent consult on
    // qubit 1 then picks its exact column; had the dropped CZ demoted the
    // survivor to unknown, the default unknown-source policy would throw.
    Circuit c = parse("H 0\nS 0\nCZ 0 1\nS_DAG 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    transitions.emplace("S_DAG", lose_from_g(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE(s.final_status[1].kind() == clifft::QubitStatusKind::Lost);
    Circuit rw = rewrite(c_ann, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::CZ) == 0);
}

TEST_CASE("rewrite: drop policy drops a single-qubit gate on a leaked qubit") {
    Circuit c = parse("H 0\nS 0\nX 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::X) == 0);
    REQUIRE(count_gate(rw, GateType::R) == 1);  // trace-out of the leaked qubit
}

TEST_CASE("rewrite: drop policy drops classical feedback onto a lost qubit") {
    Circuit c = parse("H 1\nM 1\nH 0\nS 0\nCX rec[-1] 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::CX) == 0);
}

TEST_CASE("rewrite: drop policy drops a two-qubit noise channel on a lost operand") {
    Circuit c = parse("H 0\nS 0\nDEPOLARIZE2(0.1) 0 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::DEPOLARIZE2) == 0);
}

TEST_CASE("rewrite: drop policy drops a non-restoring lost reset") {
    Circuit c = parse("H 0\nS 0\nR 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE(s.final_status[0].kind() == clifft::QubitStatusKind::Lost);  // not restored
    Circuit rw = rewrite(c_ann, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::R) == 1);  // the trace-out only
}

TEST_CASE("rewrite: drop policy keeps a measure-and-reset on a non-restoring lost qubit") {
    Circuit c = parse("H 0\nS 0\nMR 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.lost_leaked_ops = clifft::LostLeakedOpsPolicy::Drop;
    NonComputationalModel model = make_model_with_classifier(
        all_g(), std::move(transitions), classifier_with(kLost, {1.0, 0.0}), policy);

    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE(s.final_status[0].kind() == clifft::QubitStatusKind::Lost);  // not restored
    RewriteResult rw = rewrite(c_ann, s.history, model);
    REQUIRE(count_gate(rw.circuit, GateType::MR) == 0);
    REQUIRE(count_gate(rw.circuit, GateType::MPAD) == 1);  // record slot preserved
    REQUIRE(rw.circuit.num_measurements == 1);
}

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
    NonComputationalModel model =
        make_model_with_classifier(all_g(), {}, confused_classifier(0.1, 0.2));

    RewriteResult rw = rewritten_result(c, model, 1);
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
        make_model_with_classifier(all_g(), {}, classifier_with(kLost, {0.5, 0.5}));

    RewriteResult rw = rewritten_result(c, model, 1);
    REQUIRE(count_gate(rw.circuit, GateType::READOUT_NOISE) == 0);
}

TEST_CASE("rewrite: an inverted measurement swaps the confusion probabilities") {
    // Confusion physically precedes the reporting convention, and
    // invert-after-flip(p01, p10) equals flip(p10, p01)-after-invert.
    Circuit c = parse("H 0\nM !0\n");
    NonComputationalModel model =
        make_model_with_classifier(all_g(), {}, confused_classifier(0.1, 0.2));

    RewriteResult rw = rewritten_result(c, model, 1);
    const AstNode& noise = rw.circuit.nodes.back();
    REQUIRE(noise.gate == GateType::READOUT_NOISE);
    REQUIRE(noise.args[0] == 0.2);
    REQUIRE(noise.args[1] == 0.1);
}

TEST_CASE("rewrite: confusion applies to MR but not X-basis measurements") {
    Circuit c = parse("MR 0\nMX 0\n");
    NonComputationalModel model =
        make_model_with_classifier(all_g(), {}, confused_classifier(0.1, 0.2));

    RewriteResult rw = rewritten_result(c, model, 1);
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
    NonComputationalModel model = make_model_with_classifier(all_g(), {}, std::move(cl));

    Circuit c = parse("M 0\n");
    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, model), ContainsSubstring("computational level"));
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
    NonComputationalModel model = make_model_with_classifier(all_g(), {}, std::move(cl));

    Circuit c = parse("M 0\n");
    Circuit c_ann = annotate(c, model);
    HistorySample s = sample_history(c_ann, model, 1);
    REQUIRE_THROWS_WITH(rewrite(c_ann, s.history, model), ContainsSubstring("beyond the bit"));
}

TEST_CASE("annotate: gate hooks expand to per-operand LEVEL_TRANSITION annotations") {
    Circuit c = parse("H 0\nCZ 0 1\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

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
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit ann = annotate(c, model);
    REQUIRE(ann.nodes.size() == c.nodes.size());  // virtual correction: no consult point
}

TEST_CASE("annotate: an unhooked model leaves the circuit unchanged") {
    Circuit c = parse("H 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("my_leak", always_lost(LevelSet::default_set()));  // named, no hook
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit ann = annotate(c, model);
    REQUIRE(ann.nodes.size() == c.nodes.size());
}

TEST_CASE("rewrite: annotations are consumed, not emitted") {
    Circuit c = parse("S 0\nLEVEL_TRANSITION[jump] 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("jump", never_fires(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    Circuit rw = rewrite(c, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::LEVEL_TRANSITION) == 0);
    REQUIRE(count_gate(rw, GateType::S) == 1);
    REQUIRE(count_gate(rw, GateType::M) == 1);
}

TEST_CASE("rewrite: a LOSS jump on a coherent carrier inserts the trace-out R") {
    Circuit c = parse("H 0\nLOSS(1) 0\n");
    NonComputationalModel model = make_model(all_g(), {});

    HistorySample s = sample_history(c, model, 1);
    Circuit rw = rewrite(c, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::LOSS) == 0);  // consumed
    REQUIRE(count_gate(rw, GateType::R) == 1);     // coherent carrier traced out
}

TEST_CASE("rewrite: a LOSS jump on a definite atom needs no trace-out") {
    Circuit c = parse("LOSS(1) 0\n");  // Known(g) at the annotation
    NonComputationalModel model = make_model(all_g(), {});

    HistorySample s = sample_history(c, model, 1);
    Circuit rw = rewrite(c, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::R) == 0);
    REQUIRE(s.final_status[0].kind() == clifft::QubitStatusKind::Lost);
}

TEST_CASE("rewrite: a LEVEL_TRANSITION jump to a computational level materializes the carrier") {
    // Recapture a lost qubit at e: R rezeros the stale carrier, X prepares |1>.
    Circuit c = parse("LOSS(1) 0\nLEVEL_TRANSITION[recapture] 0\n");
    auto m = zeros5();
    m[1][4] = 1.0;  // lost -> e with certainty
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("recapture",
                        TransitionInstrument::from_matrix(std::move(m), LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    REQUIRE(s.final_status[0].kind() == clifft::QubitStatusKind::ComputationalKnown);
    Circuit rw = rewrite(c, s.history, model).circuit;
    REQUIRE(count_gate(rw, GateType::R) == 1);
    REQUIRE(count_gate(rw, GateType::X) == 1);
}
