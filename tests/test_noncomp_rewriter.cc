#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/transition_instrument.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <map>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::Circuit;
using clifft::default_hir_pass_manager;
using clifft::GateType;
using clifft::HirModule;
using clifft::HistorySample;
using clifft::LevelSet;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::parse;
using clifft::rewrite;
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

NonComputationalModel make_model(std::vector<double> initial_state,
                                 std::map<std::string, TransitionInstrument> transitions,
                                 NonComputationalPolicy policy = {}) {
    return NonComputationalModel(LevelSet::default_set(), std::move(initial_state),
                                 std::move(transitions), std::nullopt, policy);
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

Circuit rewritten(const Circuit& c, const NonComputationalModel& model, uint64_t seed) {
    HistorySample s = sample_history(c, model, seed);
    return rewrite(c, s.history, model);
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

TEST_CASE("rewrite: an op that demotes a known qubit before a jump still inserts the R") {
    // Qubit 0 enters H as Known(g); H demotes it to coherent Unknown and the
    // attached transition then leaks it. The carrier is coherent at jump time,
    // so a trace-out R is required even though the entry status was Known.
    Circuit c = parse("H 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("H", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(rw.nodes.size() == 2);
    REQUIRE(rw.nodes[1].gate == GateType::R);
    REQUIRE(rw.nodes[1].targets[0].value() == 0);
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
    REQUIRE(hir.num_hidden_measurements == base.num_hidden_measurements + 1);
}

TEST_CASE("rewrite: a two-qubit gate on a lost qubit rejects by default") {
    Circuit c = parse("H 0\nS 0\nCZ 0 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);  // sampler does not enforce policy
    REQUIRE_THROWS_WITH(rewrite(c, s.history, model), ContainsSubstring("CZ") &&
                                                          ContainsSubstring("Lost") &&
                                                          ContainsSubstring("qubit 0"));
}

TEST_CASE("rewrite: a single-qubit gate on a leaked qubit rejects by default") {
    Circuit c = parse("H 0\nS 0\nX 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_leaked(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    REQUIRE_THROWS_WITH(rewrite(c, s.history, model),
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
    HistorySample s = sample_history(c, reject, 1);
    REQUIRE_THROWS_WITH(rewrite(c, s.history, reject),
                        ContainsSubstring("Lost") && ContainsSubstring("qubit 0"));

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload = make_model(all_g(), std::move(transitions), restore);
    Circuit rw = rewritten(c, reload, 1);
    // The trace-out R plus the kept original reset.
    REQUIRE(count_gate(rw, GateType::R) == 2);
}

TEST_CASE("rewrite: a measurement on a lost qubit is kept") {
    Circuit c = parse("H 0\nS 0\nM 0\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", always_lost(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    Circuit rw = rewritten(c, model, 1);
    REQUIRE(count_gate(rw, GateType::M) == 1);
    REQUIRE(rw.num_measurements == 1);  // visible record preserved
}
