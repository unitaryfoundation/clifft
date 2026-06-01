#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"
#include "clifft/noncomp/sampler.h"
#include "clifft/noncomp/transition_instrument.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::AstNode;
using clifft::Circuit;
using clifft::GateType;
using clifft::HistorySample;
using clifft::LevelSet;
using clifft::NonComputationalModel;
using clifft::NonComputationalPolicy;
using clifft::QubitStatusKind;
using clifft::sample_history;
using clifft::Target;
using clifft::TransitionInstrument;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kE = 1;
constexpr uint8_t kLost = 4;

std::vector<std::vector<double>> zeros5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

// Source-dependent: g certainly jumps to lost, e does nothing. The g and
// e columns differ, so is_source_independent_on_computational() is false.
TransitionInstrument lose_from_g(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 1.0;
    return TransitionInstrument::from_matrix(std::move(m), levels);
}

// Source-independent: never jumps (all-zero columns are equal).
TransitionInstrument never_jumps(const LevelSet& levels) {
    return TransitionInstrument::from_matrix(zeros5(), levels);
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

AstNode op(GateType gate, std::vector<uint32_t> qubits) {
    std::vector<Target> targets;
    targets.reserve(qubits.size());
    for (uint32_t q : qubits) {
        targets.push_back(Target::qubit(q));
    }
    return AstNode{gate, std::move(targets), {}, 0};
}

}  // namespace

TEST_CASE("sample_history: a fixed seed produces an identical history") {
    Circuit c;
    c.num_qubits = 64;  // no ops; exercises initial-state sampling
    NonComputationalModel model = make_model({0.5, 0.5, 0.0, 0.0, 0.0}, {});

    HistorySample a = sample_history(c, model, 777);
    HistorySample b = sample_history(c, model, 777);

    REQUIRE(a.history.initial_status.size() == b.history.initial_status.size());
    for (size_t i = 0; i < a.history.initial_status.size(); ++i) {
        REQUIRE(a.history.initial_status[i].kind() == b.history.initial_status[i].kind());
        REQUIRE(a.history.initial_status[i].level_id() == b.history.initial_status[i].level_id());
    }
}

TEST_CASE("sample_history: initial-state marginals match the distribution") {
    Circuit c;
    c.num_qubits = 2000;  // no ops
    NonComputationalModel model = make_model({0.7, 0.3, 0.0, 0.0, 0.0}, {});

    HistorySample s = sample_history(c, model, 12345);
    size_t e_count = 0;
    for (const auto& status : s.history.initial_status) {
        if (status.kind() == QubitStatusKind::ComputationalKnown && status.level_id() == kE) {
            ++e_count;
        }
    }
    // Expected 600; a generous +/-120 band is ~6 sigma, so this never
    // flakes for the fixed seed.
    REQUIRE(e_count > 480);
    REQUIRE(e_count < 720);
}

TEST_CASE("sample_history: a transition on a known source fires and updates the status") {
    Circuit c;
    c.num_qubits = 1;
    c.nodes.push_back(op(GateType::H, {0}));
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("H", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    // g jumps to lost with certainty, so the jump destination wins over
    // H's normal demotion.
    REQUIRE(s.history.transitions.size() == 1);
    REQUIRE(s.history.transitions[0].op_index == 0);
    REQUIRE(s.history.transitions[0].qubit == 0);
    REQUIRE(s.history.transitions[0].jumped);
    REQUIRE(s.history.transitions[0].destination_level == kLost);
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::Lost);
}

TEST_CASE("sample_history: source-dependent transition on a known qubit is allowed") {
    Circuit c;
    c.num_qubits = 2;
    c.nodes.push_back(op(GateType::CZ, {0, 1}));  // both operands start Known(g)
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    REQUIRE(s.history.transitions.size() == 2);
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::Lost);
    REQUIRE(s.final_status[1].kind() == QubitStatusKind::Lost);
}

TEST_CASE("sample_history: source-dependent transition on an unknown qubit rejects") {
    Circuit c;
    c.num_qubits = 2;
    c.nodes.push_back(op(GateType::H, {0}));      // demotes qubit 0 to Unknown
    c.nodes.push_back(op(GateType::CZ, {0, 1}));  // source-dependent transition here
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    REQUIRE_THROWS_WITH(sample_history(c, model, 1),
                        ContainsSubstring("CZ") && ContainsSubstring("qubit 0") &&
                            ContainsSubstring("op 1") && ContainsSubstring("ComputationalUnknown"));
}

TEST_CASE("sample_history: source-independent transition on an unknown qubit is allowed") {
    Circuit c;
    c.num_qubits = 1;
    c.nodes.push_back(op(GateType::H, {0}));  // Known(g) -> Unknown (no jump)
    c.nodes.push_back(op(GateType::H, {0}));  // fires on the Unknown qubit
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("H", never_jumps(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    REQUIRE(s.history.transitions.size() == 2);
    REQUIRE_FALSE(s.history.transitions[0].jumped);
    REQUIRE_FALSE(s.history.transitions[1].jumped);
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::ComputationalUnknown);
}
