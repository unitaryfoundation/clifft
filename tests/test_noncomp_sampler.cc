#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
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
using clifft::parse;
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

// Source-dependent: g jumps to lost with probability 0.3. T[lost][g] is
// the g-source -> lost-destination entry, so this also pins orientation.
TransitionInstrument lose_from_g_30pct(const LevelSet& levels) {
    auto m = zeros5();
    m[kLost][0] = 0.3;
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
    // One record per operand, in target order.
    REQUIRE(s.history.transitions.size() == 2);
    REQUIRE(s.history.transitions[0].op_index == 0);
    REQUIRE(s.history.transitions[0].qubit == 0);
    REQUIRE(s.history.transitions[1].op_index == 0);
    REQUIRE(s.history.transitions[1].qubit == 1);
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::Lost);
    REQUIRE(s.final_status[1].kind() == QubitStatusKind::Lost);
}

TEST_CASE("sample_history: classical feedback fires no transition and demotes the target") {
    // Parsed so the rec-target encoding is exercised: M then a conditional
    // X on qubit 1 controlled by the measurement record.
    Circuit c = parse("M 0\nCX rec[-1] 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CX", lose_from_g(LevelSet::default_set()));  // would jump g->lost
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    // The CX is virtual feedback, so no transition is consulted; qubit 1 is
    // demoted, not lost. (M has no transition; qubit 0 stays Known(g).)
    REQUIRE(s.history.transitions.empty());
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(s.final_status[1].kind() == QubitStatusKind::ComputationalUnknown);
}

TEST_CASE("sample_history: conditional-Z feedback preserves a known target and fires nothing") {
    Circuit c = parse("M 0\nCZ rec[-1] 1\n");
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", lose_from_g(LevelSet::default_set()));  // would jump g->lost
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);
    // Conditional Z is phase-only: qubit 1 stays Known(g), and no
    // transition is consulted on the virtual correction.
    REQUIRE(s.history.transitions.empty());
    REQUIRE(s.final_status[1].kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(s.final_status[1].level_id() == 0);  // g
}

TEST_CASE("sample_history: MPP demotes each measured known qubit to Unknown") {
    Circuit c = parse("MPP X0*Z1");  // multi-qubit measurement on two Known(g) qubits
    NonComputationalModel model = make_model(all_g(), {});  // no MPP transition attached

    HistorySample s = sample_history(c, model, 1);
    // A multi-qubit measurement collapses each operand's definite level, so
    // both qubits demote; with no instrument attached, nothing fires.
    REQUIRE(s.history.transitions.empty());
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::ComputationalUnknown);
    REQUIRE(s.final_status[1].kind() == QubitStatusKind::ComputationalUnknown);
}

TEST_CASE("sample_history: M on Unknown then a source-dependent transition rejects") {
    Circuit c = parse("H 0\nM 0\nCZ 0 1\n");  // H -> Unknown; M keeps Unknown
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    REQUIRE_THROWS_WITH(sample_history(c, model, 1),
                        ContainsSubstring("CZ") && ContainsSubstring("ComputationalUnknown"));
}

TEST_CASE("sample_history: reset before a source-dependent transition is allowed") {
    Circuit c = parse("H 0\nR 0\nCZ 0 1\n");  // R restores qubit 0 to Known(g)
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("CZ", lose_from_g(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 1);               // no throw: source is Known(g)
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::Lost);  // g -> lost
}

TEST_CASE("sample_history: a partial jump probability matches its frequency") {
    // 2000 independent qubits, each a single H carrying a g->lost(0.3)
    // transition; all start Known(g). Tests probabilistic sampling and the
    // T[to, from] orientation (a transposed matrix would never fire).
    Circuit c;
    c.num_qubits = 2000;
    for (uint32_t q = 0; q < c.num_qubits; ++q) {
        c.nodes.push_back(op(GateType::H, {q}));
    }
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("H", lose_from_g_30pct(LevelSet::default_set()));
    NonComputationalModel model = make_model(all_g(), std::move(transitions));

    HistorySample s = sample_history(c, model, 99);
    size_t lost = 0;
    for (const auto& status : s.final_status) {
        if (status.kind() == QubitStatusKind::Lost) {
            ++lost;
        }
    }
    // Expected 600; +/-120 is ~7 sigma, so this never flakes for the seed.
    REQUIRE(lost > 480);
    REQUIRE(lost < 720);
}

TEST_CASE("sample_history: lost-qubit reset restoration is policy-gated") {
    Circuit c = parse("R 0\n");
    const std::vector<double> all_lost = {0.0, 0.0, 0.0, 0.0, 1.0};

    // Default: a lost qubit's reset does not restore it.
    NonComputationalModel keep = make_model(all_lost, {});
    REQUIRE(sample_history(c, keep, 1).final_status[0].kind() == QubitStatusKind::Lost);

    // With the policy set, the reset restores it to Known(g).
    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    NonComputationalModel reload = make_model(all_lost, {}, restore);
    HistorySample s = sample_history(c, reload, 1);
    REQUIRE(s.final_status[0].kind() == QubitStatusKind::ComputationalKnown);
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

TEST_CASE("sample_history: equalize_rates samples an unknown source instead of throwing") {
    // 200 independent H'd qubits each consult a source-dependent instrument
    // (g jumps to lost with certainty, e never). Equalized, every consult
    // fires at p_max = 1; the uniform source draw sends about half to lost
    // (the g column) and half to a pseudo-jump landing as a known
    // computational e (the padded e diagonal).
    constexpr uint32_t kN = 200;
    Circuit c;
    c.num_qubits = kN;
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::H, {q}));
    }
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::S, {q}));
    }
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", lose_from_g(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.unknown_source_policy = clifft::UnknownSourcePolicy::EqualizeRates;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    HistorySample s = sample_history(c, model, 7);

    size_t lost = 0;
    size_t known_e = 0;
    for (const auto& st : s.final_status) {
        if (st.kind() == QubitStatusKind::Lost) {
            ++lost;
        } else if (st.kind() == QubitStatusKind::ComputationalKnown && st.level_id() == kE) {
            ++known_e;
        }
    }
    REQUIRE(lost + known_e == kN);  // p_max = 1: every consult fired
    REQUIRE(lost > kN * 35 / 100);
    REQUIRE(lost < kN * 65 / 100);
    REQUIRE(s.history.transitions.size() == kN);
    for (const auto& rec : s.history.transitions) {
        REQUIRE(rec.jumped);
        REQUIRE((rec.destination_level == kE || rec.destination_level == kLost));
    }
}

TEST_CASE("sample_history: equalize_rates fires at the maximum computational rate") {
    // g jumps to lost with probability 0.3 and e never, so p_max = 0.3.
    // About 30% of unknown-source consults fire; fired consults split evenly
    // between the lost destination (g column) and the pseudo-jump onto e.
    constexpr uint32_t kN = 1000;
    Circuit c;
    c.num_qubits = kN;
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::H, {q}));
    }
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::S, {q}));
    }
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S", lose_from_g_30pct(LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.unknown_source_policy = clifft::UnknownSourcePolicy::EqualizeRates;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    HistorySample s = sample_history(c, model, 11);

    size_t fired = 0;
    for (const auto& rec : s.history.transitions) {
        if (rec.jumped) {
            ++fired;
        }
    }
    size_t lost = 0;
    for (const auto& st : s.final_status) {
        if (st.kind() == QubitStatusKind::Lost) {
            ++lost;
        }
    }
    REQUIRE(fired > 230);
    REQUIRE(fired < 370);
    REQUIRE(lost > 90);
    REQUIRE(lost < 210);
}

TEST_CASE("sample_history: equalize_rates keeps a known source exact") {
    // The policy only governs unknown sources: a known g qubit keeps its
    // exact column, which here jumps to lost with certainty.
    Circuit c;
    c.num_qubits = 1;
    c.nodes.push_back(op(GateType::S, {0}));
    for (uint64_t seed = 0; seed < 10; ++seed) {
        std::map<std::string, TransitionInstrument> transitions;
        transitions.emplace("S", lose_from_g(LevelSet::default_set()));
        NonComputationalPolicy policy;
        policy.unknown_source_policy = clifft::UnknownSourcePolicy::EqualizeRates;
        NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);
        HistorySample s = sample_history(c, model, seed);
        REQUIRE(s.final_status[0].kind() == QubitStatusKind::Lost);
    }
}

TEST_CASE("sample_history: equalize_rates known divergence on a gate-determined state") {
    // H then H returns each qubit to |g> deterministically, but status
    // tracking is instruction-known, so the qubit is ComputationalUnknown at
    // the consult. The exact channel (leaking only from e) could never fire
    // on |g>; the equalized draw still fires at p_max = 1 and the uniform
    // source draw loses about half the qubits. This pins the documented
    // approximation boundary: if status tracking ever starts promoting
    // gate-determined states, these expectations must change with it.
    constexpr uint32_t kN = 200;
    Circuit c;
    c.num_qubits = kN;
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::H, {q}));
    }
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::H, {q}));
    }
    for (uint32_t q = 0; q < kN; ++q) {
        c.nodes.push_back(op(GateType::S, {q}));
    }
    auto m = zeros5();
    m[kLost][kE] = 1.0;  // leaks only from e: exact on |g> would never fire
    std::map<std::string, TransitionInstrument> transitions;
    transitions.emplace("S",
                        TransitionInstrument::from_matrix(std::move(m), LevelSet::default_set()));
    NonComputationalPolicy policy;
    policy.unknown_source_policy = clifft::UnknownSourcePolicy::EqualizeRates;
    NonComputationalModel model = make_model(all_g(), std::move(transitions), policy);

    HistorySample s = sample_history(c, model, 13);

    size_t lost = 0;
    for (const auto& st : s.final_status) {
        if (st.kind() == QubitStatusKind::Lost) {
            ++lost;
        }
    }
    REQUIRE(lost > kN * 35 / 100);
    REQUIRE(lost < kN * 65 / 100);
}
