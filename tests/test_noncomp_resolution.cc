#include "clifft/circuit/parser.h"
#include "clifft/noncomp/preparation.h"
#include "clifft/noncomp/rewriter.h"

#include "noncomp_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <variant>

using namespace clifft;

TEST_CASE("Noncomputational resolution preserves generated and explicit operation identities") {
    const auto model = NonComputationalModel::from_spec(
        test::pure_initial_state(Level::G),
        {{"CX", test::certain_transition_from_computational(Level::LeakG)},
         {"other", test::certain_transition_from_computational(Level::Lost)}},
        std::nullopt, {}, {PartnerEffect{{0.25, 0.25, 0.25}, 0.1}, PartnerEffect{{0, 0, 1}, 0}},
        {{"CNOT", 0, InteractionSource::Leaked, {}}});
    const PreparedNonComputationalRequest request(
        parse("LEAKAGE(0.25) 0\nCX 2 5\nLOSS_INTERACTION(1, 0, 0) 2 5\n"
              "LEVEL_TRANSITION[other] 2\nLEVEL_TRANSITION[CX] 5\nHERALD_LOSS_EVENT 5"),
        model);
    const auto& resolved = request.resolved_circuit();
    const auto& circuit = resolved.circuit();
    REQUIRE(circuit.nodes.size() == 11);
    CHECK(circuit.num_measurements == 1);

    // Hooks share the physical gate's source line and precede the next input
    // operation. Equal-looking explicit annotations must not acquire a parent.
    for (uint32_t i = 2; i <= 6; ++i) {
        CHECK(resolved.operation(i).generated_by == 1);
        CHECK(circuit.nodes[i].source_line == 2);
    }
    for (const uint32_t i : {0u, 1u, 7u, 8u, 9u, 10u}) {
        CHECK_FALSE(resolved.operation(i).generated_by.has_value());
    }
    CHECK(circuit.nodes[7].source_line == 3);
    const auto& lost = std::get<ResolvedInteraction>(resolved.operation(2).annotation);
    CHECK(lost.source == 2);
    CHECK(lost.partner == 5);
    CHECK(lost.source_status == InteractionSource::Lost);
    CHECK(lost.effect.pauli == std::array<double, 3>{0, 0, 1});
    const auto& leaked = std::get<ResolvedInteraction>(resolved.operation(3).annotation);
    CHECK(leaked.source == 5);
    CHECK(leaked.partner == 2);
    CHECK(leaked.source_status == InteractionSource::Leaked);
    CHECK(std::get<LeakageTransition>(resolved.transition(3)).probability == 0.1);
    CHECK(std::get<LeakageTransition>(resolved.transition(0)).probability == 0.25);

    const auto hooked = std::get<TransitionId>(resolved.transition(5));
    CHECK(std::get<TransitionId>(resolved.transition(6)) == hooked);
    CHECK(std::get<TransitionId>(resolved.transition(9)) == hooked);
    CHECK(std::get<TransitionId>(resolved.transition(8)) != hooked);
    CHECK(resolved.instrument(hooked).prob(Level::LeakG, Level::E) == 1);
    CHECK(resolved.instrument(std::get<TransitionId>(resolved.transition(8)))
              .prob(Level::Lost, Level::G) == 1);
    const auto& probe = std::get<ResolvedStatusProbe>(resolved.operation(10).annotation);
    CHECK(probe.qubit == 5);
    CHECK(probe.source_status == InteractionSource::Lost);
}

TEST_CASE("Resolved channels own their values after inputs and prior copies are destroyed") {
    const auto resolved = [] {
        auto model = NonComputationalModel::from_spec(
            test::pure_initial_state(Level::G),
            {{"leak", test::certain_transition_from_computational(Level::LeakE)}}, std::nullopt,
            {});
        auto circuit = parse("LEVEL_TRANSITION[leak] 0\nLEAKAGE_INTERACTION(1, 0, 0, 1) 0 1");
        ResolvedNonComputationalCircuit original(circuit, model, "test");
        auto copy = original;
        circuit.nodes.clear();
        model = NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                                 std::nullopt, {});
        return ResolvedNonComputationalCircuit(std::move(copy));
    }();
    CHECK(resolved.instrument(std::get<TransitionId>(resolved.transition(0)))
              .prob(Level::LeakE, Level::G) == 1);

    // Resolution owns the transition; rewriting only needs the remaining
    // classifier and policy from this model, which deliberately has no tags.
    const auto policy_model =
        NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, std::nullopt, {});
    TrajectoryEvents events;
    events.initial_status = {QubitStatus::Computational, QubitStatus::Computational};
    events.jumps.push_back({{0, 0}, Level::LeakE});
    const auto rewrite = rewrite_continuation(resolved, events, true, policy_model);
    REQUIRE(rewrite.circuit.nodes.size() == 4);
    CHECK(rewrite.circuit.nodes[0].gate == GateType::LEVEL_TRANSITION);
    CHECK(rewrite.circuit.nodes[1].gate == GateType::R);
    CHECK(rewrite.circuit.nodes[2].gate == GateType::PAULI_CHANNEL_1);
    CHECK(rewrite.circuit.nodes[2].args == std::vector<double>{1, 0, 0});
    CHECK(rewrite.circuit.nodes[3].gate == GateType::LEAKAGE);
    CHECK(rewrite.site_targets == std::vector<AnnotationTarget>{{0, 0}, {1, 1}});
    CHECK(rewrite.forced_traceout_node == 1);
}

TEST_CASE("Resolved partner effects preserve conditions across every source and partner status") {
    const auto model =
        NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, std::nullopt, {});
    const ResolvedNonComputationalCircuit resolved(
        parse("LEAKAGE_INTERACTION(1, 0, 0, 0) 0 1\nLOSS_INTERACTION(0, 1, 0) 0 1"), model, "test");
    for (const auto source :
         {QubitStatus::Computational, QubitStatus::LeakG, QubitStatus::LeakE, QubitStatus::Lost}) {
        for (const auto partner : {QubitStatus::Computational, QubitStatus::LeakG,
                                   QubitStatus::LeakE, QubitStatus::Lost}) {
            CAPTURE(source, partner);
            TrajectoryEvents events;
            events.initial_status = {source, partner};
            const auto rewrite = rewrite_continuation(resolved, events, false, model);
            const bool expected = !is_computational(source) && is_computational(partner);
            REQUIRE(rewrite.circuit.nodes.size() == (expected ? 1 : 0));
            CHECK(rewrite.final_status == events.initial_status);
            if (expected) {
                CHECK(rewrite.circuit.nodes.front().args == (is_leaked(source)
                                                                 ? std::vector<double>{1, 0, 0}
                                                                 : std::vector<double>{0, 1, 0}));
            }
        }
    }
}
