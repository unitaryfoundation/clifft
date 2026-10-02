#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/interaction.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sample.h"

#include "noncomp_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {
NonComputationalModel interaction_model() {
    return NonComputationalModel::from_spec(
        test::pure_initial_state(Level::G), {},
        test::classifier_matrix_with_column(Level::LeakE, {0, 1}), {});
}
}  // namespace

TEST_CASE("Interaction annotations preserve ordered pairs without adding records") {
    const auto circuit = parse(
        "LEAKAGE_INTERACTION(0.1, 0.2, 0.3, 0.4) 5 2 2 7\n"
        "LOSS_INTERACTION(0, 0, 1) 7 5");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].targets[0].value() == 5);
    CHECK(circuit.nodes[0].targets[1].value() == 2);
    CHECK(circuit.nodes[1].targets[0].value() == 2);
    CHECK(circuit.nodes[1].targets[1].value() == 7);
    CHECK(circuit.num_measurements == 0);
    CHECK(circuit.num_qubits == 8);
    CHECK_THROWS_WITH(trace(circuit), ContainsSubstring("clifft.noncomp.sample"));
    const auto options = instrument_trace_options(interaction_model());
    CHECK_THROWS_WITH(trace(circuit, &options), ContainsSubstring("must be lowered"));
}

TEST_CASE("Interaction annotation arguments and targets reject malformed inputs") {
    for (const std::string instruction :
         {"LEAKAGE_INTERACTION(0, 0, 0)", "LOSS_INTERACTION(0, 0, 0, 0)",
          "LEAKAGE_INTERACTION(0.5, 0.5, 0.1, 0)", "LEAKAGE_INTERACTION(0, 0, 0, -0.1)",
          "LEAKAGE_INTERACTION(0, 0, 0, 1.1)", "LEAKAGE_INTERACTION(nan, 0, 0, 0)",
          "LEAKAGE_INTERACTION(0, 0, 0, inf)", "LOSS_INTERACTION(0, -0.1, 0)"}) {
        CAPTURE(instruction);
        CHECK_THROWS_AS(parse(instruction + " 0 1"), ParseError);
    }
    for (const std::string targets : {"", "0", "0 0", "0 1 2", "!0 1", "X0 1", "rec[-1] 1"}) {
        CAPTURE(targets);
        CHECK_THROWS_AS(parse("M 0\nLEAKAGE_INTERACTION(0, 0, 0, 0) " + targets), ParseError);
    }
    auto circuit = parse("LEAKAGE_INTERACTION(1, 0, 0, 1) 0 1");
    for (const auto& targets :
         std::vector<std::vector<Target>>{{},
                                          {Target::qubit(0)},
                                          {Target::qubit(0), Target::qubit(2)},
                                          {Target::qubit(0), Target::qubit(0)},
                                          {Target::rec(0), Target::qubit(1)},
                                          {Target::qubit(0).inverted(), Target::qubit(1)},
                                          {Target::pauli(0, Target::kPauliX), Target::qubit(1)}}) {
        circuit.nodes[0].targets = targets;
        CHECK_THROWS_WITH(sample_noncomputational(circuit, interaction_model(), 0, 1),
                          ContainsSubstring("at op 0"));
    }
}

TEST_CASE(
    "Interaction conditions distinguish both leaked levels from loss and require a live partner") {
    for (const auto source :
         {QubitStatus::Computational, QubitStatus::LeakG, QubitStatus::LeakE, QubitStatus::Lost}) {
        for (const auto partner : {QubitStatus::Computational, QubitStatus::LeakG,
                                   QubitStatus::LeakE, QubitStatus::Lost}) {
            for (const std::string name : {"LEAKAGE_INTERACTION", "LOSS_INTERACTION"}) {
                const bool leakage = name == "LEAKAGE_INTERACTION";
                const auto circuit =
                    parse(name + (leakage ? "(1, 0, 0, 0)" : "(1, 0, 0)") + " 0 1");
                TrajectoryEvents events;
                events.initial_status = {source, partner};
                const auto result =
                    rewrite_continuation(circuit, events, false, interaction_model());
                const bool expected =
                    is_computational(partner) && (leakage ? is_leaked(source) : is_lost(source));
                CAPTURE(name, source, partner);
                CHECK(result.circuit.nodes.size() == (expected ? 1 : 0));
                CHECK(result.final_status == events.initial_status);
            }
        }
    }
}

TEST_CASE("Interaction spreading preserves the Pauli prefix and original transition identity") {
    const auto circuit = parse(
        "LEAKAGE_INTERACTION(1, 0, 0, 1) 0 1\n"
        "LEAKAGE_INTERACTION(0, 0, 0, 1) 1 2");
    TrajectoryEvents events;
    events.initial_status = {QubitStatus::LeakG, QubitStatus::Computational,
                             QubitStatus::Computational};
    const auto before = rewrite_continuation(circuit, events, false, interaction_model());
    REQUIRE(before.circuit.nodes.size() == 2);
    CHECK(before.circuit.nodes[0].gate == GateType::PAULI_CHANNEL_1);
    CHECK(before.circuit.nodes[1].gate == GateType::LEAKAGE);
    REQUIRE(before.site_targets.size() == 1);
    CHECK(before.site_targets[0] == AnnotationTarget{0, 1});

    events.jumps.push_back({{0, 1}, Level::LeakE});
    const auto after = rewrite_continuation(circuit, events, true, interaction_model());
    REQUIRE(after.circuit.nodes.size() == 4);
    CHECK(after.circuit.nodes[0].gate == GateType::PAULI_CHANNEL_1);
    CHECK(after.circuit.nodes[0].args == before.circuit.nodes[0].args);
    CHECK(after.circuit.nodes[1].gate == GateType::LEAKAGE);
    CHECK(after.circuit.nodes[2].gate == GateType::R);
    CHECK(after.circuit.nodes[3].gate == GateType::LEAKAGE);
    CHECK(after.forced_traceout_node == 2);
    REQUIRE(after.site_targets.size() == 2);
    CHECK(after.site_targets[1] == AnnotationTarget{1, 2});
}

TEST_CASE(
    "Interaction trajectories apply Pauli noise before spreading through successive partners") {
    const auto circuit = parse(
        "LEAKAGE(1) 0\n"
        "LEAKAGE_INTERACTION(1, 0, 0, 1) 0 1\n"
        "LEAKAGE_INTERACTION(0, 0, 0, 1) 1 2\n"
        "HERALD_LEAKAGE_EVENT 0 1 2\nM 0 1 2");
    const auto result = sample_noncomputational(circuit, interaction_model(), 32, 19);
    REQUIRE(result.num_measurements == 6);
    for (size_t shot = 0; shot < 32; ++shot) {
        CHECK(result.measurements[shot * 6] == 1);
        CHECK(result.measurements[shot * 6 + 1] == 1);
        CHECK(result.measurements[shot * 6 + 2] == 1);
        CHECK(result.measurements[shot * 6 + 3] == 0);
        CHECK(result.measurements[shot * 6 + 4] == 1);
        CHECK(result.measurements[shot * 6 + 5] == 0);
        CHECK(result.final_status[shot * 3 + 1] == QubitStatus::LeakE);
    }
}
