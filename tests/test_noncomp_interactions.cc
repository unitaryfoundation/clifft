#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/interaction.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/rewriter.h"
#include "clifft/noncomp/sample.h"
#include "clifft/noncomp/transition_hooks.h"

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

TEST_CASE("Interaction defaults expand before transition hooks and explicit annotations") {
    const PartnerEffect depolarizing{{0.25, 0.25, 0.25}, 0.1};
    const auto model = NonComputationalModel::from_spec(
        test::pure_initial_state(Level::G),
        {{"CX", test::certain_transition_from_computational(Level::LeakG)}}, std::nullopt, {},
        {depolarizing, PartnerEffect{{0, 0, 1}, 0}}, {{"CNOT", 0, InteractionSource::Leaked, {}}});
    const auto expanded =
        expand_transition_hooks(parse("CX 2 5\nLOSS_INTERACTION(1, 0, 0) 2 5"), model);
    const auto expected = parse(
        "CX 2 5\n"
        "LOSS_INTERACTION(0, 0, 1) 2 5\n"
        "LEAKAGE_INTERACTION(0.25, 0.25, 0.25, 0.1) 5 2\n"
        "LOSS_INTERACTION(0, 0, 1) 5 2\n"
        "LEVEL_TRANSITION[CX] 2 5\n"
        "LOSS_INTERACTION(1, 0, 0) 2 5");
    REQUIRE(expanded.nodes.size() == expected.nodes.size());
    for (size_t i = 0; i < expanded.nodes.size(); ++i) {
        CAPTURE(i);
        CHECK(expanded.nodes[i].gate == expected.nodes[i].gate);
        CHECK(expanded.nodes[i].args == expected.nodes[i].args);
        CHECK(expanded.nodes[i].tag == expected.nodes[i].tag);
        REQUIRE(expanded.nodes[i].targets.size() == expected.nodes[i].targets.size());
        for (size_t j = 0; j < expanded.nodes[i].targets.size(); ++j) {
            CHECK(expanded.nodes[i].targets[j].value() == expected.nodes[i].targets[j].value());
        }
    }
}

TEST_CASE("Interaction defaults exclude noise and feedback and include native rotations") {
    const auto model =
        NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, std::nullopt, {},
                                         {PartnerEffect{{1, 0, 0}, 0}, std::nullopt}, {});
    const auto circuit = parse(
        "M 0\nCX rec[-1] 1\nCZ rec[-1] 1\n"
        "DEPOLARIZE2(0.1) 0 1\nR_XX(0.17) 0 1");
    const auto expanded = expand_transition_hooks(circuit, model);
    REQUIRE(expanded.nodes.size() == circuit.nodes.size() + 2);
    CHECK(expanded.nodes[circuit.nodes.size()].gate == GateType::LEAKAGE_INTERACTION);
    CHECK(expanded.nodes.back().targets[0].value() == 1);
}

TEST_CASE("Grouped gate pairs retain their behavior without an enabled partner effect") {
    const auto make = [](GatePartnerEffects defaults, const std::vector<InteractionRule>& rules) {
        return NonComputationalModel::from_spec(
            test::pure_initial_state(Level::G), {},
            test::classifier_matrix_with_column(Level::LeakE, {0, 1}), {}, defaults, rules);
    };
    const PartnerEffect x_error{{1, 0, 0}, 0};
    const std::vector<NonComputationalModel> models{
        make({}, {}), make({PartnerEffect{}, PartnerEffect{}}, {}),
        make({}, {{"CX", 0, InteractionSource::Leaked, {}}}),
        make({}, {{"CZ", 0, InteractionSource::Leaked, x_error}}),
        make({x_error, std::nullopt},
             {{"CX", 0, InteractionSource::Leaked, {}}, {"CX", 1, InteractionSource::Leaked, {}}})};
    auto grouped = parse("X 0\nX 2\nCX 0 1\nM 1 3");
    grouped.nodes[2].targets = {Target::qubit(0), Target::qubit(1), Target::qubit(2),
                                Target::qubit(3)};
    for (const bool leaked : {false, true}) {
        Circuit circuit = grouped;
        if (leaked) {
            circuit.nodes.insert(circuit.nodes.begin() + 2,
                                 AstNode{GateType::LEAKAGE, {Target::qubit(0)}, {1.0}});
        }
        for (size_t i = 0; i < models.size(); ++i) {
            CAPTURE(leaked, i);
            const auto result = sample_noncomputational(circuit, models[i], 16, 19);
            REQUIRE(result.num_measurements == 2);
            for (const auto bit : result.measurements) {
                // The legacy policy drops the entire grouped node if any operand leaks.
                CHECK(bit == (leaked ? 0 : 1));
            }
        }
    }
    CHECK_THROWS_WITH(sample_noncomputational(grouped, make({x_error, std::nullopt}, {}), 0, 19),
                      ContainsSubstring("one physical gate pair per node"));
}

TEST_CASE("Grouped gate pairs retain post-node transition placement without partner effects") {
    auto transition = test::zero_transition_matrix();
    transition[test::level_index(Level::LeakG)][test::level_index(Level::G)] = 1;
    const auto model = NonComputationalModel::from_spec(
        test::pure_initial_state(Level::G), {{"CX", transition}},
        test::classifier_matrix_with_column(Level::LeakE, {0, 1}), {});
    auto grouped = parse("X 0\nX 1\nX 2\nCX 0 1\nM 1");
    grouped.nodes[3].targets = {Target::qubit(0), Target::qubit(1), Target::qubit(2),
                                Target::qubit(1)};
    const auto split = parse("X 0\nX 1\nX 2\nCX 0 1 2 1\nM 1");
    const auto grouped_result = sample_noncomputational(grouped, model, 16, 19);
    const auto split_result = sample_noncomputational(split, model, 16, 19);
    for (size_t shot = 0; shot < 16; ++shot) {
        CHECK(grouped_result.measurements[shot] == 1);
        CHECK(grouped_result.final_status[shot * 3 + 1] == QubitStatus::Computational);
        CHECK(split_result.measurements[shot] == 0);
        CHECK(split_result.final_status[shot * 3 + 1] == QubitStatus::LeakG);
    }
}

TEST_CASE("Interaction models reject invalid native rules and loss spreading") {
    const auto make = [](GatePartnerEffects defaults, const std::vector<InteractionRule>& rules) {
        return NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                                std::nullopt, {}, defaults, rules);
    };
    CHECK_THROWS_WITH(make({std::nullopt, PartnerEffect{{0, 0, 0}, 0.1}}, {}),
                      ContainsSubstring("leaked source"));
    CHECK_THROWS_WITH(make({}, {{"CX", 0, InteractionSource::Lost, {{0, 0, 0}, 0.1}}}),
                      ContainsSubstring("leaked source"));
    CHECK_THROWS_WITH(make({}, {{"CX", 2, InteractionSource::Leaked, {}}}),
                      ContainsSubstring("source_operand"));
    CHECK_THROWS_WITH(make({}, {{"CX", 0, static_cast<InteractionSource>(3), {}}}),
                      ContainsSubstring("source status"));
    CHECK_THROWS_WITH(make({}, {{"CX", 0, InteractionSource::Leaked, {}},
                                {"CNOT", 0, InteractionSource::Leaked, {}}}),
                      ContainsSubstring("duplicates"));
    for (const std::string gate : {"H", "CH", "CCX", "DEPOLARIZE2", "MXX", "II", "typo"}) {
        CAPTURE(gate);
        CHECK_THROWS_WITH(make({}, {{gate, 0, InteractionSource::Leaked, {}}}),
                          ContainsSubstring("native two-qubit unitary"));
    }
}
