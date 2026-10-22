#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/fixed.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/numeric.h"

#include "noncomp_test_helpers.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <future>
#include <string>

using namespace clifft;

namespace {

NonComputationalModel model(bool ternary = false, bool restore_lost = false) {
    std::vector<std::vector<double>> classifier = {{1, 0, 0.5, 0.5, 0.5}, {0, 1, 0.5, 0.5, 0.5}};
    if (ternary) {
        classifier = {{1, 0, 0.2, 0.2, 0.2}, {0, 1, 0.3, 0.3, 0.3}, {0, 0, 0.5, 0.5, 0.5}};
    }
    const PartnerEffect depolarize{{0.25, 0.25, 0.25}, 0};
    return NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, classifier,
                                            {.reset_restores_lost = restore_lost},
                                            {depolarize, depolarize}, {});
}

FixedNonComputationalProgram fixed(const std::string& text, const NonComputationalModel& noise,
                                   bool final_status = false) {
    auto result = prepare_fixed_noncomputational(
        PreparedNonComputationalRequest(parse(text), noise, {.final_status = final_status}));
    REQUIRE(std::holds_alternative<FixedNonComputationalProgram>(result));
    return std::get<FixedNonComputationalProgram>(std::move(result));
}

void same(const NonComputationalSample& a, const NonComputationalSample& b) {
    CHECK(a.measurements == b.measurements);
    CHECK(a.heralds == b.heralds);
    CHECK(a.detectors == b.detectors);
    CHECK(a.observables == b.observables);
    CHECK(a.final_status == b.final_status);
}

}  // namespace

TEST_CASE("External readout inputs preserve reset correction and feed later records") {
    const auto plan = sampling::plan_sampling(
        trace(parse("X 0\nMR 0\nREADOUT_NOISE(0.5) rec[-1]\nCX rec[-1] 1\n"
                    "M 0 1\nDETECTOR rec[-1] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2]")));
    const sampling::ExecutablePlan compiled(plan);
    sampling::Executor executor(compiled);
    const std::array<sampling::ReadoutNoiseInput, 1> force_zero{{{false, true}}};
    executor.run_shot({}, force_zero);
    CHECK(std::vector<uint8_t>(executor.visible_records().begin(),
                               executor.visible_records().end()) == std::vector<uint8_t>{0, 0, 0});
    CHECK(executor.detectors()[0] == 0);
    CHECK(executor.observables()[0] == 0);
    const std::array<sampling::ReadoutNoiseInput, 1> force_one{{{true, false}}};
    executor.run_shot({}, force_one);
    CHECK(std::vector<uint8_t>(executor.visible_records().begin(),
                               executor.visible_records().end()) == std::vector<uint8_t>{1, 0, 1});
}

TEST_CASE("Fixed noncomputational repeated leaked readouts are independent of the survivor") {
    const auto program = fixed("H 0\nCX 0 1\nLEAKAGE(1) 0\nM 0 0 1", model());
    const uint32_t shots = 16000;
    const auto result = program.sample(shots, 12);
    std::array<uint32_t, 8> counts{};
    for (uint32_t s = 0; s < shots; ++s) {
        const auto* row = result.measurements.data() + s * 3;
        ++counts[row[0] + 2 * row[1] + 4 * row[2]];
    }
    for (const auto count : counts) {
        CHECK(std::abs(static_cast<double>(count) / shots - 0.125) < 0.015);
    }
    CHECK(result.final_status.empty());
    CHECK(program.metrics().preparation_compilations == 1);
}

TEST_CASE("Fixed noncomputational reset reports classifier values before restoring the site") {
    const auto reset = GENERATE("MR", "MRX", "MRY", "MR !", "MRX !", "MRY !");
    const std::string gate = reset;
    const bool inverted = gate.find('!') != std::string::npos;
    const std::string measurement = gate.starts_with("MRX")   ? "MX"
                                    : gate.starts_with("MRY") ? "MY"
                                                              : "M";
    const auto constant_one = NonComputationalModel::from_spec(
        test::pure_initial_state(Level::G), {}, {{{1, 0, 0, 0, 0}, {0, 1, 1, 1, 1}}}, {});
    const auto program = fixed("LEAKAGE(1) 0\n" + gate + (inverted ? "0" : " 0") +
                                   "\nCX rec[-1] 1\n" + measurement + " 0\nM 1",
                               constant_one, true);
    const auto result = program.sample(64, 5);
    for (size_t s = 0; s < result.shots; ++s) {
        CHECK(result.measurements[s * 3] == !inverted);
        CHECK(result.measurements[s * 3 + 1] == 0);
        CHECK(result.measurements[s * 3 + 2] == !inverted);
        CHECK(result.final_status[s * 2] == QubitStatus::Computational);
    }
}

TEST_CASE("Fixed noncomputational ternary readout preserves the joint herald distribution") {
    const auto program = fixed(
        "LEAKAGE(1) 0\nM 0\nCX rec[-1] 1\nM 1\n"
        "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        model(true));
    const auto result = program.sample(16000, 71);
    std::array<uint32_t, 4> counts{};
    for (size_t s = 0; s < result.shots; ++s) {
        ++counts[result.measurements[s * 2] + 2 * result.heralds[s * 2]];
        REQUIRE(result.measurements[s * 2] == result.measurements[s * 2 + 1]);
        REQUIRE(result.detectors[s] == 0);
        REQUIRE(result.observables[s] == result.measurements[s * 2]);
    }
    const std::array<double, 4> expected{0.2, 0.3, 0.25, 0.25};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(std::abs(static_cast<double>(counts[i]) / result.shots - expected[i]) < 0.02);
    }
}

TEST_CASE(
    "Fixed noncomputational partner depolarization absorbs retained gates in both directions") {
    const int source = GENERATE(0, 1);
    const auto event = GENERATE("LEAKAGE", "LOSS");
    const auto program = fixed(std::string(event) + "(1) " + std::to_string(source) +
                                   "\nCX 0 1\nM " + std::to_string(1 - source),
                               model());
    const auto result = program.sample(6000, 32);
    size_t ones = 0;
    for (const auto bit : result.measurements) {
        ones += bit;
    }
    CHECK(std::abs(static_cast<double>(ones) / result.shots - 0.5) < 0.025);
}

TEST_CASE("Fixed noncomputational distinguishes ordinary dropped noise from correlated branches") {
    const auto noise =
        GENERATE("DEPOLARIZE2(1) 0 1", "CORRELATED_ERROR(1) X0\nELSE_CORRELATED_ERROR(1) X1");
    const auto program = fixed(std::string("LEAKAGE(1) 0\n") + noise + "\nM 1", model());
    const auto result = program.sample(128, 9);
    for (const auto bit : result.measurements) {
        CHECK(bit == 0);
    }
}

TEST_CASE("Fixed noncomputational preserves non Clifford probabilities with erased labels") {
    const auto program = fixed("H 0\nT 0\nH 0\nLEAKAGE(0.3) 1\nM 0\nR 1", model(), true);
    CHECK(program.metrics().peak_active_width > 0);
    const auto result = program.sample(10000, 72);
    size_t ones = 0;
    for (const auto bit : result.measurements) {
        ones += bit;
    }
    CHECK(std::abs(static_cast<double>(ones) / result.shots -
                   std::pow(std::sin(std::acos(-1.0) / 8), 2)) < 0.02);
}

TEST_CASE("Fixed noncomputational programs own their plan and share no mutable shot state") {
    const auto program = fixed(
        "H 0\nCX 0 1\nLEAKAGE(0.3) 0\nLOSS(0.2) 1\n"
        "CX 0 1\nM 0\nHERALD_LOSS_EVENT 1\nR 0\nM 1",
        model(true), true);
    const auto expected = program.sample(1025, 123);
    auto copy = program;
    auto future = std::async(std::launch::async, [&] { return copy.sample(1025, 123, 3); });
    same(expected, program.sample(1025, 123, 2));
    same(expected, future.get());
    const auto empty = program.sample(0);
    CHECK(empty.measurements.empty());
    CHECK(empty.num_measurements == 3);
    CHECK(program.metrics().preparation_compilations == 1);
}

TEST_CASE("Fixed noncomputational rejects requests whose observable contract needs trajectories") {
    auto noise = model();
    auto circuit = parse("LEAKAGE(1) 0\nM 0");
    NonComputationalRequirements requirements{.final_status = false};
    FixedNonComputationalReason reason;
    SECTION("detailed leaked label") {
        requirements.final_status = true;
        reason = FixedNonComputationalReason::DetailedLeakedStatus;
    }
    SECTION("level sensitive classifier") {
        noise = NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                                 {{{1, 0, 1, 0, 0}, {0, 1, 0, 1, 1}}}, {});
        reason = FixedNonComputationalReason::LevelSensitiveReadout;
    }
    SECTION("unabsorbed loss interaction") {
        noise =
            NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                             {{{1, 0, 0.5, 0.5, 0.5}, {0, 1, 0.5, 0.5, 0.5}}}, {});
        circuit = parse("LOSS(0.1) 0\nCX 0 1\nM 1");
        reason = FixedNonComputationalReason::PartnerChannel;
    }
    SECTION("explicit spreading") {
        circuit = parse("LEAKAGE(1) 0\nLEAKAGE_INTERACTION(0.25,0.25,0.25,0.1) 0 1");
        reason = FixedNonComputationalReason::Spreading;
    }
    SECTION("model configured spreading") {
        noise =
            NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, std::nullopt,
                                             {}, {PartnerEffect{{0.25, 0.25, 0.25}, 0.1}, {}}, {});
        circuit = parse("LEAKAGE(1) 0\nCX 0 1");
        reason = FixedNonComputationalReason::Spreading;
    }
    SECTION("unsupported parity measurement") {
        circuit = parse("MPP Z0*Z1");
        reason = FixedNonComputationalReason::UnsupportedOperation;
    }
    SECTION("unsupported Pauli product unitary") {
        const auto gate = GENERATE("SPP", "SPP_DAG", "TPP", "TPP_DAG", "R_PAULI(0.125)");
        circuit = parse(std::string("LOSS(1) 0\n") + gate + " X0*X1");
        reason = FixedNonComputationalReason::UnsupportedOperation;
    }
    SECTION("initial mixed state") {
        noise = NonComputationalModel::from_spec({0.5, 0.5, 0, 0, 0}, {}, std::nullopt, {});
        circuit = parse("H 0");
        reason = FixedNonComputationalReason::InitialState;
    }
    SECTION("nonzero named channel") {
        auto transition = test::zero_transition_matrix();
        transition[0][0] = 0.1;
        noise = NonComputationalModel::from_spec(test::pure_initial_state(Level::G),
                                                 {{"project", transition}}, std::nullopt, {});
        circuit = parse("H 0\nLEVEL_TRANSITION[project] 0");
        reason = FixedNonComputationalReason::NamedTransition;
    }
    SECTION("width cap") {
        circuit = parse("H 0\nT 0\nM 0");
        requirements.max_active_width = 0;
        reason = FixedNonComputationalReason::ActiveWidth;
    }
    SECTION("planner width limit") {
        std::string text;
        for (uint32_t qubit = 0; qubit < kDenseActiveWidthLimit; ++qubit) {
            const auto q = std::to_string(qubit);
            text += "H " + q + "\nT " + q + "\nT " + q + "\n";
        }
        circuit = parse(text);
        reason = FixedNonComputationalReason::ActiveWidth;
    }
    const auto result = prepare_fixed_noncomputational(
        PreparedNonComputationalRequest(circuit, noise, requirements));
    REQUIRE(std::holds_alternative<FixedNonComputationalRejection>(result));
    CHECK(std::get<FixedNonComputationalRejection>(result).reason == reason);
}

TEST_CASE("Fixed noncomputational validates every consumed correlated branch") {
    const size_t branch = GENERATE(size_t{0}, size_t{1}, size_t{2});
    auto circuit = parse(
        "CORRELATED_ERROR(0.2) X0\nELSE_CORRELATED_ERROR(0.3) X0\n"
        "ELSE_CORRELATED_ERROR(0) X0\nM 0");
    circuit.nodes[branch].targets = {Target::pauli(1, Target::kPauliX)};
    const PreparedNonComputationalRequest request(circuit, model());
    CHECK_THROWS_WITH(prepare_fixed_noncomputational(request),
                      Catch::Matchers::ContainsSubstring("operand is out of range"));
}

TEST_CASE("Fixed noncomputational rejects malformed readout targets before lowering") {
    const auto target = GENERATE(Target::qubit(0), Target::rec(0).inverted(),
                                 Target{Target::rec(0).bits | Target::kPauliX});
    auto circuit = parse("M 0\nREADOUT_NOISE(1) rec[-1]");
    circuit.nodes[1].targets = {target};
    const PreparedNonComputationalRequest request(circuit, model());
    CHECK_THROWS_WITH(prepare_fixed_noncomputational(request),
                      Catch::Matchers::ContainsSubstring("plain record targets"));
}

TEST_CASE("Fixed noncomputational rejects grouped operations with whole node drop semantics") {
    const bool reset = GENERATE(false, true);
    auto circuit = parse(reset ? "X 1\nLOSS(1) 0\nR 0\nR 1\nM 1" : "LEAKAGE(1) 0\nX 0\nX 1\nM 1");
    const size_t group = reset ? 2 : 1;
    circuit.nodes[group].targets.push_back(Target::qubit(1));
    circuit.nodes.erase(circuit.nodes.begin() + group + 1);
    const PreparedNonComputationalRequest request(circuit, model(), {.final_status = false});
    const auto result = prepare_fixed_noncomputational(request);
    REQUIRE(std::holds_alternative<FixedNonComputationalRejection>(result));
    const auto& rejection = std::get<FixedNonComputationalRejection>(result);
    CHECK(rejection.reason == FixedNonComputationalReason::UnsupportedOperation);
    CHECK(rejection.operation == group);
    CHECK(rejection.source_line == group + 1);
    const auto fallback = request.sample(16, 3);
    for (const auto bit : fallback.measurements) {
        CHECK(bit == reset);
    }
}

TEST_CASE("Fixed noncomputational accepts grouped operations without whole node drops") {
    const bool reset = GENERATE(false, true);
    auto circuit = parse(reset ? "X 1\nLOSS(1) 0\nR 0\nR 1\nM 1" : "LEAKAGE(0) 0\nX 0\nX 1\nM 1");
    const size_t group = reset ? 2 : 1;
    circuit.nodes[group].targets.push_back(Target::qubit(1));
    circuit.nodes.erase(circuit.nodes.begin() + group + 1);
    const PreparedNonComputationalRequest request(circuit, model(false, true));
    const auto result = prepare_fixed_noncomputational(request);
    REQUIRE(std::holds_alternative<FixedNonComputationalProgram>(result));
    same(std::get<FixedNonComputationalProgram>(result).sample(16, 3), request.sample(16, 3));
}

TEST_CASE("Fixed noncomputational permits unreachable spreading") {
    const auto noise =
        NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {}, std::nullopt, {},
                                         {PartnerEffect{{0.25, 0.25, 0.25}, 0.1}, {}}, {});
    const auto circuit = GENERATE("LEAKAGE(0) 0\nCX 0 1", "LEAKAGE(1) 0 1\nCX 0 1");
    const auto program = fixed(circuit, noise);
    CHECK(program.sample(1, 9).num_qubits == 2);
}

TEST_CASE("Fixed noncomputational rejection identifies the expanded hook node") {
    auto transition = test::zero_transition_matrix();
    transition[0][0] = 0.1;
    const auto noise = NonComputationalModel::from_spec(test::pure_initial_state(Level::G),
                                                        {{"H", transition}}, std::nullopt, {});
    const PreparedNonComputationalRequest request(parse("X 0\nH 0"), noise);
    const auto result = prepare_fixed_noncomputational(request);
    REQUIRE(std::holds_alternative<FixedNonComputationalRejection>(result));
    const auto& rejection = std::get<FixedNonComputationalRejection>(result);
    CHECK(rejection.reason == FixedNonComputationalReason::NamedTransition);
    REQUIRE(rejection.operation == 2);
    CHECK(request.resolved_circuit().operation(*rejection.operation).generated_by == 1);
    CHECK(rejection.source_line == 2);
}
