#include "clifft/circuit/parser.h"
#include "clifft/noncomp/preparation.h"

#include "noncomp_test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <future>
#include <optional>
#include <utility>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;
using clifft::test::level_index;
using clifft::test::pure_initial_state;
using clifft::test::zero_transition_matrix;

namespace {

NonComputationalModel make_model() {
    auto leak = zero_transition_matrix();
    leak[level_index(Level::LeakG)][level_index(Level::G)] = 0.25;
    leak[level_index(Level::LeakE)][level_index(Level::E)] = 0.65;
    auto recover = zero_transition_matrix();
    recover[level_index(Level::G)][level_index(Level::LeakG)] = 0.4;
    recover[level_index(Level::E)][level_index(Level::LeakE)] = 0.3;
    recover[level_index(Level::G)][level_index(Level::Lost)] = 0.2;
    return NonComputationalModel::from_spec(
        {0.4, 0.3, 0.1, 0.1, 0.1}, {{"S", leak}, {"recover", recover}},
        {{{1.0, 0.0, 0.4, 0.2, 0.1}, {0.0, 1.0, 0.3, 0.3, 0.2}, {0.0, 0.0, 0.3, 0.5, 0.7}}},
        NonComputationalPolicy{});
}

Circuit make_circuit() {
    return parse(
        "H 0\nCX 0 1\nT 0\nS 0\nLEVEL_TRANSITION[recover] 0\n"
        "HERALD_LEAKAGE_EVENT 0\nLOSS(0.2) 1\nM 0\nCX rec[-1] 2\n"
        "MR 1\nM 2\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n");
}

void require_same_observations(const NonComputationalSample& actual,
                               const NonComputationalSample& expected) {
    REQUIRE(actual.shots == expected.shots);
    REQUIRE(actual.num_qubits == expected.num_qubits);
    REQUIRE(actual.num_measurements == expected.num_measurements);
    REQUIRE(actual.num_detectors == expected.num_detectors);
    REQUIRE(actual.num_observables == expected.num_observables);
    REQUIRE(actual.measurements == expected.measurements);
    REQUIRE(actual.detectors == expected.detectors);
    REQUIRE(actual.observables == expected.observables);
    REQUIRE(actual.heralds == expected.heralds);
}

}  // namespace

TEST_CASE("Noncomputational preparation owns its inputs across copies and moves") {
    const auto expected = sample_noncomputational(make_circuit(), make_model(), 129, 43);
    const auto request = [] {
        auto circuit = make_circuit();
        auto model = make_model();
        PreparedNonComputationalRequest original(circuit, model);
        circuit = parse("X 0\nM 0");
        model = NonComputationalModel::from_spec(pure_initial_state(Level::G), {}, std::nullopt,
                                                 NonComputationalPolicy{});
        auto copy = original;
        return PreparedNonComputationalRequest(std::move(copy));
    }();

    const auto actual = request.sample(129, 43);
    require_same_observations(actual, expected);
    REQUIRE(actual.final_status == expected.final_status);
}

TEST_CASE("Noncomputational preparation keeps execution state local to each call") {
    const PreparedNonComputationalRequest request(make_circuit(), make_model());
    const auto expected = sample_noncomputational(make_circuit(), make_model(), 257, 12345);
    REQUIRE(std::ranges::find(expected.heralds, 1) != expected.heralds.end());
    REQUIRE(std::ranges::any_of(expected.final_status,
                                [](QubitStatus status) { return !is_computational(status); }));

    const auto empty = request.sample(0);
    REQUIRE(empty.shots == 0);
    REQUIRE(empty.num_measurements == expected.num_measurements);
    REQUIRE(empty.num_qubits == expected.num_qubits);
    REQUIRE(empty.measurements.empty());
    REQUIRE(empty.final_status.empty());
    const auto other = request.sample(17, 97);
    REQUIRE(other.shots == 17);

    auto parallel = std::async(std::launch::async, [&] { return request.sample(257, 12345, 3); });
    const auto serial = request.sample(257, 12345);
    const auto concurrent = parallel.get();
    for (const auto* actual : {&serial, &concurrent}) {
        require_same_observations(*actual, expected);
        REQUIRE(actual->final_status == expected.final_status);
    }
}

TEST_CASE("Noncomputational output requirements preserve records and subsequent transitions") {
    const auto expected = sample_noncomputational(make_circuit(), make_model(), 129, 43);
    const PreparedNonComputationalRequest request(make_circuit(), make_model(),
                                                  {.final_status = false});
    for (const uint32_t threads : {1u, 3u}) {
        const auto actual = request.sample(129, 43, threads);
        require_same_observations(actual, expected);
        REQUIRE(actual.final_status.empty());
        REQUIRE(actual.final_status.capacity() == 0);
    }
    REQUIRE_FALSE(expected.final_status.empty());
}

TEST_CASE("Noncomputational preparation rejects invalid requests before execution") {
    auto circuit = make_circuit();
    auto model = make_model();
    std::string message;
    SECTION("missing named transition") {
        circuit = parse("LEVEL_TRANSITION[missing] 0");
        message = "LEVEL_TRANSITION[missing] at op 0 does not name a transition in the model";
    }
    SECTION("malformed inline probability") {
        circuit = parse("LEAKAGE(0.2) 0");
        circuit.nodes.front().args.clear();
        message = "requires exactly one argument";
    }
    SECTION("malformed transition targets") {
        const std::string instruction =
            GENERATE("LEAKAGE(0.2)", "LOSS(0.2)", "LEVEL_TRANSITION[recover]");
        circuit = parse(instruction + " 0");
        auto& targets = circuit.nodes.front().targets;
        SECTION("out of range") {
            targets = {Target::qubit(1)};
            message = "target qubit 1 is out of range at op 0";
        }
        SECTION("repeated qubit") {
            targets.push_back(Target::qubit(0));
            message = "repeats target qubit 0 at op 0";
        }
        SECTION("record reference") {
            targets = {Target::rec(0)};
            message = "at op 0 requires plain qubit targets";
        }
        SECTION("Pauli target") {
            targets = {Target::pauli(0, Target::kPauliX)};
            message = "at op 0 requires plain qubit targets";
        }
        SECTION("inverted target") {
            targets = {Target::qubit(0).inverted()};
            message = "at op 0 requires plain qubit targets";
        }
        message = "sample_noncomputational: annotation '" +
                  std::string(gate_name(circuit.nodes.front().gate)) + "' " + message;
    }
    SECTION("multi-target measurement") {
        circuit = parse("M 0");
        circuit.nodes.front().targets.push_back(Target::qubit(1));
        circuit.num_qubits = 2;
        message = "one measurement node per target";
    }
    SECTION("unsupported expectation probe takes precedence over invalid annotations") {
        circuit = parse("LEVEL_TRANSITION[missing] 0");
        circuit.num_exp_vals = 1;
        message = "EXP_VAL probes are not supported";
    }
    SECTION("missing classifier") {
        model = NonComputationalModel::from_spec(pure_initial_state(Level::Lost), {}, std::nullopt,
                                                 NonComputationalPolicy{});
        circuit = parse("M 0");
        message = "a classifier is required";
    }
    SECTION("unsupported parity measurement") {
        circuit = parse("LOSS(1) 0\nMPP Z0*Z1");
        message = "parity measurement 'MPP'";
    }
    REQUIRE_THROWS_WITH(PreparedNonComputationalRequest(circuit, model),
                        ContainsSubstring(message));
    REQUIRE_THROWS_WITH(sample_noncomputational(circuit, model, 0, 1), ContainsSubstring(message));
}

TEST_CASE("Noncomputational preparation leaves continuation width checks lazy") {
    const auto circuit = parse("H 0\nT 0");
    const auto ground = NonComputationalModel::from_spec(pure_initial_state(Level::G), {},
                                                         std::nullopt, NonComputationalPolicy{});
    const auto lost = NonComputationalModel::from_spec(pure_initial_state(Level::Lost), {},
                                                       std::nullopt, NonComputationalPolicy{});

    const PreparedNonComputationalRequest ground_request(circuit, ground, {.max_active_width = 0});
    REQUIRE(ground_request.sample(0).num_qubits == 1);
    REQUIRE_THROWS_WITH(ground_request.sample(1, 1), ContainsSubstring("max_active_width 0"));

    const PreparedNonComputationalRequest lost_request(circuit, lost, {.max_active_width = 0});
    const auto result = lost_request.sample(5, 1);
    REQUIRE(result.final_status == std::vector<QubitStatus>(5, QubitStatus::Lost));
}
