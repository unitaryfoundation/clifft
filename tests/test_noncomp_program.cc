#include "clifft/circuit/parser.h"
#include "clifft/noncomp/program.h"

#include "noncomp_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <optional>
#include <string>
#include <utility>

using namespace clifft;

TEST_CASE("Noncomputational programs retain owned preparation across copies and moves") {
    const bool final_status = GENERATE(false, true);
    NonComputationalSample expected;
    std::string explanation;
    const auto program = [&] {
        auto circuit =
            parse("H 0\nLEAKAGE(0.4) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]");
        auto model =
            NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                             {{{1, 0, 0.3, 0.3, 0.6}, {0, 1, 0.7, 0.7, 0.4}}}, {});
        NonComputationalProgram original(circuit, model, NonComputationalExecution::Auto,
                                         {.final_status = final_status});
        expected = original.sample(65, 43, 1, 1, true);
        explanation = original.explain();
        auto copy = original;
        circuit = parse("X 0\nM 0");
        model = NonComputationalModel::from_spec(test::pure_initial_state(Level::G), {},
                                                 std::nullopt, {});
        NonComputationalProgram assigned(circuit, model);
        assigned = copy;
        NonComputationalProgram moved(std::move(assigned));
        copy = std::move(moved);
        return copy;
    }();

    CHECK(program.strategy() == (final_status ? NonComputationalExecution::Trajectory
                                              : NonComputationalExecution::Fixed));
    CHECK(program.returns_final_status() == final_status);
    CHECK(program.explain() == explanation);
    REQUIRE(program.rejection().has_value() == final_status);
    if (final_status) {
        CHECK(program.rejection()->reason == FixedNonComputationalReason::DetailedLeakedStatus);
    }
    const auto actual = program.sample(65, 43, 1, 1, true);
    CHECK(actual.shots == expected.shots);
    CHECK(actual.num_qubits == expected.num_qubits);
    CHECK(actual.num_measurements == expected.num_measurements);
    CHECK(actual.num_detectors == expected.num_detectors);
    CHECK(actual.num_observables == expected.num_observables);
    CHECK(actual.measurements == expected.measurements);
    CHECK(actual.detectors == expected.detectors);
    CHECK(actual.observables == expected.observables);
    CHECK(actual.heralds == expected.heralds);
    CHECK(actual.final_status == expected.final_status);
    REQUIRE(actual.stats);
    REQUIRE(expected.stats);
    CHECK(actual.stats->preparation_compilations == expected.stats->preparation_compilations);
    CHECK(actual.stats->preparation_seconds == expected.stats->preparation_seconds);
    CHECK(actual.stats->fixed_shots == expected.stats->fixed_shots);
    CHECK(actual.stats->trajectory_shots == expected.stats->trajectory_shots);
}
