#include "clifft/circuit/parser.h"
#include "clifft/noncomp/online_state.h"
#include "clifft/noncomp/program.h"

#include "reference_clifford.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <numbers>

using namespace clifft;
using namespace clifft::noncomp_detail;

TEST_CASE("Online inverse frame agrees with scalar Clifford composition", "[noncomp][online]") {
    for (uint32_t n : {2U, 63U, 64U, 65U, 129U}) {
        CAPTURE(n);
        OnlineFrame frame(n);
        test::ReferenceTableau inverse(n);
        for (uint32_t step = 0; step < 32; ++step) {
            const uint32_t a = (step * 17) % n;
            const uint32_t b = (a + n - 1) % n;
            test::ReferenceTableau gate_inverse(n);
            switch (step % 4) {
                case 0:
                    frame.physical(GateType::H, a, 0, {});
                    gate_inverse.append_h(a);
                    break;
                case 1:
                    frame.physical(GateType::S, a, 0, {});
                    gate_inverse.append_s_dag(a);
                    break;
                case 2:
                    frame.physical(GateType::CX, a, b, {});
                    gate_inverse.append_cx(a, b);
                    break;
                case 3:
                    frame.physical(GateType::S_DAG, a, 0, {});
                    gate_inverse.append_s(a);
                    break;
            }
            inverse = gate_inverse.then(inverse);
        }
        for (uint32_t q = 0; q < n; ++q) {
            for (bool z : {false, true}) {
                const auto actual = frame.rows[q + (z ? n : 0)].view();
                const auto& expected = z ? inverse.z_output(q) : inverse.x_output(q);
                REQUIRE(actual.phase() == expected.phase());
                for (uint32_t j = 0; j < n; ++j) {
                    REQUIRE(actual.x().bit_get(j) == expected.x(j));
                    REQUIRE(actual.z().bit_get(j) == expected.z(j));
                }
            }
        }
    }
}

TEST_CASE("Online activation retains a conditional filter across growth", "[noncomp][online]") {
    OnlineState quantum(65);
    PauliString axis(65);
    quantum.frame.physical(GateType::H, 64, 0, {});
    quantum.frame.axis(64, 'Z', axis);
    const double g = std::sqrt(0.8), e = std::sqrt(0.3), probability = 0.55;
    const auto before = quantum.frame.rows;
    const auto* coefficients = quantum.state.real_data();
    quantum.no_fire(axis.view(), g, e, probability);
    REQUIRE(quantum.pending());
    REQUIRE(quantum.state.active_width() == 0);
    REQUIRE(quantum.state.real_data() == coefficients);
    for (size_t row = 0; row < before.size(); ++row)
        REQUIRE(quantum.frame.rows[row] == before[row]);

    quantum.state.ensure_capacity(1);
    quantum.finish_activation();
    REQUIRE_FALSE(quantum.pending());
    REQUIRE(quantum.peak == 1);
    quantum.frame.axis(64, 'Z', axis);
    REQUIRE(quantum.populations(axis.view()).one == Catch::Approx(0.3 / 1.1));
    quantum.frame.axis(64, 'X', axis);
    // The no-fire branch retains the coherence of diag(g,e)|+>.
    REQUIRE(quantum.populations(axis.view()).one == Catch::Approx(0.5 - g * e / (g * g + e * e)));

    const auto* grown = quantum.state.real_data();
    quantum.reset();
    REQUIRE(quantum.state.real_data() == grown);
    REQUIRE(quantum.state.max_active_width() == 1);
    REQUIRE(quantum.state.active_width() == 0);
}

TEST_CASE("Online source populations preserve a tiny occupied branch", "[noncomp][online]") {
    OnlineState quantum(1);
    PauliString axis(1);
    quantum.frame.axis(0, 'X', axis);
    constexpr double angle = 1e-10;
    quantum.rotate(axis.view(), {sampling::prepare_promotion(angle), {}});
    REQUIRE(quantum.pending());
    quantum.state.ensure_capacity(1);
    quantum.finish_activation();
    quantum.frame.axis(0, 'Z', axis);
    const double expected = std::pow(std::sin(angle * std::numbers::pi / 2), 2);
    const auto populations = quantum.populations(axis.view());
    REQUIRE(populations.one > 0);
    REQUIRE(populations.one == Catch::Approx(expected).epsilon(1e-12).margin(0));
}

TEST_CASE("Online preparation rejects malformed native operands before dispatch",
          "[noncomp][online]") {
    const auto model = NonComputationalModel::from_spec({1, 0, 0, 0, 0}, {}, std::nullopt, {});
    for (unsigned mutation = 0; mutation < 6; ++mutation) {
        auto circuit = parse("H 0\nCX 0 1\nX_ERROR(0.2) 1\nM 0 1\nDETECTOR rec[-1]");
        switch (mutation) {
            case 0:
                circuit.nodes[1].targets.pop_back();
                break;
            case 1:
                circuit.nodes[0].targets[0] = Target::qubit(3);
                break;
            case 2:
                circuit.nodes[2].args.clear();
                break;
            case 3:
                circuit.nodes[2].args[0] = -0.1;
                break;
            case 4:
                circuit.nodes.back().targets[0] = Target::rec(circuit.num_measurements);
                break;
            case 5:
                ++circuit.num_measurements;
                break;
        }
        CAPTURE(mutation);
        REQUIRE_THROWS_AS(
            NonComputationalProgram(circuit, model, NonComputationalExecution::Online),
            std::invalid_argument);
    }
}

TEST_CASE("Online preparation requires contiguous native correlated error chains",
          "[noncomp][online]") {
    const auto model = NonComputationalModel::from_spec({1, 0, 0, 0, 0}, {}, std::nullopt, {});
    for (bool interrupted : {false, true}) {
        auto circuit = parse("E(1) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1");
        if (interrupted)
            circuit.nodes.insert(circuit.nodes.begin() + 1, {GateType::H, {Target::qubit(0)}, {}});
        else
            circuit.nodes.erase(circuit.nodes.begin());
        REQUIRE_THROWS_WITH(
            NonComputationalProgram(circuit, model, NonComputationalExecution::Online),
            Catch::Matchers::ContainsSubstring("must immediately follow"));
    }
}

TEST_CASE("Online preparation restricts native inverted targets to supported gates",
          "[noncomp][online]") {
    const auto model = NonComputationalModel::from_spec({1, 0, 0, 0, 0}, {}, std::nullopt, {});
    for (const auto* text :
         {"R_PAULI(0.25) X0", "E(0.5) X0", "E(0.5) X0\nELSE_CORRELATED_ERROR(0.5) X0"}) {
        auto circuit = parse(text);
        auto& target = circuit.nodes.back().targets[0];
        target = target.inverted();
        REQUIRE_THROWS_WITH(
            NonComputationalProgram(circuit, model, NonComputationalExecution::Online),
            Catch::Matchers::ContainsSubstring("inverted target"));
    }
    REQUIRE_NOTHROW(NonComputationalProgram(parse("SPP !X0\nTPP !Y0\nM !0"), model,
                                            NonComputationalExecution::Online));
}

TEST_CASE("Online preparation rejects grouped native ordinary operations", "[noncomp][online]") {
    const auto model = NonComputationalModel::from_spec({1, 0, 0, 0, 0}, {}, std::nullopt, {});
    for (const auto* text :
         {"H 0 1", "CX 0 1 2 3", "DEPOLARIZE2(0.1) 0 1 2 3", "DEPOLARIZE3(0.1) 0 1 2 3 4 5"}) {
        auto circuit = parse(text);
        auto& targets = circuit.nodes[0].targets;
        targets.insert(targets.end(), circuit.nodes[1].targets.begin(),
                       circuit.nodes[1].targets.end());
        circuit.nodes.erase(circuit.nodes.begin() + 1);
        REQUIRE_THROWS_WITH(
            NonComputationalProgram(circuit, model, NonComputationalExecution::Online),
            Catch::Matchers::ContainsSubstring("one target group"));
    }
}
