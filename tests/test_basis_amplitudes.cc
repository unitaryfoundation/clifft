#include "clifft/api/basis_amplitudes.h"
#include "clifft/circuit/parser.h"
#include "clifft/circuit/qasm2_parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/phase_aware_frontend.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

namespace {

std::complex<double> amplitude(const std::string& circuit_text, uint64_t basis) {
    const clifft::Circuit circuit = clifft::parse(circuit_text);
    std::vector<uint64_t> mask((static_cast<size_t>(circuit.num_qubits) + 63U) / 64U, 0);
    if (!mask.empty()) {
        mask[0] = basis;
    }
    return clifft::sampling::BasisAmplitudeQuery(circuit, mask).evaluate();
}

clifft::Circuit layered_circuit(clifft::GateType first, clifft::GateType second,
                                uint32_t num_qubits) {
    clifft::Circuit circuit;
    circuit.num_qubits = num_qubits;
    circuit.nodes.reserve(2 * static_cast<size_t>(num_qubits));
    for (const clifft::GateType gate : {first, second}) {
        for (uint32_t q = 0; q < num_qubits; ++q) {
            circuit.nodes.push_back(clifft::AstNode{.gate = gate,
                                                    .targets = {clifft::Target::qubit(q)},
                                                    .args = {},
                                                    .source_line = 0,
                                                    .tag = {}});
        }
    }
    return circuit;
}

using clifft::test::check_complex;

}  // namespace

TEST_CASE("Basis amplitude query retains canonical Clifford phases") {
    check_complex(amplitude("H 0", 0), {clifft::test::kInvSqrt2, 0.0});
    check_complex(amplitude("H 0", 1), {clifft::test::kInvSqrt2, 0.0});
    check_complex(amplitude("H 0\nS 0", 0), {clifft::test::kInvSqrt2, 0.0});
    check_complex(amplitude("H 0\nS 0", 1), {0.0, clifft::test::kInvSqrt2});
    check_complex(amplitude("Y 0", 1), {0.0, 1.0});
    check_complex(amplitude("X 0\nZ 0", 1), {-1.0, 0.0});
    check_complex(amplitude("X 0", 0), {0.0, 0.0});
    check_complex(amplitude("H 0\nCX 0 1", 1), {0.0, 0.0});
    check_complex(amplitude("SQRT_X 0", 0), {0.5, 0.5});
    check_complex(amplitude("SQRT_X 0", 1), {0.5, -0.5});
}

TEST_CASE("Basis amplitude query retains named phase gate scalars") {
    const auto eighth_turn = std::polar(1.0, std::numbers::pi / 4.0);
    check_complex(amplitude("X 0\nT 0", 1), eighth_turn);
    check_complex(amplitude("X 0\nT_DAG 0", 1), std::conj(eighth_turn));

    check_complex(amplitude("SPP X0*Z1", 0), {0.5, 0.5});
    check_complex(amplitude("SPP X0*Z1", 1), {0.5, -0.5});

    const std::complex<double> identity_coefficient = (1.0 + eighth_turn) / 2.0;
    const std::complex<double> pauli_coefficient = (1.0 - eighth_turn) / 2.0;
    check_complex(amplitude("TPP X0*Y1", 0), identity_coefficient);
    check_complex(amplitude("TPP X0*Y1", 3), std::complex<double>{0.0, 1.0} * pauli_coefficient);
}

TEST_CASE("Basis amplitude query retains entangled relative phase interference") {
    const std::complex<double> eighth_turn = std::polar(1.0, std::numbers::pi / 4.0);
    const clifft::Circuit circuit = clifft::parse("H 0\nCX 0 1\nT 1\nCX 0 1\nH 0");
    const std::vector<std::complex<double>> expected{
        (1.0 + eighth_turn) / 2.0,
        (1.0 - eighth_turn) / 2.0,
        {0.0, 0.0},
        {0.0, 0.0},
    };
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        INFO("basis " << basis);
        const std::vector<uint64_t> output{basis};
        check_complex(clifft::sampling::BasisAmplitudeQuery(circuit, output).evaluate(),
                      expected[basis]);
    }
}

TEST_CASE("Basis amplitude query retains persistent three qubit correlations") {
    const std::complex<double> phased = std::polar(clifft::test::kInvSqrt2, std::numbers::pi / 4.0);
    const clifft::Circuit circuit = clifft::parse("H 0\nCX 0 1\nCX 1 2\nT 2");
    for (uint64_t basis = 0; basis < 8; ++basis) {
        const std::complex<double> expected =
            basis == 0   ? std::complex<double>{clifft::test::kInvSqrt2, 0.0}
            : basis == 7 ? phased
                         : std::complex<double>{0.0, 0.0};
        INFO("basis " << basis);
        const std::vector<uint64_t> output{basis};
        check_complex(clifft::sampling::BasisAmplitudeQuery(circuit, output).evaluate(), expected);
    }
}

TEST_CASE("Basis amplitude query retains cross word Bell correlations") {
    const std::complex<double> phased = std::polar(clifft::test::kInvSqrt2, std::numbers::pi / 4.0);
    const clifft::Circuit circuit = clifft::parse("H 0\nCX 0 69\nT 69\nX 69");
    for (uint64_t q0 = 0; q0 < 2; ++q0) {
        for (uint64_t q69 = 0; q69 < 2; ++q69) {
            std::vector<uint64_t> output(2, 0);
            output[0] = q0;
            output[1] = q69 << 5U;
            const std::complex<double> expected =
                q0 == 0 && q69 == 1   ? std::complex<double>{clifft::test::kInvSqrt2, 0.0}
                : q0 == 1 && q69 == 0 ? phased
                                      : std::complex<double>{0.0, 0.0};
            INFO("q0 " << q0 << " q69 " << q69);
            check_complex(clifft::sampling::BasisAmplitudeQuery(circuit, output).evaluate(),
                          expected);
        }
    }
}

TEST_CASE("Basis amplitude query retains exponential rotation scalars") {
    const auto quarter_turn = std::polar(1.0, std::numbers::pi / 4.0);
    check_complex(amplitude("R_Z(0.5) 0", 0), std::conj(quarter_turn));
    check_complex(amplitude("X 0\nR_Z(0.5) 0", 1), quarter_turn);

    constexpr double alpha = 0.3;
    check_complex(amplitude("R_X(0.3) 0", 0), {std::cos(std::numbers::pi * alpha / 2.0), 0.0});
    check_complex(amplitude("R_X(0.3) 0", 1), {0.0, -std::sin(std::numbers::pi * alpha / 2.0)});

    check_complex(amplitude("R_PAULI(0.3) X0*Y1", 0),
                  {std::cos(std::numbers::pi * alpha / 2.0), 0.0});
    check_complex(amplitude("R_PAULI(0.3) X0*Y1", 3),
                  {std::sin(std::numbers::pi * alpha / 2.0), 0.0});

    constexpr double theta = 0.31;
    constexpr double phi = -0.47;
    constexpr double lambda = 0.59;
    check_complex(amplitude("U3(0.31,-0.47,0.59) 0", 0),
                  std::polar(std::cos(std::numbers::pi * theta / 2.0),
                             -std::numbers::pi * (lambda + phi) / 2.0));
    check_complex(amplitude("U3(0.31,-0.47,0.59) 0", 1),
                  std::polar(std::sin(std::numbers::pi * theta / 2.0),
                             std::numbers::pi * (phi - lambda) / 2.0));
}

TEST_CASE("Basis amplitude query reduces large rotation phases before evaluation") {
    constexpr const char* kLargeOddHalfTurn = "1000000000000001";
    check_complex(amplitude(std::string("R_X(") + kLargeOddHalfTurn + ") 0", 1), {0.0, -1.0});
    check_complex(amplitude(std::string("R_Y(") + kLargeOddHalfTurn + ") 0", 1), {1.0, 0.0});
    check_complex(amplitude(std::string("R_Z(") + kLargeOddHalfTurn + ") 0", 0), {0.0, -1.0});
    check_complex(amplitude("R_Z(1000000000000000.5) 0", 0),
                  std::polar(1.0, -std::numbers::pi / 4.0));
}

TEST_CASE("Basis amplitude query rejects nonunitary circuits") {
    const clifft::Circuit circuit = clifft::parse("M 0");
    const std::vector<uint64_t> output{0};
    REQUIRE_THROWS_AS(clifft::sampling::BasisAmplitudeQuery(circuit, output),
                      std::invalid_argument);
}

TEST_CASE("Basis amplitude query rejects incomplete paired operations") {
    clifft::Circuit circuit;
    circuit.num_qubits = 3;
    circuit.nodes.push_back(clifft::AstNode{
        .gate = clifft::GateType::CX,
        .targets = {clifft::Target::qubit(0), clifft::Target::qubit(1), clifft::Target::qubit(2)},
        .args = {},
        .source_line = 0,
        .tag = {}});
    const std::vector<uint64_t> output{0};
    REQUIRE_THROWS_AS(clifft::sampling::BasisAmplitudeQuery(circuit, output),
                      std::invalid_argument);
}

TEST_CASE("Trace surfaces reject malformed raw circuit nodes consistently") {
    const std::vector<uint64_t> output{0};
    auto check_rejected = [&](clifft::AstNode node, uint32_t num_qubits = 1) {
        clifft::Circuit circuit;
        circuit.num_qubits = num_qubits;
        circuit.nodes.push_back(std::move(node));
        REQUIRE_THROWS_AS(clifft::trace(circuit), std::invalid_argument);
        REQUIRE_THROWS_AS(clifft::sampling::BasisAmplitudeQuery(circuit, output),
                          std::invalid_argument);
    };

    SECTION("missing rotation argument") {
        check_rejected(
            {.gate = clifft::GateType::R_Z, .targets = {clifft::Target::qubit(0)}, .args = {}});
    }
    SECTION("duplicate pair target") {
        check_rejected({.gate = clifft::GateType::SQRT_XX,
                        .targets = {clifft::Target::qubit(0), clifft::Target::qubit(0)},
                        .args = {}});
    }
    SECTION("duplicate controlled target") {
        check_rejected({.gate = clifft::GateType::CZ,
                        .targets = {clifft::Target::qubit(0), clifft::Target::qubit(0)},
                        .args = {}});
    }
    SECTION("duplicate swap target") {
        check_rejected({.gate = clifft::GateType::ISWAP,
                        .targets = {clifft::Target::qubit(0), clifft::Target::qubit(0)},
                        .args = {}});
    }
    SECTION("record before measurement") {
        check_rejected({.gate = clifft::GateType::CX,
                        .targets = {clifft::Target::rec(0), clifft::Target::qubit(0)},
                        .args = {}});
    }
}

TEST_CASE("Basis amplitude query tracks multiple planner coordinate changes") {
    const clifft::Qasm2Import imported = clifft::parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        u3(1.461680316239578,0.2513357258846405,-1.7820438849981484) q[1];
        u3(-1.7352196490850789,0.4246563334480231,0.7252454631256664) q[0];
        z q[1]; x q[0]; z q[0]; tdg q[1];
        cx q[0],q[1]; cx q[1],q[0]; cx q[1],q[0]; s q[1];
        cz q[0],q[1]; cx q[0],q[1]; cz q[1],q[0]; cz q[1],q[0];
        cx q[0],q[1]; cz q[0],q[1]; t q[0];
        u3(-0.6035718212573498,0.9243903923110417,1.5708038144007919) q[0];
    )");
    const std::vector<uint64_t> output{0};
    const std::complex<double> input_phase =
        std::polar(1.0, std::numbers::pi * imported.global_phase_half_turns);
    const clifft::sampling::BasisAmplitudeQuery query(imported.circuit, output, input_phase);
    check_complex(query.evaluate(), {-0.5260513809700655, -0.09917410893680134});
}

TEST_CASE("Basis amplitude query retains terminal Pauli correction phases") {
    const clifft::Qasm2Import imported = clifft::parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        t q[1];
        cx q[0],q[1];
        u3(0.31,-0.47,0.59) q[0];
    )");
    const std::complex<double> input_phase =
        std::polar(1.0, std::numbers::pi * imported.global_phase_half_turns);
    constexpr double theta = 0.31;
    constexpr double phi = -0.47;
    constexpr double lambda = 0.59;
    const double cosine = clifft::test::kInvSqrt2 * std::cos(theta / 2.0);
    const double sine = clifft::test::kInvSqrt2 * std::sin(theta / 2.0);
    const std::vector<std::complex<double>> expected{
        {cosine, 0.0},
        std::polar(sine, phi),
        -std::polar(sine, lambda),
        std::polar(cosine, phi + lambda),
    };
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        const std::vector<uint64_t> output{basis};
        INFO("basis " << basis);
        check_complex(
            clifft::sampling::BasisAmplitudeQuery(imported.circuit, output, input_phase).evaluate(),
            expected[basis]);
    }
}

TEST_CASE("Basis amplitude query retains phases across effect driven reactivation") {
    constexpr double alpha = -0.91895274734932797;
    constexpr double beta = -0.97914769400380908;
    const clifft::Qasm2Import imported = clifft::parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        ry(-0.91895274734932797) q[0];
        cx q[0],q[1];
        rx(-0.97914769400380908) q[0];
    )");
    const double cosine_alpha = std::cos(alpha / 2.0);
    const double sine_alpha = std::sin(alpha / 2.0);
    const double cosine_beta = std::cos(beta / 2.0);
    const double sine_beta = std::sin(beta / 2.0);
    const std::vector<std::complex<double>> expected{
        {cosine_alpha * cosine_beta, 0.0},
        {0.0, -cosine_alpha * sine_beta},
        {0.0, -sine_alpha * sine_beta},
        {sine_alpha * cosine_beta, 0.0},
    };
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        const std::vector<uint64_t> output{basis};
        INFO("basis " << basis);
        check_complex(clifft::sampling::BasisAmplitudeQuery(imported.circuit, output).evaluate(),
                      expected[basis]);
    }
}

TEST_CASE("Basis amplitude query retains inverse Clifford row phases") {
    const clifft::Circuit circuit = clifft::parse("S_DAG 2\nX 0\nSWAP 0 1\nS_DAG 1\nCX 2 1");
    for (uint64_t basis = 0; basis < 8; ++basis) {
        const std::vector<uint64_t> output{basis};
        INFO("basis " << basis);
        const clifft::sampling::BasisAmplitudeQuery query(circuit, output);
        check_complex(query.evaluate(), basis == 2 ? std::complex<double>{0.0, -1.0}
                                                   : std::complex<double>{0.0, 0.0});
    }
}

TEST_CASE("Basis amplitude query contracts its selected output") {
    const clifft::Circuit circuit = clifft::parse("H 1\nH 0\nT 1\nT 0");
    const std::vector<uint64_t> output{1};
    const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

    CHECK(query.peak_active_width() == 0);
    check_complex(query.evaluate(), std::polar(0.5, std::numbers::pi / 4.0));
}

TEST_CASE("Basis amplitude query avoids whole-state width in either gate order") {
    constexpr uint32_t num_qubits = 60;
    const std::vector<uint64_t> output{0};
    for (const clifft::GateType first : {clifft::GateType::H, clifft::GateType::T}) {
        const clifft::GateType second =
            first == clifft::GateType::H ? clifft::GateType::T : clifft::GateType::H;
        const clifft::Circuit circuit = layered_circuit(first, second, num_qubits);
        const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

        INFO("first gate " << clifft::gate_name(first));
        CHECK(query.peak_active_width() == 0);
        // This amplitude's squared magnitude is below the sampling dust
        // threshold, but a selected-amplitude query must still retain it.
        check_complex(query.evaluate(), {std::ldexp(1.0, -30), 0.0});
    }
}

TEST_CASE("Basis amplitude query preserves exact dyadic terminal factors") {
    for (const uint32_t num_qubits : {64U, 512U, 2048U}) {
        clifft::Circuit circuit;
        circuit.num_qubits = num_qubits;
        for (uint32_t q = 0; q < num_qubits; ++q) {
            circuit.nodes.push_back(
                {.gate = clifft::GateType::H, .targets = {clifft::Target::qubit(q)}, .args = {}});
        }
        const std::vector<uint64_t> output((num_qubits + 63U) / 64U, 0);
        const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

        INFO("num_qubits " << num_qubits);
        CHECK(query.evaluate() ==
              std::complex<double>{std::ldexp(1.0, -static_cast<int>(num_qubits / 2U)), 0.0});
    }
}

TEST_CASE("Basis amplitude query handles boundary sized outputs") {
    const clifft::Circuit empty;
    check_complex(clifft::sampling::BasisAmplitudeQuery(empty, std::vector<uint64_t>{}).evaluate(),
                  {1.0, 0.0});

    const clifft::Circuit multiword = clifft::parse("X 69");
    std::vector<uint64_t> output(2, 0);
    output[1] = uint64_t{1} << 5U;
    check_complex(clifft::sampling::BasisAmplitudeQuery(multiword, output).evaluate(), {1.0, 0.0});

    output[1] = uint64_t{1} << 6U;
    REQUIRE_THROWS_AS(clifft::sampling::BasisAmplitudeQuery(multiword, output),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::BasisAmplitudeQuery(
                          multiword, std::span<const uint64_t>(output).first(1)),
                      std::invalid_argument);
}

TEST_CASE("Basis amplitude query retains active numerical dust") {
    constexpr double alpha = 1e-10;
    const clifft::Circuit circuit = clifft::parse("R_Y(1e-10) 0");
    const std::vector<uint64_t> output{1};
    const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

    CHECK(query.peak_active_width() == 1);
    check_complex(query.evaluate(), {std::sin(std::numbers::pi * alpha / 2.0), 0.0}, 1e-20);
}

TEST_CASE("Phase-aware Clifford input composition preserves operator order") {
    clifft::PhaseAwareCliffordFrame frame(3);
    frame.apply_named_gate(clifft::GateType::H, std::array<uint32_t, 1>{0});
    clifft::PauliString axis(3);
    axis.set_pauli(1, true, true);
    axis.set_sign(false);
    frame.apply_pauli_rotation(axis.view(), false);
    frame.apply_named_gate(clifft::GateType::S, std::array<uint32_t, 1>{2});

    const std::array first_input{
        clifft::PhaseAwareCliffordFrame::NamedOperation{clifft::GateType::CX, {0, 1}},
        clifft::PhaseAwareCliffordFrame::NamedOperation{clifft::GateType::X, {2}}};
    const std::array second_input{
        clifft::PhaseAwareCliffordFrame::NamedOperation{clifft::GateType::SWAP, {1, 2}},
        clifft::PhaseAwareCliffordFrame::NamedOperation{clifft::GateType::S_DAG, {0}}};
    frame.compose_input(first_input);
    frame.compose_input(second_input);

    for (uint64_t physical = 0; physical < 8; ++physical) {
        const std::vector<uint64_t> physical_basis{physical};
        const clifft::StabilizerChForm actual = frame.inverse_on_basis(physical_basis);
        clifft::StabilizerChForm expected(3);
        for (uint32_t q = 0; q < 3; ++q) {
            if (((physical >> q) & 1U) != 0) {
                expected.apply_x(q);
            }
        }
        expected.apply_s_dag(2);
        expected.apply_pauli_rotation(axis.view(), true);
        expected.apply_h(0);
        expected.apply_x(2);
        expected.apply_cx(0, 1);
        expected.apply_s(0);
        expected.apply_swap(1, 2);

        for (uint64_t virtual_basis = 0; virtual_basis < 8; ++virtual_basis) {
            INFO("physical basis " << physical << " virtual basis " << virtual_basis);
            const std::vector<uint64_t> output{virtual_basis};
            check_complex(actual.amplitude(output), expected.amplitude(output));
        }
    }
}

TEST_CASE("Phase-aware named operations reject unsupported shapes") {
    using NamedOperation = clifft::PhaseAwareCliffordFrame::NamedOperation;

    CHECK_THROWS_AS(NamedOperation(clifft::GateType::T, {0}), std::invalid_argument);
    CHECK_THROWS_AS(NamedOperation(clifft::GateType::H, {0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(NamedOperation(clifft::GateType::CX, {0}), std::invalid_argument);
}
