#include "clifft/api/basis_amplitudes.h"
#include "clifft/circuit/parser.h"
#include "clifft/circuit/qasm2_parser.h"

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

void check_complex(std::complex<double> actual, std::complex<double> expected,
                   double tolerance = 1e-12) {
    INFO("actual " << actual << " expected " << expected);
    CHECK(std::abs(actual - expected) < tolerance);
}

}  // namespace

TEST_CASE("Basis amplitude query retains canonical Clifford phases") {
    constexpr double inv_sqrt_2 = 0.707106781186547524400844362104849039;

    check_complex(amplitude("H 0", 0), {inv_sqrt_2, 0.0});
    check_complex(amplitude("H 0", 1), {inv_sqrt_2, 0.0});
    check_complex(amplitude("H 0\nS 0", 0), {inv_sqrt_2, 0.0});
    check_complex(amplitude("H 0\nS 0", 1), {0.0, inv_sqrt_2});
    check_complex(amplitude("Y 0", 1), {0.0, 1.0});
    check_complex(amplitude("X 0\nZ 0", 1), {-1.0, 0.0});
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

TEST_CASE("Basis amplitude query calibrates inverse Clifford rows") {
    const clifft::Circuit circuit = clifft::parse("S_DAG 2\nX 0\nSWAP 0 1\nS_DAG 1\nCX 2 1");
    for (uint64_t basis = 0; basis < 8; ++basis) {
        const std::vector<uint64_t> output{basis};
        INFO("basis " << basis);
        const clifft::sampling::BasisAmplitudeQuery query(circuit, output);
        check_complex(query.evaluate(), basis == 2 ? std::complex<double>{0.0, -1.0}
                                                   : std::complex<double>{0.0, 0.0});
    }
}

TEST_CASE("Basis amplitude query uses its output effect to reduce active width") {
    const clifft::Circuit circuit = clifft::parse("H 1\nH 0\nT 1\nT 0");
    const std::vector<uint64_t> output{1};
    const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

    CHECK(query.peak_active_width() == 0);
    check_complex(query.evaluate(), std::polar(0.5, std::numbers::pi / 4.0));
}

TEST_CASE("Basis amplitude query falls back when the forward orientation is too wide") {
    constexpr uint32_t num_qubits = 60;
    const clifft::Circuit circuit =
        layered_circuit(clifft::GateType::H, clifft::GateType::T, num_qubits);
    const std::vector<uint64_t> output{0};
    const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

    CHECK(query.peak_active_width() == 0);
    check_complex(query.evaluate(), {std::ldexp(1.0, -30), 0.0});
}

TEST_CASE("Basis amplitude query keeps a viable forward orientation") {
    constexpr uint32_t num_qubits = 60;
    const clifft::Circuit circuit =
        layered_circuit(clifft::GateType::T, clifft::GateType::H, num_qubits);
    const std::vector<uint64_t> output{0};
    const clifft::sampling::BasisAmplitudeQuery query(circuit, output);

    CHECK(query.peak_active_width() == 0);
    check_complex(query.evaluate(), {std::ldexp(1.0, -30), 0.0});
}
