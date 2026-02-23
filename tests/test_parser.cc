// Circuit parser tests.
//
// Tests the parser's ability to:
// - Parse basic gates (H, CX, T, M)
// - Handle rec[-k] references
// - Parse MPP with multiple Pauli products
// - Decompose resets (R -> M + CX rec[-1])
// - Parse classical feedback (CX rec[-k] q)
// - Report errors for unknown gates and malformed syntax

#include "ucc/circuit/circuit.h"
#include "ucc/circuit/parser.h"
#include "ucc/circuit/target.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ucc;

TEST_CASE("Parse empty circuit", "[parser]") {
    auto circuit = parse("");
    REQUIRE(circuit.nodes.empty());
    REQUIRE(circuit.num_qubits == 0);
    REQUIRE(circuit.num_measurements == 0);
}

TEST_CASE("Parse comments and whitespace", "[parser]") {
    auto circuit = parse(R"(
        # This is a comment
        H 0  # inline comment

        # Another comment
        X 1
    )");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.nodes[0].gate == GateType::H);
    REQUIRE(circuit.nodes[1].gate == GateType::X);
}

TEST_CASE("Parse single-qubit gates", "[parser]") {
    auto circuit = parse(R"(
        H 0
        S 1
        S_DAG 2
        X 3
        Y 4
        Z 5
        T 6
        T_DAG 7
    )");

    REQUIRE(circuit.nodes.size() == 8);
    REQUIRE(circuit.num_qubits == 8);

    REQUIRE(circuit.nodes[0].gate == GateType::H);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[1].gate == GateType::S);
    REQUIRE(circuit.nodes[2].gate == GateType::S_DAG);
    REQUIRE(circuit.nodes[3].gate == GateType::X);
    REQUIRE(circuit.nodes[4].gate == GateType::Y);
    REQUIRE(circuit.nodes[5].gate == GateType::Z);
    REQUIRE(circuit.nodes[6].gate == GateType::T);
    REQUIRE(circuit.nodes[7].gate == GateType::T_DAG);
}

TEST_CASE("Parse multi-target single-qubit gates", "[parser]") {
    auto circuit = parse("H 0 1 2 3");

    // Should expand to 4 separate nodes.
    REQUIRE(circuit.nodes.size() == 4);
    REQUIRE(circuit.num_qubits == 4);

    for (size_t i = 0; i < 4; i++) {
        REQUIRE(circuit.nodes[i].gate == GateType::H);
        REQUIRE(circuit.nodes[i].targets.size() == 1);
        REQUIRE(circuit.nodes[i].targets[0].value() == i);
    }
}

TEST_CASE("Parse two-qubit gates", "[parser]") {
    auto circuit = parse(R"(
        CX 0 1
        CY 2 3
        CZ 4 5
    )");

    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_qubits == 6);

    REQUIRE(circuit.nodes[0].gate == GateType::CX);
    REQUIRE(circuit.nodes[0].targets.size() == 2);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(circuit.nodes[0].targets[1].value() == 1);

    REQUIRE(circuit.nodes[1].gate == GateType::CY);
    REQUIRE(circuit.nodes[2].gate == GateType::CZ);
}

TEST_CASE("Parse multi-pair two-qubit gates", "[parser]") {
    auto circuit = parse("CX 0 1 2 3 4 5");

    // Should expand to 3 pairs.
    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_qubits == 6);

    REQUIRE(circuit.nodes[0].gate == GateType::CX);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(circuit.nodes[0].targets[1].value() == 1);

    REQUIRE(circuit.nodes[1].gate == GateType::CX);
    REQUIRE(circuit.nodes[1].targets[0].value() == 2);
    REQUIRE(circuit.nodes[1].targets[1].value() == 3);

    REQUIRE(circuit.nodes[2].gate == GateType::CX);
    REQUIRE(circuit.nodes[2].targets[0].value() == 4);
    REQUIRE(circuit.nodes[2].targets[1].value() == 5);
}

TEST_CASE("Parse CNOT alias", "[parser]") {
    auto circuit = parse("CNOT 0 1");

    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.nodes[0].gate == GateType::CX);
}

TEST_CASE("Parse measurements", "[parser]") {
    auto circuit = parse(R"(
        M 0
        MX 1
        MY 2
    )");

    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_measurements == 3);

    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[1].gate == GateType::MX);
    REQUIRE(circuit.nodes[2].gate == GateType::MY);
}

TEST_CASE("Parse multi-target measurements", "[parser]") {
    auto circuit = parse("M 0 1 2");

    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_measurements == 3);

    for (size_t i = 0; i < 3; i++) {
        REQUIRE(circuit.nodes[i].gate == GateType::M);
        REQUIRE(circuit.nodes[i].targets[0].value() == i);
    }
}

TEST_CASE("Parse MR and MRX as first-class gates", "[parser]") {
    // MR and MRX are kept as first-class gates (not decomposed)
    auto circuit = parse(R"(
        MR 0
        MRX 1
    )");

    // 2 measure-reset gates = 2 nodes
    REQUIRE(circuit.nodes.size() == 2);
    // MR and MRX produce visible measurements
    REQUIRE(circuit.num_measurements == 2);

    // MR 0
    REQUIRE(circuit.nodes[0].gate == GateType::MR);
    REQUIRE(circuit.nodes[0].targets.size() == 1);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    // MRX 1
    REQUIRE(circuit.nodes[1].gate == GateType::MRX);
    REQUIRE(circuit.nodes[1].targets.size() == 1);
    REQUIRE(circuit.nodes[1].targets[0].value() == 1);
}

TEST_CASE("Parse MPP single product", "[parser]") {
    auto circuit = parse("MPP X0*Z1*Y2");

    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.num_measurements == 1);
    REQUIRE(circuit.nodes[0].gate == GateType::MPP);
    REQUIRE(circuit.nodes[0].targets.size() == 3);

    // Check Pauli tags.
    REQUIRE(circuit.nodes[0].targets[0].pauli() == Target::kPauliX);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[0].targets[1].pauli() == Target::kPauliZ);
    REQUIRE(circuit.nodes[0].targets[1].value() == 1);

    REQUIRE(circuit.nodes[0].targets[2].pauli() == Target::kPauliY);
    REQUIRE(circuit.nodes[0].targets[2].value() == 2);
}

TEST_CASE("Parse MPP multiple products", "[parser]") {
    auto circuit = parse("MPP X0*X1 Z0*Z1 Y2");

    // Should unroll into 3 separate AstNodes.
    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_measurements == 3);

    // First product: X0*X1
    REQUIRE(circuit.nodes[0].gate == GateType::MPP);
    REQUIRE(circuit.nodes[0].targets.size() == 2);
    REQUIRE(circuit.nodes[0].targets[0].pauli() == Target::kPauliX);
    REQUIRE(circuit.nodes[0].targets[1].pauli() == Target::kPauliX);

    // Second product: Z0*Z1
    REQUIRE(circuit.nodes[1].gate == GateType::MPP);
    REQUIRE(circuit.nodes[1].targets.size() == 2);
    REQUIRE(circuit.nodes[1].targets[0].pauli() == Target::kPauliZ);
    REQUIRE(circuit.nodes[1].targets[1].pauli() == Target::kPauliZ);

    // Third product: Y2 (single Pauli)
    REQUIRE(circuit.nodes[2].gate == GateType::MPP);
    REQUIRE(circuit.nodes[2].targets.size() == 1);
    REQUIRE(circuit.nodes[2].targets[0].pauli() == Target::kPauliY);
}

TEST_CASE("Parse noisy MPP decomposes to MPP plus READOUT_NOISE", "[parser][noise]") {
    // MPP(p) X0*Z1 should decompose to MPP + READOUT_NOISE
    auto circuit = parse("MPP(0.001) X0*Z1");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_measurements == 1);

    // First node: clean MPP with arg=0
    REQUIRE(circuit.nodes[0].gate == GateType::MPP);
    REQUIRE(circuit.nodes[0].arg == 0.0);
    REQUIRE(circuit.nodes[0].targets.size() == 2);

    // Second node: READOUT_NOISE with original probability
    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].arg == 0.001);
    REQUIRE(circuit.nodes[1].targets.size() == 1);
    REQUIRE(circuit.nodes[1].targets[0].is_rec());
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // First measurement
}

TEST_CASE("Parse noisy MPP multiple products each get READOUT_NOISE", "[parser][noise]") {
    // MPP(0.002) X0*X1 Z2*Z3 should produce 4 nodes: MPP, READOUT_NOISE, MPP, READOUT_NOISE
    auto circuit = parse("MPP(0.002) X0*X1 Z2*Z3");

    REQUIRE(circuit.nodes.size() == 4);
    REQUIRE(circuit.num_measurements == 2);

    // First product: MPP + READOUT_NOISE
    REQUIRE(circuit.nodes[0].gate == GateType::MPP);
    REQUIRE(circuit.nodes[0].arg == 0.0);
    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].arg == 0.002);
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // meas index 0

    // Second product: MPP + READOUT_NOISE
    REQUIRE(circuit.nodes[2].gate == GateType::MPP);
    REQUIRE(circuit.nodes[2].arg == 0.0);
    REQUIRE(circuit.nodes[3].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[3].arg == 0.002);
    REQUIRE(circuit.nodes[3].targets[0].value() == 1);  // meas index 1
}

TEST_CASE("Parse reset R as first-class gate", "[parser]") {
    auto circuit = parse("R 0");

    // R is kept as a first-class gate, not decomposed
    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.num_measurements == 0);  // R has no visible measurement

    REQUIRE(circuit.nodes[0].gate == GateType::R);
    REQUIRE(circuit.nodes[0].targets.size() == 1);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(!circuit.nodes[0].targets[0].is_rec());
}

TEST_CASE("Parse reset RX as first-class gate", "[parser]") {
    auto circuit = parse("RX 0");

    // RX is kept as a first-class gate, not decomposed
    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.num_measurements == 0);  // RX has no visible measurement

    REQUIRE(circuit.nodes[0].gate == GateType::RX);
}

TEST_CASE("Parse multiple resets", "[parser]") {
    auto circuit = parse("R 0 1 2");

    // 3 resets -> 3 nodes (one per qubit)
    REQUIRE(circuit.nodes.size() == 3);
    REQUIRE(circuit.num_measurements == 0);  // No visible measurements from R

    // All should be R gates
    REQUIRE(circuit.nodes[0].gate == GateType::R);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(circuit.nodes[1].gate == GateType::R);
    REQUIRE(circuit.nodes[1].targets[0].value() == 1);
    REQUIRE(circuit.nodes[2].gate == GateType::R);
    REQUIRE(circuit.nodes[2].targets[0].value() == 2);
}

TEST_CASE("Parse classical feedback CX rec[-k] q", "[parser]") {
    auto circuit = parse(R"(
        M 0
        M 1
        CX rec[-1] 2
        CZ rec[-2] 3
    )");

    REQUIRE(circuit.nodes.size() == 4);
    REQUIRE(circuit.num_measurements == 2);

    // CX rec[-1] 2 -> first target is rec[1], second is qubit 2
    REQUIRE(circuit.nodes[2].gate == GateType::CX);
    REQUIRE(circuit.nodes[2].targets[0].is_rec());
    REQUIRE(circuit.nodes[2].targets[0].value() == 1);  // rec[-1] -> abs 1
    REQUIRE(!circuit.nodes[2].targets[1].is_rec());
    REQUIRE(circuit.nodes[2].targets[1].value() == 2);

    // CZ rec[-2] 3 -> first target is rec[0], second is qubit 3
    REQUIRE(circuit.nodes[3].gate == GateType::CZ);
    REQUIRE(circuit.nodes[3].targets[0].is_rec());
    REQUIRE(circuit.nodes[3].targets[0].value() == 0);  // rec[-2] -> abs 0
    REQUIRE(circuit.nodes[3].targets[1].value() == 3);
}

TEST_CASE("Parse TICK annotation", "[parser]") {
    auto circuit = parse(R"(
        H 0
        TICK
        CX 0 1
        TICK
        M 0 1
    )");

    REQUIRE(circuit.nodes.size() == 6);  // H, TICK, CX, TICK, M, M

    REQUIRE(circuit.nodes[1].gate == GateType::TICK);
    REQUIRE(circuit.nodes[1].targets.empty());

    REQUIRE(circuit.nodes[3].gate == GateType::TICK);
}

TEST_CASE("Parse gate with parenthesized argument", "[parser]") {
    // Even though we don't use noise in MVP, parser should skip the argument.
    auto circuit = parse("H(0.5) 0");

    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.nodes[0].gate == GateType::H);
    REQUIRE(circuit.nodes[0].arg == 0.5);
}

TEST_CASE("Parse inverted measurement targets", "[parser]") {
    auto circuit = parse("M !0 1 !2");

    REQUIRE(circuit.nodes.size() == 3);

    REQUIRE(circuit.nodes[0].targets[0].is_inverted());
    REQUIRE(!circuit.nodes[1].targets[0].is_inverted());
    REQUIRE(circuit.nodes[2].targets[0].is_inverted());
}

TEST_CASE("Error: unknown gate", "[parser]") {
    REQUIRE_THROWS_AS(parse("FOOBAR 0"), ParseError);
}

TEST_CASE("Error: REPEAT not supported", "[parser]") {
    REQUIRE_THROWS_AS(parse("REPEAT 10 {\nH 0\n}"), ParseError);
}

TEST_CASE("Error: odd number of targets for two-qubit gate", "[parser]") {
    REQUIRE_THROWS_AS(parse("CX 0 1 2"), ParseError);
}

TEST_CASE("Error: rec reference out of bounds", "[parser]") {
    // No measurements yet, so rec[-1] is invalid.
    REQUIRE_THROWS_AS(parse("CX rec[-1] 0"), ParseError);
}

TEST_CASE("Error: positive rec offset", "[parser]") {
    REQUIRE_THROWS_AS(parse("M 0\nCX rec[0] 1"), ParseError);
}

TEST_CASE("Error: unclosed parenthesis", "[parser]") {
    REQUIRE_THROWS_AS(parse("H(0.5 0"), ParseError);
}

TEST_CASE("Error: invalid MPP Pauli", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPP W0"), ParseError);
}

TEST_CASE("Error: MPP with no products", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPP"), ParseError);
}

TEST_CASE("Error: MPP trailing asterisk", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPP X0*"), ParseError);
}

TEST_CASE("Error: MPP leading asterisk", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPP *X0"), ParseError);
}

TEST_CASE("Error: MPP consecutive asterisks", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPP X0**Z1"), ParseError);
}

TEST_CASE("Error: R with no targets", "[parser]") {
    REQUIRE_THROWS_AS(parse("R"), ParseError);
}

TEST_CASE("Error: RX with no targets", "[parser]") {
    REQUIRE_THROWS_AS(parse("RX"), ParseError);
}

TEST_CASE("Error: TICK with targets", "[parser]") {
    REQUIRE_THROWS_AS(parse("TICK 0"), ParseError);
}

TEST_CASE("Error: stray closing brace", "[parser]") {
    REQUIRE_THROWS_AS(parse("}"), ParseError);
}

TEST_CASE("Error: rec target on non-feedback gate", "[parser]") {
    // rec targets are only valid for CX/CZ feedback
    REQUIRE_THROWS_AS(parse("M 0\nH rec[-1]"), ParseError);
    REQUIRE_THROWS_AS(parse("M 0\nM rec[-1]"), ParseError);
    REQUIRE_THROWS_AS(parse("M 0\nCY rec[-1] 1"), ParseError);
}

TEST_CASE("Error: invalid feedback syntax", "[parser]") {
    // CX qubit rec is wrong order (should be rec qubit)
    REQUIRE_THROWS_AS(parse("M 0\nCX 1 rec[-1]"), ParseError);
    // Both rec targets is invalid
    REQUIRE_THROWS_AS(parse("M 0\nM 1\nCX rec[-1] rec[-2]"), ParseError);
}

TEST_CASE("Error: rec with trailing characters", "[parser]") {
    REQUIRE_THROWS_AS(parse("M 0\nCX rec[-1]foo 1"), ParseError);
}

TEST_CASE("Realistic circuit: Bell state preparation and measurement", "[parser]") {
    auto circuit = parse(R"(
        H 0
        CX 0 1
        M 0 1
    )");

    REQUIRE(circuit.nodes.size() == 4);  // H, CX, M, M
    REQUIRE(circuit.num_qubits == 2);
    REQUIRE(circuit.num_measurements == 2);
}

TEST_CASE("Realistic circuit: T gate with feedback", "[parser]") {
    auto circuit = parse(R"(
        H 0
        T 0
        M 0
        CX rec[-1] 1
        H 1
    )");

    REQUIRE(circuit.nodes.size() == 5);
    REQUIRE(circuit.num_qubits == 2);
    REQUIRE(circuit.num_measurements == 1);

    REQUIRE(circuit.nodes[1].gate == GateType::T);
    REQUIRE(circuit.nodes[3].gate == GateType::CX);
    REQUIRE(circuit.nodes[3].targets[0].is_rec());
}

TEST_CASE("Realistic circuit: surface code syndrome extraction", "[parser]") {
    auto circuit = parse(R"(
        # Simple 3-qubit syndrome extraction
        R 0
        R 1
        R 2
        H 1
        CX 1 0
        CX 1 2
        H 1
        M 1
    )");

    // 3 resets + H + CX + CX + H + M = 8 nodes
    REQUIRE(circuit.nodes.size() == 8);
    REQUIRE(circuit.num_qubits == 3);
    // Resets have no visible measurement, only M 1 counts
    REQUIRE(circuit.num_measurements == 1);
}

// =============================================================================
// Phase 2.1: Noise and QEC gate parsing tests
// =============================================================================

TEST_CASE("Parse noise gates: X_ERROR Y_ERROR Z_ERROR", "[parser][noise]") {
    auto circuit = parse(R"(
        X_ERROR(0.001) 0 1 2
        Y_ERROR(0.002) 3
        Z_ERROR(0.003) 4 5
    )");

    REQUIRE(circuit.nodes.size() == 6);
    REQUIRE(circuit.num_qubits == 6);

    REQUIRE(circuit.nodes[0].gate == GateType::X_ERROR);
    REQUIRE(circuit.nodes[0].arg == Catch::Approx(0.001));
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[3].gate == GateType::Y_ERROR);
    REQUIRE(circuit.nodes[3].arg == Catch::Approx(0.002));

    REQUIRE(circuit.nodes[4].gate == GateType::Z_ERROR);
    REQUIRE(circuit.nodes[4].arg == Catch::Approx(0.003));
}

TEST_CASE("Parse noise gates: DEPOLARIZE1 DEPOLARIZE2", "[parser][noise]") {
    auto circuit = parse(R"(
        DEPOLARIZE1(0.01) 0 1
        DEPOLARIZE2(0.02) 2 3 4 5
    )");

    // DEPOLARIZE1 is single-qubit arity: 2 nodes
    // DEPOLARIZE2 is pair arity: 2 nodes (pairs 2-3, 4-5)
    REQUIRE(circuit.nodes.size() == 4);
    REQUIRE(circuit.num_qubits == 6);

    REQUIRE(circuit.nodes[0].gate == GateType::DEPOLARIZE1);
    REQUIRE(circuit.nodes[0].arg == Catch::Approx(0.01));

    REQUIRE(circuit.nodes[2].gate == GateType::DEPOLARIZE2);
    REQUIRE(circuit.nodes[2].arg == Catch::Approx(0.02));
    REQUIRE(circuit.nodes[2].targets.size() == 2);
    REQUIRE(circuit.nodes[2].targets[0].value() == 2);
    REQUIRE(circuit.nodes[2].targets[1].value() == 3);
}

TEST_CASE("Parse noisy measurement: M with readout noise decomposes", "[parser][noise]") {
    auto circuit = parse("M(0.001) 0");

    // Should decompose into M + READOUT_NOISE
    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_measurements == 1);

    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[0].arg == 0.0);  // Clean measurement
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].arg == Catch::Approx(0.001));
    REQUIRE(circuit.nodes[1].targets[0].is_rec());
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // rec[0]
}

TEST_CASE("Parse multi-target noisy measurement interleaves", "[parser][noise]") {
    auto circuit = parse("M(0.002) 0 1 2");

    // M 0, READOUT_NOISE rec[0], M 1, READOUT_NOISE rec[1], M 2, READOUT_NOISE rec[2]
    REQUIRE(circuit.nodes.size() == 6);
    REQUIRE(circuit.num_measurements == 3);

    // Check interleaving pattern
    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // rec[0]

    REQUIRE(circuit.nodes[2].gate == GateType::M);
    REQUIRE(circuit.nodes[2].targets[0].value() == 1);

    REQUIRE(circuit.nodes[3].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[3].targets[0].value() == 1);  // rec[1]

    REQUIRE(circuit.nodes[4].gate == GateType::M);
    REQUIRE(circuit.nodes[4].targets[0].value() == 2);

    REQUIRE(circuit.nodes[5].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[5].targets[0].value() == 2);  // rec[2]
}

TEST_CASE("Parse MX MY with readout noise", "[parser][noise]") {
    auto circuit = parse(R"(
        MX(0.003) 0
        MY(0.004) 1
    )");

    REQUIRE(circuit.nodes.size() == 4);
    REQUIRE(circuit.num_measurements == 2);

    REQUIRE(circuit.nodes[0].gate == GateType::MX);
    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].arg == Catch::Approx(0.003));

    REQUIRE(circuit.nodes[2].gate == GateType::MY);
    REQUIRE(circuit.nodes[3].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[3].arg == Catch::Approx(0.004));
}

TEST_CASE("Parse M without noise: no READOUT_NOISE emitted", "[parser][noise]") {
    auto circuit = parse("M 0 1");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[1].gate == GateType::M);
}

TEST_CASE("Parse QUBIT_COORDS silently discarded", "[parser][qec]") {
    auto circuit = parse(R"(
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1.5, 2.25) 1
        H 0
    )");

    // Only the H gate should be in the AST
    REQUIRE(circuit.nodes.size() == 1);
    REQUIRE(circuit.nodes[0].gate == GateType::H);
    REQUIRE(circuit.num_qubits == 1);
}

TEST_CASE("Parse SHIFT_COORDS silently discarded", "[parser][qec]") {
    auto circuit = parse(R"(
        H 0
        SHIFT_COORDS(0, 0, 1)
        H 1
    )");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.nodes[0].gate == GateType::H);
    REQUIRE(circuit.nodes[1].gate == GateType::H);
}

TEST_CASE("Parse DETECTOR with rec targets", "[parser][qec]") {
    auto circuit = parse(R"(
        M 0 1
        DETECTOR rec[-1] rec[-2]
    )");

    REQUIRE(circuit.nodes.size() == 3);  // M, M, DETECTOR
    REQUIRE(circuit.num_detectors == 1);

    auto& det = circuit.nodes[2];
    REQUIRE(det.gate == GateType::DETECTOR);
    REQUIRE(det.targets.size() == 2);
    REQUIRE(det.targets[0].is_rec());
    REQUIRE(det.targets[0].value() == 1);  // rec[-1] -> absolute index 1
    REQUIRE(det.targets[1].is_rec());
    REQUIRE(det.targets[1].value() == 0);  // rec[-2] -> absolute index 0
}

TEST_CASE("Parse DETECTOR with coordinates discarded", "[parser][qec]") {
    auto circuit = parse(R"(
        M 0
        DETECTOR(1.25, 0.25, 0) rec[-1]
    )");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_detectors == 1);

    auto& det = circuit.nodes[1];
    REQUIRE(det.gate == GateType::DETECTOR);
    REQUIRE(det.targets.size() == 1);
    REQUIRE(det.targets[0].value() == 0);
}

TEST_CASE("Parse OBSERVABLE_INCLUDE", "[parser][qec]") {
    auto circuit = parse(R"(
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-2]
    )");

    REQUIRE(circuit.nodes.size() == 5);     // M, M, M, OBS_INC, OBS_INC
    REQUIRE(circuit.num_observables == 3);  // max index 2 + 1

    auto& obs0 = circuit.nodes[3];
    REQUIRE(obs0.gate == GateType::OBSERVABLE_INCLUDE);
    REQUIRE(obs0.arg == 0.0);  // Observable index 0
    REQUIRE(obs0.targets.size() == 2);
    REQUIRE(obs0.targets[0].value() == 0);  // rec[-3] -> 0
    REQUIRE(obs0.targets[1].value() == 2);  // rec[-1] -> 2

    auto& obs2 = circuit.nodes[4];
    REQUIRE(obs2.arg == 2.0);  // Observable index 2
    REQUIRE(obs2.targets.size() == 1);
    REQUIRE(obs2.targets[0].value() == 1);  // rec[-2] -> 1
}

TEST_CASE("Error: R with noise argument", "[parser][noise]") {
    REQUIRE_THROWS_AS(parse("R(0.001) 0"), ParseError);
}

TEST_CASE("Error: RX with noise argument", "[parser][noise]") {
    REQUIRE_THROWS_AS(parse("RX(0.001) 0"), ParseError);
}

TEST_CASE("Error: DETECTOR with qubit target", "[parser][qec]") {
    REQUIRE_THROWS_AS(parse("M 0\nDETECTOR 0"), ParseError);
}

TEST_CASE("Error: OBSERVABLE_INCLUDE with qubit target", "[parser][qec]") {
    REQUIRE_THROWS_AS(parse("M 0\nOBSERVABLE_INCLUDE(0) 0"), ParseError);
}

TEST_CASE("Error: OBSERVABLE_INCLUDE with negative index", "[parser][qec]") {
    REQUIRE_THROWS_AS(parse("M 0\nOBSERVABLE_INCLUDE(-1) rec[-1]"), ParseError);
}

TEST_CASE("Error: OBSERVABLE_INCLUDE with non-integer index", "[parser][qec]") {
    REQUIRE_THROWS_AS(parse("M 0\nOBSERVABLE_INCLUDE(1.5) rec[-1]"), ParseError);
}

TEST_CASE("Parse representative QEC circuit", "[parser][qec][integration]") {
    // Minimal circuit exercising all Phase 2.1 parsing features:
    // - Coordinate annotations (discarded)
    // - Noise channels (X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1, DEPOLARIZE2)
    // - Noisy measurements decomposed to READOUT_NOISE
    // - DETECTOR with coordinates and rec targets
    // - OBSERVABLE_INCLUDE with observable index
    // - Classical feedback
    constexpr const char* kQecCircuit = R"(
        # Coordinate annotations (silently discarded)
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(0, 1) 3

        # Data qubit initialization with noise
        RX 0 1
        R 2 3
        X_ERROR(0.001) 2 3
        Z_ERROR(0.001) 0 1
        DEPOLARIZE1(0.001) 0 1 2 3
        TICK

        # Entangling layer with two-qubit noise
        CX 0 2 1 3
        DEPOLARIZE2(0.001) 0 2 1 3
        TICK

        # Syndrome extraction with noisy measurement
        M(0.001) 2 3
        CX rec[-2] 0 rec[-1] 1
        DETECTOR(0, 1, 0) rec[-2]
        DETECTOR(1, 1, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK

        # Second round
        R 2 3
        Y_ERROR(0.001) 2 3
        CX 0 2 1 3
        DEPOLARIZE2(0.001) 0 2 1 3
        MX(0.001) 0 1
        M(0.001) 2 3

        # Final detectors and observable
        DETECTOR(0, 1, 1) rec[-4] rec[-2]
        DETECTOR(1, 1, 1) rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-4] rec[-3]
    )";

    auto circuit = parse(kQecCircuit);

    // Verify basic circuit properties
    REQUIRE(circuit.num_qubits == 4);
    REQUIRE(circuit.num_detectors == 4);
    REQUIRE(circuit.num_observables == 1);

    // Count gate types
    size_t noise_count = 0;
    size_t detector_count = 0;
    size_t observable_count = 0;
    size_t readout_noise_count = 0;
    size_t tick_count = 0;

    for (const auto& node : circuit.nodes) {
        switch (node.gate) {
            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR:
            case GateType::DEPOLARIZE1:
            case GateType::DEPOLARIZE2:
                noise_count++;
                break;
            case GateType::DETECTOR:
                detector_count++;
                break;
            case GateType::OBSERVABLE_INCLUDE:
                observable_count++;
                break;
            case GateType::READOUT_NOISE:
                readout_noise_count++;
                break;
            case GateType::TICK:
                tick_count++;
                break;
            default:
                break;
        }
    }

    // Verify noise gates parsed:
    // X_ERROR(2,3)=2 + Z_ERROR(0,1)=2 + DEP1(0,1,2,3)=4 + DEP2(0-2,1-3)=2
    // + Y_ERROR(2,3)=2 + DEP2(0-2,1-3)=2 = 14
    REQUIRE(noise_count == 14);
    REQUIRE(detector_count == 4);
    REQUIRE(observable_count == 1);
    // 6 noisy measurements: M(0.001) 2 3, MX(0.001) 0 1, M(0.001) 2 3
    REQUIRE(readout_noise_count == 6);
    REQUIRE(tick_count == 3);

    // Verify QUBIT_COORDS and SHIFT_COORDS were discarded (no nodes for them)
    for (const auto& node : circuit.nodes) {
        // These gate types don't exist - coords are fully discarded
        REQUIRE(node.gate != GateType::UNKNOWN);
    }
}
