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

TEST_CASE("Parse MR and MRX", "[parser]") {
    auto circuit = parse(R"(
        MR 0
        MRX 1
    )");

    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_measurements == 2);

    REQUIRE(circuit.nodes[0].gate == GateType::MR);
    REQUIRE(circuit.nodes[1].gate == GateType::MRX);
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

TEST_CASE("Parse reset decomposition R", "[parser]") {
    auto circuit = parse("R 0");

    // R 0 -> M 0; CX rec[-1] 0
    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_measurements == 1);

    // First: M 0
    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[0].targets.size() == 1);
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(!circuit.nodes[0].targets[0].is_rec());

    // Second: CX rec[0] 0 (rec[-1] resolved to absolute index 0)
    REQUIRE(circuit.nodes[1].gate == GateType::CX);
    REQUIRE(circuit.nodes[1].targets.size() == 2);
    REQUIRE(circuit.nodes[1].targets[0].is_rec());
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);
    REQUIRE(!circuit.nodes[1].targets[1].is_rec());
    REQUIRE(circuit.nodes[1].targets[1].value() == 0);
}

TEST_CASE("Parse reset decomposition RX", "[parser]") {
    auto circuit = parse("RX 0");

    // RX 0 -> MX 0; CZ rec[-1] 0
    REQUIRE(circuit.nodes.size() == 2);
    REQUIRE(circuit.num_measurements == 1);

    REQUIRE(circuit.nodes[0].gate == GateType::MX);
    REQUIRE(circuit.nodes[1].gate == GateType::CZ);
    REQUIRE(circuit.nodes[1].targets[0].is_rec());
}

TEST_CASE("Parse multiple resets", "[parser]") {
    auto circuit = parse("R 0 1 2");

    // 3 resets -> 6 nodes (M + CX for each)
    REQUIRE(circuit.nodes.size() == 6);
    REQUIRE(circuit.num_measurements == 3);

    // Check structure: M, CX, M, CX, M, CX
    REQUIRE(circuit.nodes[0].gate == GateType::M);
    REQUIRE(circuit.nodes[1].gate == GateType::CX);
    REQUIRE(circuit.nodes[2].gate == GateType::M);
    REQUIRE(circuit.nodes[3].gate == GateType::CX);
    REQUIRE(circuit.nodes[4].gate == GateType::M);
    REQUIRE(circuit.nodes[5].gate == GateType::CX);

    // Verify rec references point to the immediately preceding measurement.
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // rec[-1] -> 0
    REQUIRE(circuit.nodes[3].targets[0].value() == 1);  // rec[-1] -> 1
    REQUIRE(circuit.nodes[5].targets[0].value() == 2);  // rec[-1] -> 2
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

    // 3 resets (6 nodes) + H + CX + CX + H + M = 11 nodes
    REQUIRE(circuit.nodes.size() == 11);
    REQUIRE(circuit.num_qubits == 3);
    // 3 from resets + 1 final measurement = 4
    REQUIRE(circuit.num_measurements == 4);
}
