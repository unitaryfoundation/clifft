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
#include "ucc/circuit/gate_data.h"
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

    // First node: clean MPP with empty args (no heap alloc for 0.0)
    REQUIRE(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].args.empty());
    REQUIRE(circuit.nodes[0].targets.size() == 2);

    // Second node: READOUT_NOISE with original probability
    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].args[0] == 0.001);
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
    CHECK(circuit.nodes[0].args.empty());
    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].args[0] == 0.002);
    REQUIRE(circuit.nodes[1].targets[0].value() == 0);  // meas index 0

    // Second product: MPP + READOUT_NOISE
    REQUIRE(circuit.nodes[2].gate == GateType::MPP);
    CHECK(circuit.nodes[2].args.empty());
    REQUIRE(circuit.nodes[3].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[3].args[0] == 0.002);
    REQUIRE(circuit.nodes[3].targets[0].value() == 1);  // meas index 1
}

TEST_CASE("Parse MPP rejects duplicate qubit in product", "[parser]") {
    // X0*Z0 has qubit 0 appearing twice which causes silent phase loss
    CHECK_THROWS_AS(parse("MPP X0*Z0"), ParseError);
    CHECK_THROWS_AS(parse("MPP Y1*X1"), ParseError);
    // Different qubits in same product is fine
    CHECK_NOTHROW(parse("MPP X0*Z1"));
    // Same qubit in different products is fine
    CHECK_NOTHROW(parse("MPP X0 Z0"));
    // High qubit indices must not cause excessive memory allocation
    CHECK_NOTHROW(parse("MPP X100000*Z200000"));
    CHECK_THROWS_AS(parse("MPP X100000*Z100000"), ParseError);
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
    REQUIRE(circuit.nodes[0].args[0] == 0.5);
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

TEST_CASE("REPEAT basic unrolling", "[parser]") {
    auto c = parse("REPEAT 3 {\nH 0\n}");
    REQUIRE(c.nodes.size() == 3);
    for (auto& n : c.nodes) {
        REQUIRE(n.gate == GateType::H);
        REQUIRE(n.targets.size() == 1);
        REQUIRE(n.targets[0].value() == 0);
    }
    REQUIRE(c.num_qubits == 1);
}

TEST_CASE("REPEAT with measurements and rec references", "[parser]") {
    // M 0; REPEAT 3 { CX rec[-1] 1; M 1 }
    // Should produce 7 nodes: M, (CX, M) x 3
    // CX rec refs should point to measurements 0, 1, 2 respectively.
    auto c = parse("M 0\nREPEAT 3 {\nCX rec[-1] 1\nM 1\n}");
    REQUIRE(c.nodes.size() == 7);
    REQUIRE(c.nodes[0].gate == GateType::M);  // M 0 -> meas #0

    // Iteration 1: CX rec[-1] 1 -> rec[0], M 1 -> meas #1
    REQUIRE(c.nodes[1].gate == GateType::CX);
    REQUIRE(c.nodes[1].targets[0].is_rec());
    REQUIRE(c.nodes[1].targets[0].value() == 0);  // points to meas #0
    REQUIRE(c.nodes[2].gate == GateType::M);

    // Iteration 2: CX rec[-1] 1 -> rec[1], M 1 -> meas #2
    REQUIRE(c.nodes[3].gate == GateType::CX);
    REQUIRE(c.nodes[3].targets[0].value() == 1);  // points to meas #1
    REQUIRE(c.nodes[4].gate == GateType::M);

    // Iteration 3: CX rec[-1] 1 -> rec[2], M 1 -> meas #3
    REQUIRE(c.nodes[5].gate == GateType::CX);
    REQUIRE(c.nodes[5].targets[0].value() == 2);  // points to meas #2
    REQUIRE(c.nodes[6].gate == GateType::M);

    REQUIRE(c.num_measurements == 4);
}

TEST_CASE("REPEAT nested", "[parser]") {
    // REPEAT 2 { REPEAT 3 { H 0 } }
    auto c = parse("REPEAT 2 {\nREPEAT 3 {\nH 0\n}\n}");
    REQUIRE(c.nodes.size() == 6);  // 2 * 3 = 6
    for (auto& n : c.nodes) {
        REQUIRE(n.gate == GateType::H);
    }
}

TEST_CASE("REPEAT safety limit", "[parser]") {
    // Use a small custom limit to trigger quickly.
    // REPEAT 10 { H 0 } = 10 nodes, exceeds limit of 5.
    REQUIRE_THROWS_AS(parse("REPEAT 10 {\nH 0\n}", 5), ParseError);

    // Just under the limit should succeed.
    auto c = parse("REPEAT 5 {\nH 0\n}", 5);
    REQUIRE(c.nodes.size() == 5);
}

TEST_CASE("REPEAT with empty body completes instantly", "[parser]") {
    // A massive empty repeat must not spin for billions of iterations.
    auto c1 = parse("REPEAT 999999999 {\n}\nH 0");
    CHECK(c1.nodes.size() == 1);  // only the H

    // Body with only comments is also empty
    auto c2 = parse("REPEAT 999999999 {\n# just a comment\n}\nH 0");
    CHECK(c2.nodes.size() == 1);

    // Body with whitespace only
    auto c3 = parse("REPEAT 999999999 {\n   \n\t\n}\nH 0");
    CHECK(c3.nodes.size() == 1);

    // Non-empty body still works
    auto c4 = parse("REPEAT 3 {\nH 0\n}");
    CHECK(c4.nodes.size() == 3);
}

TEST_CASE("REPEAT error: missing brace", "[parser]") {
    REQUIRE_THROWS_AS(parse("REPEAT 3\nH 0"), ParseError);
}

TEST_CASE("REPEAT error: max recursion depth exceeded", "[parser]") {
    // Build a circuit with 101 levels of nesting
    std::string text;
    for (int i = 0; i < 101; i++)
        text += "REPEAT 1 {\n";
    text += "H 0\n";
    for (int i = 0; i < 101; i++)
        text += "}\n";
    CHECK_THROWS_AS(parse(text), ParseError);

    // 100 levels should be fine
    std::string ok;
    for (int i = 0; i < 100; i++)
        ok += "REPEAT 1 {\n";
    ok += "H 0\n";
    for (int i = 0; i < 100; i++)
        ok += "}\n";
    CHECK_NOTHROW(parse(ok));
}

TEST_CASE("REPEAT error: zero count", "[parser]") {
    REQUIRE_THROWS_AS(parse("REPEAT 0 {\nH 0\n}"), ParseError);
}

TEST_CASE("REPEAT error: missing closing brace", "[parser]") {
    REQUIRE_THROWS_AS(parse("REPEAT 3 {\nH 0\n"), ParseError);
}

TEST_CASE("REPEAT with multiple gates per iteration", "[parser]") {
    auto c = parse("REPEAT 2 {\nH 0\nCX 0 1\nM 0 1\n}");
    // Each iteration: H, CX, M(0), M(1) = 4 nodes
    REQUIRE(c.nodes.size() == 8);
    REQUIRE(c.num_measurements == 4);
    REQUIRE(c.num_qubits == 2);
}

TEST_CASE("REPEAT ignores braces inside comments", "[parser]") {
    // The closing brace in the comment should not terminate the block.
    auto c = parse("REPEAT 2 {\nH 0 # tricky }\nM 0\n}");
    REQUIRE(c.nodes.size() == 4);  // 2 * (H + M)
    REQUIRE(c.num_measurements == 2);
}

TEST_CASE("REPEAT line numbers after block are correct", "[parser]") {
    // Syntax error on line 5 should report correct line number.
    try {
        (void)parse("H 0\nREPEAT 2 {\nH 0\n}\nBADGATE 0");
        REQUIRE(false);  // Should not reach here.
    } catch (const ParseError& e) {
        // Line 5 is where BADGATE is.
        CHECK(e.line() == 5);
    }
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
// Noise and QEC gate parsing tests
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
    REQUIRE(circuit.nodes[0].args[0] == Catch::Approx(0.001));
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[3].gate == GateType::Y_ERROR);
    REQUIRE(circuit.nodes[3].args[0] == Catch::Approx(0.002));

    REQUIRE(circuit.nodes[4].gate == GateType::Z_ERROR);
    REQUIRE(circuit.nodes[4].args[0] == Catch::Approx(0.003));
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
    REQUIRE(circuit.nodes[0].args[0] == Catch::Approx(0.01));

    REQUIRE(circuit.nodes[2].gate == GateType::DEPOLARIZE2);
    REQUIRE(circuit.nodes[2].args[0] == Catch::Approx(0.02));
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
    REQUIRE(circuit.nodes[0].args[0] == 0.0);  // Clean measurement
    REQUIRE(circuit.nodes[0].targets[0].value() == 0);

    REQUIRE(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[1].args[0] == Catch::Approx(0.001));
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
    REQUIRE(circuit.nodes[1].args[0] == Catch::Approx(0.003));

    REQUIRE(circuit.nodes[2].gate == GateType::MY);
    REQUIRE(circuit.nodes[3].gate == GateType::READOUT_NOISE);
    REQUIRE(circuit.nodes[3].args[0] == Catch::Approx(0.004));
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
    REQUIRE(obs0.args[0] == 0.0);  // Observable index 0
    REQUIRE(obs0.targets.size() == 2);
    REQUIRE(obs0.targets[0].value() == 0);  // rec[-3] -> 0
    REQUIRE(obs0.targets[1].value() == 2);  // rec[-1] -> 2

    auto& obs2 = circuit.nodes[4];
    REQUIRE(obs2.args[0] == 2.0);  // Observable index 2
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
    // Minimal circuit exercising all noise and QEC parsing features:
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

// --- Phase 1: Aliases, No-Ops, and MPAD ---

TEST_CASE("Parse Stim aliases for single-qubit gates", "[parser]") {
    auto circuit = parse("H_XZ 0\nSQRT_Z 1\nSQRT_Z_DAG 2");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].gate == GateType::H);
    CHECK(circuit.nodes[1].gate == GateType::S);
    CHECK(circuit.nodes[2].gate == GateType::S_DAG);
}

TEST_CASE("Parse Stim aliases for two-qubit gates", "[parser]") {
    auto circuit = parse("ZCX 0 1\nZCY 2 3\nZCZ 4 5");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].gate == GateType::CX);
    CHECK(circuit.nodes[1].gate == GateType::CY);
    CHECK(circuit.nodes[2].gate == GateType::CZ);
    CHECK(circuit.num_qubits == 6);
}

TEST_CASE("Parse Stim aliases for measurements and resets", "[parser]") {
    auto circuit = parse("MZ 0\nMRZ 1\nRZ 2");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].gate == GateType::M);
    CHECK(circuit.nodes[1].gate == GateType::MR);
    CHECK(circuit.nodes[2].gate == GateType::R);
    CHECK(circuit.num_measurements == 2);
}

TEST_CASE("Parse identity no-ops - I and II", "[parser]") {
    auto circuit = parse("I 0 1 2\nII 3 4 5 6");
    CHECK(circuit.nodes.empty());
    CHECK(circuit.num_qubits == 7);
}

TEST_CASE("Parse identity no-ops - I_ERROR and II_ERROR", "[parser]") {
    auto circuit = parse("I_ERROR(0.01) 0\nII_ERROR(0.02) 1 2");
    CHECK(circuit.nodes.empty());
    CHECK(circuit.num_qubits == 3);
}

TEST_CASE("Parse II with odd targets fails", "[parser]") {
    REQUIRE_THROWS_AS(parse("II 0 1 2"), ParseError);
}

TEST_CASE("Parse I updates num_qubits for large indices", "[parser]") {
    auto circuit = parse("I 49");
    CHECK(circuit.nodes.empty());
    CHECK(circuit.num_qubits == 50);
}

TEST_CASE("Parse MPAD with valid targets", "[parser]") {
    auto circuit = parse("MPAD 1 0 1");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].gate == GateType::MPAD);
    CHECK(circuit.nodes[0].targets[0].value() == 1);
    CHECK(circuit.nodes[1].targets[0].value() == 0);
    CHECK(circuit.nodes[2].targets[0].value() == 1);
    CHECK(circuit.num_measurements == 3);
    CHECK(circuit.num_qubits == 0);
}

TEST_CASE("Parse MPAD with invalid target fails", "[parser]") {
    REQUIRE_THROWS_AS(parse("MPAD 2"), ParseError);
}

TEST_CASE("Parse MPAD does not affect num_qubits", "[parser]") {
    auto circuit = parse("H 3\nMPAD 1 0");
    CHECK(circuit.num_qubits == 4);
    CHECK(circuit.num_measurements == 2);
}

TEST_CASE("Parse noisy MPAD decomposes to MPAD plus READOUT_NOISE", "[parser]") {
    auto circuit = parse("MPAD(0.01) 1");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].gate == GateType::MPAD);
    CHECK(circuit.nodes[1].gate == GateType::READOUT_NOISE);
}

TEST_CASE("Parse MPAD rejects rec targets", "[parser]") {
    // Requires M first so rec[-1] resolves.
    REQUIRE_THROWS_AS(parse("M 0\nMPAD rec[-1]"), ParseError);
}

// --- Phase 2: Pair Measurements and Y-Resets ---

TEST_CASE("Parse MXX desugars to MPP with X-tagged targets", "[parser]") {
    auto circuit = parse("MXX 0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].targets.size() == 2);
    CHECK(circuit.nodes[0].targets[0].pauli() == Target::kPauliX);
    CHECK(circuit.nodes[0].targets[0].value() == 0);
    CHECK(circuit.nodes[0].targets[1].pauli() == Target::kPauliX);
    CHECK(circuit.nodes[0].targets[1].value() == 1);
    CHECK(circuit.num_measurements == 1);
}

TEST_CASE("Parse MYY desugars to MPP with Y-tagged targets", "[parser]") {
    auto circuit = parse("MYY 2 3");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].targets[0].pauli() == Target::kPauliY);
    CHECK(circuit.nodes[0].targets[1].pauli() == Target::kPauliY);
}

TEST_CASE("Parse MZZ desugars to MPP with Z-tagged targets", "[parser]") {
    auto circuit = parse("MZZ 0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].targets[0].pauli() == Target::kPauliZ);
    CHECK(circuit.nodes[0].targets[1].pauli() == Target::kPauliZ);
}

TEST_CASE("Parse MXX with multiple pairs", "[parser]") {
    auto circuit = parse("MXX 0 1 2 3");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[1].gate == GateType::MPP);
    CHECK(circuit.num_measurements == 2);
}

TEST_CASE("Parse MXX odd targets fails", "[parser]") {
    REQUIRE_THROWS_AS(parse("MXX 0 1 2"), ParseError);
}

TEST_CASE("Parse MXX preserves inversion flags", "[parser]") {
    auto circuit = parse("MXX !0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].targets[0].is_inverted());
    CHECK(!circuit.nodes[0].targets[1].is_inverted());
}

TEST_CASE("Parse RY and MRY", "[parser]") {
    auto circuit = parse("RY 0\nMRY 1");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].gate == GateType::RY);
    CHECK(circuit.nodes[1].gate == GateType::MRY);
    CHECK(circuit.num_measurements == 1);
    CHECK(circuit.num_qubits == 2);
}

// --- Phase 4: Multi-Parameter Noise ---

TEST_CASE("Parse multi-arg parenthesized arguments", "[parser]") {
    auto circuit = parse("PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::PAULI_CHANNEL_1);
    REQUIRE(circuit.nodes[0].args.size() == 3);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.1));
    CHECK(circuit.nodes[0].args[1] == Catch::Approx(0.2));
    CHECK(circuit.nodes[0].args[2] == Catch::Approx(0.3));
}

TEST_CASE("Parse PAULI_CHANNEL_2 with 15 args", "[parser]") {
    auto circuit = parse(
        "PAULI_CHANNEL_2(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, "
        "0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15) 0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::PAULI_CHANNEL_2);
    REQUIRE(circuit.nodes[0].args.size() == 15);
    CHECK(circuit.nodes[0].args[14] == Catch::Approx(0.15));
    CHECK(circuit.num_qubits == 2);
}

TEST_CASE("Parse PAULI_CHANNEL_1 broadcasts to multiple targets", "[parser]") {
    auto circuit = parse("PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0 1 2");
    REQUIRE(circuit.nodes.size() == 3);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(circuit.nodes[i].args.size() == 3);
        CHECK(circuit.nodes[i].args[0] == Catch::Approx(0.1));
    }
}

// --- Review Round 2 edge cases ---

TEST_CASE("REPEAT opening brace in comment is ignored", "[parser]") {
    // The '{' in the comment should not be mistaken for the opening brace.
    auto c = parse("REPEAT 2\n# wait for {\n{\nH 0\n}");
    REQUIRE(c.nodes.size() == 2);
    CHECK(c.nodes[0].gate == GateType::H);
    CHECK(c.nodes[1].gate == GateType::H);
}

TEST_CASE("REPEAT inline comment after opening brace is allowed", "[parser]") {
    auto c = parse("REPEAT 2 { # start loop\nH 0\n}");
    REQUIRE(c.nodes.size() == 2);
    CHECK(c.nodes[0].gate == GateType::H);
}

TEST_CASE("DETECTOR emits empty args vector", "[parser]") {
    auto circuit = parse("M 0\nDETECTOR rec[-1]");
    // The DETECTOR node should have an empty args vector (no heap alloc for 0.0).
    auto& det_node = circuit.nodes.back();
    CHECK(det_node.gate == GateType::DETECTOR);
    CHECK(det_node.args.empty());
}

TEST_CASE("Clean MXX emits empty args vector", "[parser]") {
    auto circuit = parse("MXX 0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].args.empty());
}

TEST_CASE("Clean MPP emits empty args vector", "[parser]") {
    auto circuit = parse("MPP X0*Z1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::MPP);
    CHECK(circuit.nodes[0].args.empty());
}

TEST_CASE("PAULI_CHANNEL_1 rejects wrong arg count", "[parser]") {
    REQUIRE_THROWS_AS(parse("PAULI_CHANNEL_1(0.1, 0.2) 0"), ParseError);
    REQUIRE_THROWS_AS(parse("PAULI_CHANNEL_1(0.1) 0"), ParseError);
    REQUIRE_THROWS_AS(parse("PAULI_CHANNEL_1(0.1, 0.2, 0.3, 0.4) 0"), ParseError);
}

TEST_CASE("PAULI_CHANNEL_2 rejects wrong arg count", "[parser]") {
    REQUIRE_THROWS_AS(parse("PAULI_CHANNEL_2(0.01, 0.02, 0.03) 0 1"), ParseError);
}

// =============================================================================
// Source Line Tracking
// =============================================================================

TEST_CASE("Source line tracking for simple gates", "[parser][source_line]") {
    auto circuit = parse("H 0\nT 1\nM 0");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 2);
    CHECK(circuit.nodes[2].source_line == 3);
}

TEST_CASE("Source line tracking with blank lines and comments", "[parser][source_line]") {
    auto circuit = parse("H 0\n\n# comment\nT 1\nM 0");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 4);
    CHECK(circuit.nodes[2].source_line == 5);
}

TEST_CASE("Source line tracking with multi-target expansion", "[parser][source_line]") {
    // "H 0 1 2" on one line expands to 3 AstNodes, all from line 1
    auto circuit = parse("H 0 1 2");
    REQUIRE(circuit.nodes.size() == 3);
    for (auto& node : circuit.nodes) {
        CHECK(node.source_line == 1);
    }
}

TEST_CASE("Source line tracking with REPEAT block", "[parser][source_line]") {
    // Line 1: REPEAT 2 {
    // Line 2: H 0
    // Line 3: T 0
    // Line 4: }
    // Line 5: M 0
    auto circuit = parse("REPEAT 2 {\nH 0\nT 0\n}\nM 0");
    // 2 iterations of (H, T) + M = 5 nodes
    REQUIRE(circuit.nodes.size() == 5);
    // First iteration: lines map back to original block
    CHECK(circuit.nodes[0].source_line == 2);
    CHECK(circuit.nodes[1].source_line == 3);
    // Second iteration: same source lines
    CHECK(circuit.nodes[2].source_line == 2);
    CHECK(circuit.nodes[3].source_line == 3);
    // M after the block
    CHECK(circuit.nodes[4].source_line == 5);
}

TEST_CASE("Source line tracking for two-qubit gates", "[parser][source_line]") {
    auto circuit = parse("CX 0 1\nCZ 2 3");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 2);
}

TEST_CASE("Source line tracking for MPP", "[parser][source_line]") {
    auto circuit = parse("MPP X0*Z1 Y2");
    // MPP with two products -> 2 AstNodes, both from line 1
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 1);
}

TEST_CASE("Source line tracking for DETECTOR and OBSERVABLE_INCLUDE", "[parser][source_line]") {
    auto circuit = parse("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 2);
    CHECK(circuit.nodes[2].source_line == 3);
}

TEST_CASE("Source line tracking for noisy measurement decomposition", "[parser][source_line]") {
    // M(0.01) 0 decomposes into M + READOUT_NOISE, both from same line
    auto circuit = parse("M(0.01) 0");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].gate == GateType::M);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].gate == GateType::READOUT_NOISE);
    CHECK(circuit.nodes[1].source_line == 1);
}

TEST_CASE("Source line tracking for TICK", "[parser][source_line]") {
    auto circuit = parse("H 0\nTICK\nM 0");
    REQUIRE(circuit.nodes.size() == 3);
    CHECK(circuit.nodes[0].source_line == 1);
    CHECK(circuit.nodes[1].source_line == 2);
    CHECK(circuit.nodes[2].source_line == 3);
}

// =========================================================================
// GateTraits lookup table
// =========================================================================

TEST_CASE("GateTraits: spot-check single-qubit Cliffords", "[gate_data]") {
    CHECK(gate_arity(GateType::H) == GateArity::SINGLE);
    CHECK(is_clifford(GateType::H));
    CHECK(!is_measurement(GateType::H));
    CHECK(!is_noise_gate(GateType::H));
    CHECK(gate_name(GateType::H) == "H");

    CHECK(is_clifford(GateType::S));
    CHECK(is_clifford(GateType::S_DAG));
    CHECK(is_clifford(GateType::X));
    CHECK(is_clifford(GateType::Y));
    CHECK(is_clifford(GateType::Z));
    CHECK(is_clifford(GateType::SQRT_X));
    CHECK(is_clifford(GateType::C_XYZ));
    CHECK(is_clifford(GateType::C_ZYNX));
}

TEST_CASE("GateTraits: non-Clifford T gates", "[gate_data]") {
    CHECK(!is_clifford(GateType::T));
    CHECK(!is_clifford(GateType::T_DAG));
    CHECK(gate_arity(GateType::T) == GateArity::SINGLE);
    CHECK(gate_name(GateType::T) == "T");
    CHECK(gate_name(GateType::T_DAG) == "T_DAG");
}

TEST_CASE("GateTraits: two-qubit Cliffords are PAIR arity", "[gate_data]") {
    CHECK(gate_arity(GateType::CX) == GateArity::PAIR);
    CHECK(gate_arity(GateType::CZ) == GateArity::PAIR);
    CHECK(gate_arity(GateType::SWAP) == GateArity::PAIR);
    CHECK(gate_arity(GateType::ISWAP) == GateArity::PAIR);
    CHECK(gate_arity(GateType::YCZ) == GateArity::PAIR);
    CHECK(is_clifford(GateType::CX));
    CHECK(is_clifford(GateType::YCZ));
}

TEST_CASE("GateTraits: measurements", "[gate_data]") {
    CHECK(is_measurement(GateType::M));
    CHECK(is_measurement(GateType::MX));
    CHECK(is_measurement(GateType::MY));
    CHECK(is_measurement(GateType::MR));
    CHECK(is_measurement(GateType::MRX));
    CHECK(is_measurement(GateType::MRY));
    CHECK(is_measurement(GateType::MPP));
    CHECK(is_measurement(GateType::MPAD));
    CHECK(!is_measurement(GateType::H));
    CHECK(!is_measurement(GateType::R));
}

TEST_CASE("GateTraits: measure-reset subset", "[gate_data]") {
    CHECK(is_measure_reset(GateType::MR));
    CHECK(is_measure_reset(GateType::MRX));
    CHECK(is_measure_reset(GateType::MRY));
    CHECK(!is_measure_reset(GateType::M));
    CHECK(!is_measure_reset(GateType::R));
}

TEST_CASE("GateTraits: resets", "[gate_data]") {
    CHECK(is_reset(GateType::R));
    CHECK(is_reset(GateType::RX));
    CHECK(is_reset(GateType::RY));
    CHECK(!is_reset(GateType::MR));
    CHECK(!is_reset(GateType::H));
}

TEST_CASE("GateTraits: identity no-ops", "[gate_data]") {
    CHECK(is_identity_noop(GateType::I));
    CHECK(is_identity_noop(GateType::II));
    CHECK(is_identity_noop(GateType::I_ERROR));
    CHECK(is_identity_noop(GateType::II_ERROR));
    CHECK(!is_identity_noop(GateType::H));
    CHECK(gate_arity(GateType::II) == GateArity::PAIR);
    CHECK(gate_arity(GateType::II_ERROR) == GateArity::PAIR);
}

TEST_CASE("GateTraits: noise channels", "[gate_data]") {
    CHECK(is_noise_gate(GateType::X_ERROR));
    CHECK(is_noise_gate(GateType::Y_ERROR));
    CHECK(is_noise_gate(GateType::Z_ERROR));
    CHECK(is_noise_gate(GateType::DEPOLARIZE1));
    CHECK(is_noise_gate(GateType::DEPOLARIZE2));
    CHECK(is_noise_gate(GateType::PAULI_CHANNEL_1));
    CHECK(is_noise_gate(GateType::PAULI_CHANNEL_2));
    CHECK(is_noise_gate(GateType::READOUT_NOISE));
    CHECK(!is_noise_gate(GateType::M));
    CHECK(gate_arity(GateType::DEPOLARIZE2) == GateArity::PAIR);
    CHECK(gate_arity(GateType::PAULI_CHANNEL_2) == GateArity::PAIR);
}

TEST_CASE("GateTraits: annotations are ANNOTATION arity", "[gate_data]") {
    CHECK(gate_arity(GateType::TICK) == GateArity::ANNOTATION);
    CHECK(gate_arity(GateType::DETECTOR) == GateArity::ANNOTATION);
    CHECK(gate_arity(GateType::OBSERVABLE_INCLUDE) == GateArity::ANNOTATION);
    CHECK(gate_name(GateType::TICK) == "TICK");
}

TEST_CASE("GateTraits: MPP is MULTI arity", "[gate_data]") {
    CHECK(gate_arity(GateType::MPP) == GateArity::MULTI);
    CHECK(is_measurement(GateType::MPP));
    CHECK(!is_clifford(GateType::MPP));
}

TEST_CASE("GateTraits: pair measurements MXX MYY MZZ", "[gate_data]") {
    CHECK(gate_arity(GateType::MXX) == GateArity::PAIR);
    CHECK(gate_arity(GateType::MYY) == GateArity::PAIR);
    CHECK(gate_arity(GateType::MZZ) == GateArity::PAIR);
    CHECK(is_measurement(GateType::MXX));
}

TEST_CASE("GateTraits: UNKNOWN sentinel", "[gate_data]") {
    CHECK(gate_arity(GateType::UNKNOWN) == GateArity::SINGLE);
    CHECK(!is_clifford(GateType::UNKNOWN));
    CHECK(!is_measurement(GateType::UNKNOWN));
    CHECK(!is_noise_gate(GateType::UNKNOWN));
    CHECK(gate_name(GateType::UNKNOWN) == "UNKNOWN");
}

// =============================================================================
// Parameterized rotation gate parsing
// =============================================================================

TEST_CASE("Parse R_Z with single arg", "[parser][rotation]") {
    auto circuit = parse("R_Z(0.25) 0");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::R_Z);
    CHECK(circuit.nodes[0].targets[0].value() == 0);
    REQUIRE(circuit.nodes[0].args.size() == 1);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.25));
}

TEST_CASE("Parse R_X and R_Y", "[parser][rotation]") {
    auto circuit = parse("R_X(0.5) 0\nR_Y(1.0) 1");
    REQUIRE(circuit.nodes.size() == 2);
    CHECK(circuit.nodes[0].gate == GateType::R_X);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.5));
    CHECK(circuit.nodes[1].gate == GateType::R_Y);
    CHECK(circuit.nodes[1].args[0] == Catch::Approx(1.0));
}

TEST_CASE("Parse U3 with three args", "[parser][rotation]") {
    auto circuit = parse("U3(0.5, 0.25, 0.75) 0");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::U3);
    REQUIRE(circuit.nodes[0].args.size() == 3);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.5));
    CHECK(circuit.nodes[0].args[1] == Catch::Approx(0.25));
    CHECK(circuit.nodes[0].args[2] == Catch::Approx(0.75));
}

TEST_CASE("Parse R_ZZ pair gate", "[parser][rotation]") {
    auto circuit = parse("R_ZZ(0.3) 0 1");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::R_ZZ);
    REQUIRE(circuit.nodes[0].targets.size() == 2);
    CHECK(circuit.nodes[0].targets[0].value() == 0);
    CHECK(circuit.nodes[0].targets[1].value() == 1);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.3));
}

TEST_CASE("Parse R_PAULI with Pauli product", "[parser][rotation]") {
    auto circuit = parse("R_PAULI(0.1) X0*Y1*Z2");
    REQUIRE(circuit.nodes.size() == 1);
    CHECK(circuit.nodes[0].gate == GateType::R_PAULI);
    REQUIRE(circuit.nodes[0].targets.size() == 3);
    CHECK(circuit.nodes[0].targets[0].pauli() == Target::kPauliX);
    CHECK(circuit.nodes[0].targets[0].value() == 0);
    CHECK(circuit.nodes[0].targets[1].pauli() == Target::kPauliY);
    CHECK(circuit.nodes[0].targets[1].value() == 1);
    CHECK(circuit.nodes[0].targets[2].pauli() == Target::kPauliZ);
    CHECK(circuit.nodes[0].targets[2].value() == 2);
    CHECK(circuit.nodes[0].args[0] == Catch::Approx(0.1));
}

TEST_CASE("Parse R_Z broadcasts across targets", "[parser][rotation]") {
    auto circuit = parse("R_Z(0.5) 0 1 2");
    REQUIRE(circuit.nodes.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(circuit.nodes[i].gate == GateType::R_Z);
        CHECK(circuit.nodes[i].targets[0].value() == i);
        CHECK(circuit.nodes[i].args[0] == Catch::Approx(0.5));
    }
}

TEST_CASE("Parse R_Z missing arg is error", "[parser][rotation]") {
    CHECK_THROWS_AS(parse("R_Z 0"), ParseError);
}

TEST_CASE("Parse U3 wrong arg count is error", "[parser][rotation]") {
    CHECK_THROWS_AS(parse("U3(0.5) 0"), ParseError);
    CHECK_THROWS_AS(parse("U3(0.5, 0.25) 0"), ParseError);
}

TEST_CASE("Parse R_PAULI missing product is error", "[parser][rotation]") {
    CHECK_THROWS_AS(parse("R_PAULI(0.1)"), ParseError);
}

TEST_CASE("GateTraits: rotation gate arities", "[gate_data][rotation]") {
    CHECK(gate_arity(GateType::R_X) == GateArity::SINGLE);
    CHECK(gate_arity(GateType::R_Y) == GateArity::SINGLE);
    CHECK(gate_arity(GateType::R_Z) == GateArity::SINGLE);
    CHECK(gate_arity(GateType::U3) == GateArity::SINGLE);
    CHECK(gate_arity(GateType::R_XX) == GateArity::PAIR);
    CHECK(gate_arity(GateType::R_YY) == GateArity::PAIR);
    CHECK(gate_arity(GateType::R_ZZ) == GateArity::PAIR);
    CHECK(gate_arity(GateType::R_PAULI) == GateArity::MULTI);
    CHECK(!is_clifford(GateType::R_Z));
    CHECK(!is_measurement(GateType::R_Z));
}
