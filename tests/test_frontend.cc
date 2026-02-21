// Front-End unit tests
//
// Tests the trace() function which converts Circuit -> HirModule
// Key invariant: Clifford gates are absorbed, T gates emit HeisenbergOps
// with correctly rewound Pauli masks.

#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using namespace ucc;

// Convenience helpers for readable mask assertions
inline uint64_t X(size_t q) {
    return 1ULL << q;
}
inline uint64_t Z(size_t q) {
    return 1ULL << q;
}

TEST_CASE("Frontend: identity circuit produces empty HIR", "[frontend]") {
    auto circuit = parse("TICK");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 0);
    REQUIRE(hir.num_qubits == 0);
}

TEST_CASE("Frontend: pure Clifford circuit produces empty HIR", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        S 0
        CX 0 1
        H 1
    )");
    auto hir = trace(circuit);

    // All Cliffords absorbed - no HIR ops
    REQUIRE(hir.num_ops() == 0);
    REQUIRE(hir.num_qubits == 2);
}

TEST_CASE("Frontend: single T gate on qubit 0", "[frontend]") {
    auto circuit = parse("T 0");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);

    // Without any Cliffords, rewound Z0 is just Z0
    REQUIRE(hir.ops[0].destab_mask() == 0);   // No X
    REQUIRE(hir.ops[0].stab_mask() == Z(0));  // Z on qubit 0
    REQUIRE(hir.ops[0].sign() == false);
}

TEST_CASE("Frontend: H then T - rewound Z becomes X", "[frontend]") {
    // This is the key test from the MVP plan DoD:
    // "H 0; T 0" should emit HIR with mask corresponding to +X axis
    auto circuit = parse(R"(
        H 0
        T 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);

    // After H, the Z observable is conjugated to X
    // So the T gate's rewound Z is X on qubit 0
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
    REQUIRE(hir.ops[0].stab_mask() == 0);       // No Z
    REQUIRE(hir.ops[0].sign() == false);
}

TEST_CASE("Frontend: H; S; T - rewound Z is still X (S commutes with Z)", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        S 0
        T 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);

    // S commutes with Z, so rewound Z after H;S is still X
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
    REQUIRE(hir.ops[0].stab_mask() == 0);       // No Z
}

TEST_CASE("Frontend: T_DAG gate", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        T_DAG 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[0].is_dagger() == true);    // T†, not T
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
}

TEST_CASE("Frontend: multiple T gates on different qubits", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        H 1
        T 0 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // T on qubit 0: rewound Z is X0
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
    REQUIRE(hir.ops[0].stab_mask() == 0);

    // T on qubit 1: rewound Z is X1
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].destab_mask() == X(1));  // X on qubit 1
    REQUIRE(hir.ops[1].stab_mask() == 0);
}

TEST_CASE("Frontend: CX entangles qubits - T sees multi-qubit Pauli", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        CX 0 1
        T 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);

    // After H 0; CX 0 1:
    // Z1 rewound = (CX H)† Z1 (CX H) = H† CX† Z1 CX H = H† (Z0 Z1) H = X0 Z1
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
    REQUIRE(hir.ops[0].stab_mask() == Z(1));    // Z on qubit 1
}

TEST_CASE("Frontend: Z-basis measurement", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);

    // After H, Z0 is conjugated to X0
    REQUIRE(hir.ops[0].destab_mask() == X(0));  // X on qubit 0
    REQUIRE(hir.ops[0].stab_mask() == 0);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});
}

TEST_CASE("Frontend: X-basis measurement", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        MX 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);

    // MX measures X observable
    // After H, X0 is conjugated to Z0
    REQUIRE(hir.ops[0].destab_mask() == 0);
    REQUIRE(hir.ops[0].stab_mask() == Z(0));  // Z on qubit 0
}

TEST_CASE("Frontend: reset decomposition (R = M + CX rec[-1])", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        R 0
    )");
    auto hir = trace(circuit);

    // R is decomposed by parser into M + CX rec[-1] q
    // Should produce: MEASURE + CONDITIONAL_PAULI
    REQUIRE(hir.num_ops() == 2);

    // First op: measurement
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});

    // Second op: conditional X (from CX rec[-1] 0)
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});
    // The conditional X is rewound through the tableau
}

TEST_CASE("Frontend: MR decomposition (MR = M + CX rec[-1])", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        MR 0
    )");
    auto hir = trace(circuit);

    // MR is decomposed by parser into M + CX rec[-1] q (same as R)
    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});
}

TEST_CASE("Frontend: MRX decomposition (MRX = MX + CZ rec[-1])", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        MRX 0
    )");
    auto hir = trace(circuit);

    // MRX is decomposed by parser into MX + CZ rec[-1] q
    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});
}

TEST_CASE("Frontend: measurement record indexing", "[frontend]") {
    auto circuit = parse(R"(
        M 0
        M 1
        M 2
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 3);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});
    REQUIRE(hir.ops[1].meas_record_idx() == MeasRecordIdx{1});
    REQUIRE(hir.ops[2].meas_record_idx() == MeasRecordIdx{2});
}

TEST_CASE("Frontend: MPP single Pauli product", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        H 1
        MPP X0*X1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);

    // MPP X0*X1 measures the X0⊗X1 observable
    // After H on both qubits, X is conjugated to Z
    // So rewound X0*X1 = Z0*Z1
    REQUIRE(hir.ops[0].destab_mask() == 0);            // No X
    REQUIRE(hir.ops[0].stab_mask() == (Z(0) | Z(1)));  // Z on qubits 0 and 1
}

TEST_CASE("Frontend: MPP Z product", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        CX 0 1
        MPP Z0*Z1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);

    // After H 0; CX 0 1:
    // Z0 rewound = X0
    // Z1 rewound = X0 Z1
    // Product Z0*Z1 = X0 * (X0 Z1) = Z1 (X0 cancels)
    REQUIRE(hir.ops[0].destab_mask() == 0);
    REQUIRE(hir.ops[0].stab_mask() == Z(1));  // Just Z1
}

TEST_CASE("Frontend: exceeds 64 qubit limit", "[frontend]") {
    Circuit circuit;
    circuit.num_qubits = 65;  // Exceeds MVP limit

    REQUIRE_THROWS_AS(trace(circuit), std::runtime_error);
}

TEST_CASE("Frontend: T count tracking", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        T 0
        T 1
        T_DAG 2
        M 0
    )");
    auto hir = trace(circuit);

    // 3 T/T† gates + 1 measurement
    REQUIRE(hir.num_ops() == 4);
    REQUIRE(hir.num_t_gates() == 3);
}

// =============================================================================
// AG Pivot Tests (Task 3.3)
// =============================================================================

TEST_CASE("Frontend: deterministic measurement - no AG matrix", "[frontend][ag]") {
    // Measuring |0> in Z-basis is deterministic (outcome always 0)
    auto circuit = parse(R"(
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() == AgMatrixIdx::None);  // Deterministic
    REQUIRE(hir.ag_matrices.empty());                          // No AG matrices stored
}

TEST_CASE("Frontend: anti-commuting Z measurement - generates AG matrix", "[frontend][ag]") {
    // After H, measuring in Z-basis is random (50/50)
    // This should generate an AG pivot matrix
    auto circuit = parse(R"(
        H 0
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);  // Anti-commuting
    REQUIRE(hir.ag_matrices.size() == 1);                      // One AG matrix stored
}

TEST_CASE("Frontend: deterministic X measurement after H", "[frontend][ag]") {
    // H|0> = |+>, so MX is deterministic (outcome always 0)
    auto circuit = parse(R"(
        H 0
        MX 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() == AgMatrixIdx::None);  // Deterministic
    REQUIRE(hir.ag_matrices.empty());
}

TEST_CASE("Frontend: anti-commuting X measurement", "[frontend][ag]") {
    // |0> measured in X-basis is random (50/50)
    auto circuit = parse(R"(
        MX 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);  // Anti-commuting
    REQUIRE(hir.ag_matrices.size() == 1);
}

TEST_CASE("Frontend: mid-circuit measurement updates tableau", "[frontend][ag]") {
    // Key test: after measuring qubit 0, the tableau should be collapsed.
    // A subsequent measurement on the same qubit should be deterministic.
    auto circuit = parse(R"(
        H 0
        M 0
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // First measurement: anti-commuting (H|0> measured in Z)
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});

    // Second measurement: deterministic (qubit already collapsed to Z eigenstate)
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].ag_matrix_idx() == AgMatrixIdx::None);
    REQUIRE(hir.ops[1].meas_record_idx() == MeasRecordIdx{1});

    // Only one AG matrix (from first measurement)
    REQUIRE(hir.ag_matrices.size() == 1);
}

TEST_CASE("Frontend: multiple independent measurements", "[frontend][ag]") {
    // Two independent qubits, each with anti-commuting measurement
    auto circuit = parse(R"(
        H 0
        H 1
        M 0 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // Both measurements are anti-commuting
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);
    REQUIRE(hir.ops[1].ag_matrix_idx() != AgMatrixIdx::None);

    // Two distinct AG matrices
    REQUIRE(hir.ag_matrices.size() == 2);
    auto idx0 = static_cast<uint32_t>(hir.ops[0].ag_matrix_idx());
    auto idx1 = static_cast<uint32_t>(hir.ops[1].ag_matrix_idx());
    REQUIRE(idx0 != idx1);
}

TEST_CASE("Frontend: entangled measurement with Bell state", "[frontend][ag]") {
    // Create Bell state |00> + |11>, then measure qubit 0
    // This should be anti-commuting (50/50)
    auto circuit = parse(R"(
        H 0
        CX 0 1
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);
    REQUIRE(hir.ag_matrices.size() == 1);
}

TEST_CASE("Frontend: Bell state - second qubit deterministic after first measured",
          "[frontend][ag]") {
    // Bell state: after measuring qubit 0, qubit 1 is determined
    auto circuit = parse(R"(
        H 0
        CX 0 1
        M 0
        M 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // First measurement: anti-commuting
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);

    // Second measurement: deterministic (perfectly correlated with first)
    REQUIRE(hir.ops[1].ag_matrix_idx() == AgMatrixIdx::None);

    // Only one AG matrix
    REQUIRE(hir.ag_matrices.size() == 1);
}

TEST_CASE("Frontend: reset after measurement - T gate sees updated tableau", "[frontend][ag]") {
    // After reset, the qubit is back to |0>, so T on it should see Z_q
    auto circuit = parse(R"(
        H 0
        R 0
        T 0
    )");
    auto hir = trace(circuit);

    // R is decomposed to M + CX rec[-1]
    // So we have: MEASURE, CONDITIONAL_PAULI, T_GATE
    REQUIRE(hir.num_ops() == 3);

    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[2].op_type() == OpType::T_GATE);

    // After reset, the T gate sees Z_q (qubit is in |0> state)
    // The destab_mask should be 0 (no X), stab_mask should be Z(0)
    REQUIRE(hir.ops[2].destab_mask() == 0);
    REQUIRE(hir.ops[2].stab_mask() == Z(0));
}

TEST_CASE("Frontend: MPP deterministic measurement", "[frontend][ag]") {
    // After CX, Z0*Z1 stabilizes the state
    // Measuring Z0*Z1 on a Bell state should be deterministic
    auto circuit = parse(R"(
        H 0
        CX 0 1
        MPP Z0*Z1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);

    // Z0*Z1 is a stabilizer of the Bell state, so measurement is deterministic
    REQUIRE(hir.ops[0].ag_matrix_idx() == AgMatrixIdx::None);
    REQUIRE(hir.ag_matrices.empty());
}

TEST_CASE("Frontend: MPP anti-commuting measurement", "[frontend][ag]") {
    // X0*X1 anti-commutes with the |00> state (not a stabilizer)
    auto circuit = parse(R"(
        MPP X0*X1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);
    REQUIRE(hir.ag_matrices.size() == 1);
}

TEST_CASE("Frontend: ag_ref_outcome records collapse choice", "[frontend][ag]") {
    // The ag_ref_outcome field records what outcome the compiler chose
    // This is needed by the VM to know how to apply the AG pivot
    auto circuit = parse(R"(
        H 0
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);

    // ag_ref_outcome should be 0 or 1 (the compiler's chosen outcome)
    uint8_t ref = hir.ops[0].ag_ref_outcome();
    REQUIRE((ref == 0 || ref == 1));
}

// =============================================================================
// Classical Control Tests (Task 3.4)
// =============================================================================

TEST_CASE("Frontend: classical feedback sees collapsed tableau", "[frontend][classical]") {
    // After H; M, the qubit is collapsed to |0> or |1>
    // The CX rec[-1] 0 should see Z_0 (not the pre-measurement X_0)
    auto circuit = parse(R"(
        H 0
        M 0
        CX rec[-1] 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // First: measurement (anti-commuting since H|0> measured in Z)
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() != AgMatrixIdx::None);

    // Second: conditional X on qubit 0
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});

    // After collapse, the qubit is in a Z eigenstate
    // So the rewound X_0 should map to X_0 (identity tableau on that qubit)
    // This means destab_mask should have bit 0 set (X component)
    REQUIRE(hir.ops[1].destab_mask() == X(0));
    REQUIRE(hir.ops[1].stab_mask() == 0);
}

TEST_CASE("Frontend: classical feedback on entangled qubits", "[frontend][classical]") {
    // Create Bell state, measure qubit 0, apply conditional Z on qubit 1
    // After measuring qubit 0, qubit 1 is also collapsed (correlated)
    auto circuit = parse(R"(
        H 0
        CX 0 1
        M 0
        CZ rec[-1] 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    // First: measurement (anti-commuting on Bell state)
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);

    // Second: conditional Z on qubit 1
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});

    // After Bell state collapse to |00> or |11>, the qubits are perfectly correlated.
    // In the Heisenberg picture, Z_1 is still represented as Z_0*Z_1 because
    // measuring Z_0 determines Z_1 (they have the same value).
    // So rewound Z_1 = Z_0*Z_1
    REQUIRE(hir.ops[1].destab_mask() == 0);
    REQUIRE(hir.ops[1].stab_mask() == (Z(0) | Z(1)));  // Z_0 * Z_1
}

TEST_CASE("Frontend: multiple resets in sequence", "[frontend][classical]") {
    // Multiple resets should all work correctly with updated tableau
    auto circuit = parse(R"(
        H 0
        H 1
        R 0
        R 1
        T 0 1
    )");
    auto hir = trace(circuit);

    // R 0 -> M 0 + CX rec[-1] 0
    // R 1 -> M 1 + CX rec[-1] 1
    // T 0 + T 1
    REQUIRE(hir.num_ops() == 6);

    // M 0, CX rec[-1] 0, M 1, CX rec[-1] 1, T 0, T 1
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);            // M 0
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);  // CX rec[-1] 0
    REQUIRE(hir.ops[2].op_type() == OpType::MEASURE);            // M 1
    REQUIRE(hir.ops[3].op_type() == OpType::CONDITIONAL_PAULI);  // CX rec[-1] 1
    REQUIRE(hir.ops[4].op_type() == OpType::T_GATE);             // T 0
    REQUIRE(hir.ops[5].op_type() == OpType::T_GATE);             // T 1

    // After both resets, both qubits should be in |0> state
    // So T gates should see Z_q (not modified by Clifford frame)
    REQUIRE(hir.ops[4].destab_mask() == 0);
    REQUIRE(hir.ops[4].stab_mask() == Z(0));
    REQUIRE(hir.ops[5].destab_mask() == 0);
    REQUIRE(hir.ops[5].stab_mask() == Z(1));
}

// =============================================================================
// Regression Tests (Review Feedback Fixes)
// =============================================================================

TEST_CASE("Frontend: deterministic measurement with outcome 1 sets ag_ref",
          "[frontend][regression]") {
    // Critical bug fix: ag_ref must be set even for deterministic measurements
    // X gate flips |0> to |1>, so M should deterministically give 1
    auto circuit = parse(R"(
        X 0
        M 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].ag_matrix_idx() == AgMatrixIdx::None);  // Deterministic
    REQUIRE(hir.ops[0].ag_ref_outcome() == 1);                 // Must be 1, not 0!
}

TEST_CASE("Frontend: deterministic MX measurement with outcome 1", "[frontend][regression]") {
    // H;X|0> = H|1> = |->, so MX gives deterministic 1
    auto circuit = parse(R"(
        X 0
        H 0
        MX 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].ag_matrix_idx() == AgMatrixIdx::None);  // Deterministic
    REQUIRE(hir.ops[0].ag_ref_outcome() == 1);                 // Must be 1!
}

TEST_CASE("Frontend: broadcast classical feedback CX rec[-2] 0 rec[-1] 1",
          "[frontend][regression]") {
    // Test that classical feedback loop handles multiple pairs
    // Manually construct circuit with broadcast CX feedback
    Circuit circuit;
    circuit.num_qubits = 2;
    circuit.num_measurements = 2;

    // H 0; H 1
    circuit.nodes.push_back({GateType::H, {Target::qubit(0)}, 0.0});
    circuit.nodes.push_back({GateType::H, {Target::qubit(1)}, 0.0});

    // M 0; M 1
    circuit.nodes.push_back({GateType::M, {Target::qubit(0)}, 0.0});
    circuit.nodes.push_back({GateType::M, {Target::qubit(1)}, 0.0});

    // CX rec[-2] 0 rec[-1] 1 (broadcast form)
    AstNode cx_node;
    cx_node.gate = GateType::CX;
    cx_node.targets.push_back(Target::rec(0));  // rec[-2] -> absolute 0
    cx_node.targets.push_back(Target::qubit(0));
    cx_node.targets.push_back(Target::rec(1));  // rec[-1] -> absolute 1
    cx_node.targets.push_back(Target::qubit(1));
    circuit.nodes.push_back(cx_node);

    auto hir = trace(circuit);

    // M 0, M 1, CX rec[-2] 0, CX rec[-1] 1
    REQUIRE(hir.num_ops() == 4);

    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});

    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].meas_record_idx() == MeasRecordIdx{1});

    // Both conditional paulis should be emitted
    REQUIRE(hir.ops[2].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[2].controlling_meas() == ControllingMeasIdx{0});

    REQUIRE(hir.ops[3].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[3].controlling_meas() == ControllingMeasIdx{1});
}

TEST_CASE("Frontend: CY classical feedback throws", "[frontend][regression]") {
    // CY rec[-k] q is not supported in MVP
    // We need to manually construct this since the parser won't generate it
    Circuit circuit;
    circuit.num_qubits = 1;
    circuit.num_measurements = 1;

    // First add a measurement
    AstNode m_node;
    m_node.gate = GateType::M;
    m_node.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(m_node);

    // Then add CY rec[-1] 0
    AstNode cy_node;
    cy_node.gate = GateType::CY;
    cy_node.targets.push_back(Target::rec(0));  // rec[-1] -> absolute index 0
    cy_node.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(cy_node);

    REQUIRE_THROWS_AS(trace(circuit), std::runtime_error);
}
