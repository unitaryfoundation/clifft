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
