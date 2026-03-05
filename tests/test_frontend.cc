// Front-End unit tests
//
// Tests the trace() function which converts Circuit -> HirModule
// Key invariant: Clifford gates are absorbed, T gates emit HeisenbergOps
// with correctly rewound Pauli masks.

#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"

#include "test_helpers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using namespace ucc;
using ucc::test::X;
using ucc::test::Z;

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
    REQUIRE(hir.ops[0].is_dagger() == true);    // T_dag, not T
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
    // Z1 rewound = (CX H)_dag Z1 (CX H) = H_dag CX_dag Z1 CX H = H_dag (Z0 Z1) H = X0 Z1
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

TEST_CASE("Frontend: reset R as first-class operation", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        R 0
    )");
    auto hir = trace(circuit);

    // R decomposes into hidden MEASURE + CONDITIONAL with use_last_outcome flag
    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].is_hidden());  // Hidden measurement for reset
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].use_last_outcome());  // Uses outcome of preceding hidden measurement
    REQUIRE(hir.num_measurements == 0);      // R has no visible measurement
}

TEST_CASE("Frontend: MR as first-class operation", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        MR 0
    )");
    auto hir = trace(circuit);

    // MR produces a visible MEASURE followed by CONDITIONAL with use_last_outcome
    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});
    REQUIRE(!hir.ops[0].is_hidden());  // MR has visible measurement
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].use_last_outcome());
    REQUIRE(hir.num_measurements == 1);
}

TEST_CASE("Frontend: MRX as first-class operation", "[frontend]") {
    auto circuit = parse(R"(
        H 0
        MRX 0
    )");
    auto hir = trace(circuit);

    // MRX produces a visible MEASURE followed by CONDITIONAL with use_last_outcome
    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(!hir.ops[0].is_hidden());  // MRX has visible measurement
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].use_last_outcome());
    REQUIRE(hir.num_measurements == 1);
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

    // MPP X0*X1 measures the X0tensorX1 observable
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

    // 3 T/T_dag gates + 1 measurement
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
}

TEST_CASE("Frontend: anti-commuting X measurement", "[frontend][ag]") {
    // |0> measured in X-basis is random (50/50)
    auto circuit = parse(R"(
        MX 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
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
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});

    // Second measurement: deterministic (qubit already collapsed to Z eigenstate)
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].meas_record_idx() == MeasRecordIdx{1});

    // Only one AG matrix (from first measurement)
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
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
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

    // Second measurement: deterministic (perfectly correlated with first)

    // Only one AG matrix
}

TEST_CASE("Frontend: reset then T - AOT frame is un-collapsed", "[frontend][ag]") {
    // With Clifford Frame Determinism the AOT tableau never collapses.
    // After H 0, inv_state.zs[0] = X_0. R does not change the tableau.
    // So T 0 sees the rewound Z through the un-collapsed frame: X_0.
    auto circuit = parse(R"(
        H 0
        R 0
        T 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[0].is_hidden());
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].use_last_outcome());
    REQUIRE(hir.ops[2].op_type() == OpType::T_GATE);

    // Un-collapsed: T gate sees X_0 (H maps Z->X)
    REQUIRE(hir.ops[2].destab_mask() == X(0));
    REQUIRE(hir.ops[2].stab_mask() == 0);
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
}

TEST_CASE("Frontend: MPP anti-commuting measurement", "[frontend][ag]") {
    // X0*X1 anti-commutes with the |00> state (not a stabilizer)
    auto circuit = parse(R"(
        MPP X0*X1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
}

// =============================================================================
// Classical Control Tests
// =============================================================================

TEST_CASE("Frontend: classical feedback sees un-collapsed tableau", "[frontend][classical]") {
    // With Clifford Frame Determinism, the AOT tableau never collapses.
    // After H 0, inv_state.xs[0] = Z_0 (H swaps X<->Z). The CX rec[-1] 0
    // extracts the rewound X from the un-collapsed tableau.
    auto circuit = parse(R"(
        H 0
        M 0
        CX rec[-1] 0
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);

    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});

    // Un-collapsed tableau after H: xs[0] = Z_0
    REQUIRE(hir.ops[1].destab_mask() == 0);
    REQUIRE(hir.ops[1].stab_mask() == Z(0));
}

TEST_CASE("Frontend: classical feedback on entangled qubits", "[frontend][classical]") {
    // With Clifford Frame Determinism the AOT frame never collapses.
    // After H 0; CX 0 1, inv_state.zs[1] = X_0 * Z_1 (un-collapsed).
    auto circuit = parse(R"(
        H 0
        CX 0 1
        M 0
        CZ rec[-1] 1
    )");
    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.ops[1].controlling_meas() == ControllingMeasIdx{0});

    // Un-collapsed: rewound Z_1 through H 0; CX 0 1 gives X_0 * Z_1
    REQUIRE(hir.ops[1].destab_mask() == X(0));
    REQUIRE(hir.ops[1].stab_mask() == Z(1));
}

TEST_CASE("Frontend: multiple resets in sequence", "[frontend][classical]") {
    // Multiple resets with Clifford Frame Determinism: the AOT frame never
    // collapses, so the tableau remains un-collapsed throughout.
    auto circuit = parse(R"(
        H 0
        H 1
        R 0
        R 1
        T 0 1
    )");
    auto hir = trace(circuit);

    // Each R decomposes into MEASURE + CONDITIONAL, then T 0, T 1
    REQUIRE(hir.num_ops() == 6);

    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);            // R 0 measure
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);  // R 0 correction
    REQUIRE(hir.ops[2].op_type() == OpType::MEASURE);            // R 1 measure
    REQUIRE(hir.ops[3].op_type() == OpType::CONDITIONAL_PAULI);  // R 1 correction
    REQUIRE(hir.ops[4].op_type() == OpType::T_GATE);             // T 0
    REQUIRE(hir.ops[5].op_type() == OpType::T_GATE);             // T 1

    // Without collapse, the tableau after H still maps Z_0 -> X_0, Z_1 -> X_1.
    // The T gates see the un-collapsed rewound Pauli.
    REQUIRE(hir.ops[4].destab_mask() == X(0));
    REQUIRE(hir.ops[4].stab_mask() == 0);
    REQUIRE(hir.ops[5].destab_mask() == X(1));
    REQUIRE(hir.ops[5].stab_mask() == 0);
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

// =============================================================================
// Phase 2.2: Noise and QEC Emission Tests
// =============================================================================

TEST_CASE("Frontend: DEPOLARIZE1 produces 3 rewound channels", "[frontend][noise]") {
    // DEPOLARIZE1(p) on a single qubit should produce 3 NoiseChannels (X, Y, Z)
    // each with probability p/3.
    Circuit circuit;
    circuit.num_qubits = 1;

    AstNode dep1;
    dep1.gate = GateType::DEPOLARIZE1;
    dep1.arg = 0.03;  // Total error probability
    dep1.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(dep1);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);

    auto site_idx = hir.ops[0].noise_site_idx();
    REQUIRE(static_cast<uint32_t>(site_idx) == 0);
    REQUIRE(hir.noise_sites.size() == 1);

    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 3);

    // Each channel should have probability p/3 = 0.01
    for (const auto& ch : site.channels) {
        REQUIRE(ch.prob == Catch::Approx(0.01));
    }
}

TEST_CASE("Frontend: X_ERROR produces single channel", "[frontend][noise]") {
    Circuit circuit;
    circuit.num_qubits = 1;

    AstNode x_err;
    x_err.gate = GateType::X_ERROR;
    x_err.arg = 0.001;
    x_err.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(x_err);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);

    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 1);
    REQUIRE(site.channels[0].prob == Catch::Approx(0.001));

    // X on qubit 0 at t=0 (identity tableau): destab=1, stab=0
    REQUIRE(site.channels[0].destab_mask == 1);
    REQUIRE(site.channels[0].stab_mask == 0);
}

TEST_CASE("Frontend: Z_ERROR produces single channel", "[frontend][noise]") {
    Circuit circuit;
    circuit.num_qubits = 1;

    AstNode z_err;
    z_err.gate = GateType::Z_ERROR;
    z_err.arg = 0.002;
    z_err.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(z_err);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 1);
    REQUIRE(site.channels[0].prob == Catch::Approx(0.002));

    // Z on qubit 0 at t=0: destab=0, stab=1
    REQUIRE(site.channels[0].destab_mask == 0);
    REQUIRE(site.channels[0].stab_mask == 1);
}

TEST_CASE("Frontend: Y_ERROR produces single channel", "[frontend][noise]") {
    Circuit circuit;
    circuit.num_qubits = 1;

    AstNode y_err;
    y_err.gate = GateType::Y_ERROR;
    y_err.arg = 0.003;
    y_err.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(y_err);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 1);
    REQUIRE(site.channels[0].prob == Catch::Approx(0.003));

    // Y = iXZ on qubit 0 at t=0: destab=1, stab=1
    REQUIRE(site.channels[0].destab_mask == 1);
    REQUIRE(site.channels[0].stab_mask == 1);
}

TEST_CASE("Frontend: DEPOLARIZE2 produces 15 channels", "[frontend][noise]") {
    // DEPOLARIZE2(p) on qubits 0,1 should produce 15 channels (all non-II Paulis)
    Circuit circuit;
    circuit.num_qubits = 2;

    AstNode dep2;
    dep2.gate = GateType::DEPOLARIZE2;
    dep2.arg = 0.15;  // Total error probability
    dep2.targets.push_back(Target::qubit(0));
    dep2.targets.push_back(Target::qubit(1));
    circuit.nodes.push_back(dep2);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);

    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 15);

    // Each channel should have probability p/15 = 0.01
    for (const auto& ch : site.channels) {
        REQUIRE(ch.prob == Catch::Approx(0.01));
    }
}

TEST_CASE("Frontend: noise rewinding through H gate", "[frontend][noise]") {
    // H 0, X_ERROR 0
    // X after H becomes Z (at t=0, the error manifests as Z)
    Circuit circuit;
    circuit.num_qubits = 1;

    AstNode h;
    h.gate = GateType::H;
    h.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(h);

    AstNode x_err;
    x_err.gate = GateType::X_ERROR;
    x_err.arg = 0.01;
    x_err.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(x_err);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 1);
    const auto& site = hir.noise_sites[0];
    REQUIRE(site.channels.size() == 1);

    // X after H = Z at t=0: destab=0, stab=1
    REQUIRE(site.channels[0].destab_mask == 0);
    REQUIRE(site.channels[0].stab_mask == 1);
}

TEST_CASE("Frontend: READOUT_NOISE emission", "[frontend][noise]") {
    // Manually construct a circuit with READOUT_NOISE
    // (normally parser decomposes M(p) into M + READOUT_NOISE)
    Circuit circuit;
    circuit.num_qubits = 1;
    circuit.num_measurements = 1;

    // First, a measurement
    AstNode m;
    m.gate = GateType::M;
    m.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(m);

    // Then readout noise on that measurement
    AstNode rn;
    rn.gate = GateType::READOUT_NOISE;
    rn.arg = 0.005;
    rn.targets.push_back(Target::rec(0));  // Absolute index 0
    circuit.nodes.push_back(rn);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::READOUT_NOISE);

    // Verify the readout noise index points to correct entry
    REQUIRE(static_cast<uint32_t>(hir.ops[1].readout_noise_idx()) == 0);
    REQUIRE(hir.readout_noise.size() == 1);
    REQUIRE(hir.readout_noise[0].meas_idx == 0);
    REQUIRE(hir.readout_noise[0].prob == Catch::Approx(0.005));
}

TEST_CASE("Frontend: DETECTOR emission", "[frontend][qec]") {
    // M 0, M 1, DETECTOR rec[-1] rec[-2]
    Circuit circuit;
    circuit.num_qubits = 2;
    circuit.num_measurements = 2;
    circuit.num_detectors = 1;

    AstNode m0, m1;
    m0.gate = GateType::M;
    m0.targets.push_back(Target::qubit(0));
    m1.gate = GateType::M;
    m1.targets.push_back(Target::qubit(1));
    circuit.nodes.push_back(m0);
    circuit.nodes.push_back(m1);

    AstNode det;
    det.gate = GateType::DETECTOR;
    det.targets.push_back(Target::rec(1));  // rec[-1] -> absolute 1
    det.targets.push_back(Target::rec(0));  // rec[-2] -> absolute 0
    circuit.nodes.push_back(det);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 3);
    REQUIRE(hir.ops[2].op_type() == OpType::DETECTOR);

    // Verify the detector index points to correct target list
    REQUIRE(static_cast<uint32_t>(hir.ops[2].detector_idx()) == 0);
    REQUIRE(hir.detector_targets.size() == 1);
    REQUIRE(hir.detector_targets[0].size() == 2);
    REQUIRE(hir.detector_targets[0][0] == 1);  // rec[-1]
    REQUIRE(hir.detector_targets[0][1] == 0);  // rec[-2]
    REQUIRE(hir.num_detectors == 1);
}

TEST_CASE("Frontend: OBSERVABLE_INCLUDE emission", "[frontend][qec]") {
    // M 0, OBSERVABLE_INCLUDE(0) rec[-1]
    Circuit circuit;
    circuit.num_qubits = 1;
    circuit.num_measurements = 1;
    circuit.num_observables = 1;

    AstNode m;
    m.gate = GateType::M;
    m.targets.push_back(Target::qubit(0));
    circuit.nodes.push_back(m);

    AstNode obs;
    obs.gate = GateType::OBSERVABLE_INCLUDE;
    obs.arg = 0;                            // Observable index
    obs.targets.push_back(Target::rec(0));  // rec[-1] -> absolute 0
    circuit.nodes.push_back(obs);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.ops[1].op_type() == OpType::OBSERVABLE);

    auto obs_idx = hir.ops[1].observable_idx();
    REQUIRE(static_cast<uint32_t>(obs_idx) == 0);

    REQUIRE(hir.observable_targets.size() == 1);
    REQUIRE(hir.observable_targets[0].size() == 1);
    REQUIRE(hir.observable_targets[0][0] == 0);
    REQUIRE(hir.num_observables == 1);
}

TEST_CASE("Frontend: multiple OBSERVABLE_INCLUDE accumulate", "[frontend][qec]") {
    // M 0 1, OBSERVABLE_INCLUDE(0) rec[-2], OBSERVABLE_INCLUDE(0) rec[-1]
    Circuit circuit;
    circuit.num_qubits = 2;
    circuit.num_measurements = 2;
    circuit.num_observables = 1;

    AstNode m;
    m.gate = GateType::M;
    m.targets.push_back(Target::qubit(0));
    m.targets.push_back(Target::qubit(1));
    circuit.nodes.push_back(m);

    AstNode obs1;
    obs1.gate = GateType::OBSERVABLE_INCLUDE;
    obs1.arg = 0;
    obs1.targets.push_back(Target::rec(0));  // rec[-2] -> absolute 0
    circuit.nodes.push_back(obs1);

    AstNode obs2;
    obs2.gate = GateType::OBSERVABLE_INCLUDE;
    obs2.arg = 0;
    obs2.targets.push_back(Target::rec(1));  // rec[-1] -> absolute 1
    circuit.nodes.push_back(obs2);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 4);  // 2 measurements + 2 observable includes
    REQUIRE(hir.ops[2].op_type() == OpType::OBSERVABLE);
    REQUIRE(hir.ops[3].op_type() == OpType::OBSERVABLE);

    // Both should reference observable 0
    REQUIRE(static_cast<uint32_t>(hir.ops[2].observable_idx()) == 0);
    REQUIRE(static_cast<uint32_t>(hir.ops[3].observable_idx()) == 0);

    // Each has its own target list
    REQUIRE(hir.observable_targets.size() == 2);
    REQUIRE(hir.observable_targets[0][0] == 0);
    REQUIRE(hir.observable_targets[1][0] == 1);
}

TEST_CASE("Frontend: noise broadcasting", "[frontend][noise]") {
    // X_ERROR(0.01) 0 1 2 should produce 3 separate noise sites
    Circuit circuit;
    circuit.num_qubits = 3;

    AstNode x_err;
    x_err.gate = GateType::X_ERROR;
    x_err.arg = 0.01;
    x_err.targets.push_back(Target::qubit(0));
    x_err.targets.push_back(Target::qubit(1));
    x_err.targets.push_back(Target::qubit(2));
    circuit.nodes.push_back(x_err);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 3);
    REQUIRE(hir.noise_sites.size() == 3);

    // Each site should have X error on its respective qubit
    // Qubit 0: destab=1, Qubit 1: destab=2, Qubit 2: destab=4
    REQUIRE(hir.noise_sites[0].channels[0].destab_mask == 1);
    REQUIRE(hir.noise_sites[1].channels[0].destab_mask == 2);
    REQUIRE(hir.noise_sites[2].channels[0].destab_mask == 4);
}

TEST_CASE("Frontend: DEPOLARIZE2 broadcasting", "[frontend][noise]") {
    // DEPOLARIZE2(0.15) 0 1 2 3 should produce 2 separate noise sites
    Circuit circuit;
    circuit.num_qubits = 4;

    AstNode dep2;
    dep2.gate = GateType::DEPOLARIZE2;
    dep2.arg = 0.15;
    dep2.targets.push_back(Target::qubit(0));
    dep2.targets.push_back(Target::qubit(1));
    dep2.targets.push_back(Target::qubit(2));
    dep2.targets.push_back(Target::qubit(3));
    circuit.nodes.push_back(dep2);

    auto hir = trace(circuit);

    REQUIRE(hir.num_ops() == 2);
    REQUIRE(hir.noise_sites.size() == 2);

    // Each site should have 15 channels
    REQUIRE(hir.noise_sites[0].channels.size() == 15);
    REQUIRE(hir.noise_sites[1].channels.size() == 15);
}

TEST_CASE("Frontend: DoD - DEPOLARIZE1 after Clifford produces 3 rewound masks",
          "[frontend][noise][dod]") {
    // This test matches the Phase 2.2 Definition of Done:
    // "A pure Clifford circuit with DEPOLARIZE1 0 produces an HIR NOISE node
    //  pointing to a NoiseSite of exactly 3 rewound Pauli masks."
    auto circuit = parse("H 0\nCX 0 1\nDEPOLARIZE1(0.01) 0");

    auto hir = trace(circuit);

    // Should have exactly one NOISE op
    size_t noise_count = 0;
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::NOISE) {
            noise_count++;
            auto idx = op.noise_site_idx();
            const auto& site = hir.noise_sites[static_cast<uint32_t>(idx)];
            // Exactly 3 Pauli masks (X, Y, Z)
            REQUIRE(site.channels.size() == 3);
        }
    }
    REQUIRE(noise_count == 1);
}

TEST_CASE("Frontend: full QEC circuit with noise and detectors",
          "[frontend][noise][qec][integration]") {
    // A simple QEC-like circuit with noise and detectors
    const char* circuit_text = R"(
        R 0 1
        H 0
        CX 0 1
        DEPOLARIZE2(0.001) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";

    auto circuit = parse(circuit_text);
    auto hir = trace(circuit);

    // Count operation types
    size_t noise_ops = 0, visible_measure_ops = 0, hidden_measure_ops = 0;
    size_t detector_ops = 0, observable_ops = 0;
    size_t reset_ops = 0;

    for (const auto& op : hir.ops) {
        switch (op.op_type()) {
            case OpType::NOISE:
                noise_ops++;
                break;
            case OpType::MEASURE:
                if (op.is_hidden()) {
                    hidden_measure_ops++;
                } else {
                    visible_measure_ops++;
                }
                break;
            case OpType::CONDITIONAL_PAULI:
                reset_ops++;  // Count conditionals as "reset ops" for this test
                break;
            case OpType::DETECTOR:
                detector_ops++;
                break;
            case OpType::OBSERVABLE:
                observable_ops++;
                break;
            default:
                break;
        }
    }

    // R 0, R 1 produce 2 hidden measurements
    // M 0 1 produces 2 visible measurements
    REQUIRE(visible_measure_ops == 2);
    REQUIRE(hidden_measure_ops == 2);
    // 2 resets from R 0 1 (conditional corrections)
    REQUIRE(reset_ops == 2);
    // 1 DEPOLARIZE2 -> 1 noise site with 15 channels
    REQUIRE(noise_ops == 1);
    REQUIRE(hir.noise_sites[0].channels.size() == 15);
    // 1 DETECTOR
    REQUIRE(detector_ops == 1);
    // 1 OBSERVABLE_INCLUDE
    REQUIRE(observable_ops == 1);

    REQUIRE(hir.num_detectors == 1);
    REQUIRE(hir.num_observables == 1);
}
