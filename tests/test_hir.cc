// HIR (Heisenberg IR) unit tests

#include "clifft/frontend/hir.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <tuple>
#include <utility>

using namespace clifft;
using clifft::test::X;
using clifft::test::Z;

using clifft::test::pauli_masks;

// =============================================================================
// HeisenbergOp Tests
// =============================================================================

// Note: sizeof(HeisenbergOp) == 32 is enforced by static_assert in hir.h.
// We don't duplicate that check here - the code won't compile if it fails.

TEST_CASE("HeisenbergOp::make_tgate", "[hir]") {
    SECTION("T gate with Z on qubit 0") {
        auto [destab, stab] = pauli_masks("Z");
        auto op = HeisenbergOp::make_tgate(destab, stab, /*sign=*/false);

        REQUIRE(op.op_type() == OpType::T_GATE);
        REQUIRE(op.destab_mask() == 0);    // No X
        REQUIRE(op.stab_mask() == Z(0));   // Z on qubit 0
        REQUIRE(op.sign() == false);       // Positive phase
        REQUIRE(op.is_dagger() == false);  // T, not T_dag
    }

    SECTION("T_dag gate with X on qubit 1, negative sign") {
        auto op = HeisenbergOp::make_tgate(X(1), 0, /*sign=*/true, /*dagger=*/true);

        REQUIRE(op.op_type() == OpType::T_GATE);
        REQUIRE(op.destab_mask() == X(1));  // X on qubit 1
        REQUIRE(op.stab_mask() == 0);       // No Z
        REQUIRE(op.sign() == true);         // Negative phase
        REQUIRE(op.is_dagger() == true);    // T_dag
    }

    SECTION("T gate with Y on qubit 2 (both X and Z bits set)") {
        // "Y__" means Y on qubit 2, I on qubits 1 and 0 (rightmost is qubit 0)
        auto [destab, stab] = pauli_masks("Y__");
        auto op = HeisenbergOp::make_tgate(destab, stab, false);

        REQUIRE(op.destab_mask() == X(2));  // X component of Y
        REQUIRE(op.stab_mask() == Z(2));    // Z component of Y
    }
}

TEST_CASE("HeisenbergOp::set_pauli replaces masks and sign", "[hir]") {
    auto op = HeisenbergOp::make_tgate(0, Z(0), /*sign=*/false);
    REQUIRE(op.stab_mask() == Z(0));
    REQUIRE(op.sign() == false);

    op.set_pauli(X(1), Z(2), true);
    REQUIRE(op.destab_mask() == X(1));
    REQUIRE(op.stab_mask() == Z(2));
    REQUIRE(op.sign() == true);
    // OpType and flags are preserved
    REQUIRE(op.op_type() == OpType::T_GATE);
    REQUIRE(op.is_dagger() == false);
}

TEST_CASE("HeisenbergOp bitword operations", "[hir]") {
    // Verify bitword<64> provides expected operations for commutation checks
    auto op1 = HeisenbergOp::make_tgate(X(1) | X(3), Z(0) | Z(2), false);  // X1 X3 Z0 Z2
    auto op2 = HeisenbergOp::make_tgate(X(2) | X(3), Z(0) | Z(1), false);  // X2 X3 Z0 Z1

    // Test popcount (useful for commutation: count overlapping X-Z pairs)
    // op1.destab & op2.stab = X1 X3 & Z0 Z1 = bit 1 only
    REQUIRE((op1.destab_mask() & op2.stab_mask()).popcount() == 1);
    // op1.stab & op2.destab = Z0 Z2 & X2 X3 = bit 2 only
    REQUIRE((op1.stab_mask() & op2.destab_mask()).popcount() == 1);

    // Test XOR for combining masks
    auto xor_destab = op1.destab_mask() ^ op2.destab_mask();
    REQUIRE(xor_destab == (X(1) | X(2)));  // X3 cancels, X1 and X2 remain
}

TEST_CASE("HeisenbergOp::make_measure", "[hir]") {
    SECTION("Measurement with record index") {
        auto op = HeisenbergOp::make_measure(X(0), 0, /*sign=*/false, MeasRecordIdx{5});

        REQUIRE(op.op_type() == OpType::MEASURE);
        REQUIRE(op.destab_mask() == X(0));
        REQUIRE(op.stab_mask() == 0);
        REQUIRE(op.meas_record_idx() == MeasRecordIdx{5});
    }

    SECTION("Multi-qubit measurement") {
        auto op = HeisenbergOp::make_measure(X(0) | X(1), 0, false, MeasRecordIdx{10});
        REQUIRE(op.meas_record_idx() == MeasRecordIdx{10});
    }
}

TEST_CASE("HeisenbergOp::make_conditional", "[hir]") {
    auto op = HeisenbergOp::make_conditional(X(0), 0, /*sign=*/false, ControllingMeasIdx{7});

    REQUIRE(op.op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(op.destab_mask() == X(0));
    REQUIRE(op.controlling_meas() == ControllingMeasIdx{7});
}

TEST_CASE("pauli_masks helper", "[hir][helper]") {
    // Verify our test helper works correctly

    SECTION("Single paulis") {
        auto [d, s] = pauli_masks("X");
        REQUIRE(d == X(0));
        REQUIRE(s == 0);

        std::tie(d, s) = pauli_masks("Z");
        REQUIRE(d == 0);
        REQUIRE(s == Z(0));

        std::tie(d, s) = pauli_masks("Y");
        REQUIRE(d == X(0));
        REQUIRE(s == Z(0));

        std::tie(d, s) = pauli_masks("I");
        REQUIRE(d == 0);
        REQUIRE(s == 0);
    }

    SECTION("Multi-qubit strings") {
        // "XYZ" = X on q2, Y on q1, Z on q0
        auto [d, s] = pauli_masks("XYZ");
        REQUIRE(d == (X(2) | X(1)));  // X on q2, X component of Y on q1
        REQUIRE(s == (Z(1) | Z(0)));  // Z component of Y on q1, Z on q0

        // "IXZI" = I on q3, X on q2, Z on q1, I on q0
        std::tie(d, s) = pauli_masks("IXZI");
        REQUIRE(d == X(2));
        REQUIRE(s == Z(1));
    }
}

// =============================================================================
// stim::Tableau API used by the Front-End
// =============================================================================

TEST_CASE("stim::Tableau identity and Hadamard", "[hir]") {
    stim::Tableau<kStimWidth> tab(4);  // 4 qubits

    // Default is identity: X_k -> X_k, Z_k -> Z_k
    REQUIRE(tab.num_qubits == 4);

    // Check that xs[0] is X_0 (destabilizer for qubit 0)
    auto x0 = tab.xs[0];
    REQUIRE(x0.xs[0] == true);   // Has X component on qubit 0
    REQUIRE(x0.zs[0] == false);  // No Z component on qubit 0
    REQUIRE(x0.sign == false);   // Positive sign

    // Check that zs[0] is Z_0 (stabilizer for qubit 0)
    auto z0 = tab.zs[0];
    REQUIRE(z0.xs[0] == false);  // No X component on qubit 0
    REQUIRE(z0.zs[0] == true);   // Has Z component on qubit 0
    REQUIRE(z0.sign == false);   // Positive sign

    // Apply a Hadamard to qubit 0 - should swap X and Z
    tab.prepend_H_XZ(0);
    auto x0_after = tab.xs[0];
    auto z0_after = tab.zs[0];

    // After H: X_0 -> Z_0, Z_0 -> X_0
    REQUIRE(x0_after.xs[0] == false);  // No X
    REQUIRE(x0_after.zs[0] == true);   // Has Z
    REQUIRE(z0_after.xs[0] == true);   // Has X
    REQUIRE(z0_after.zs[0] == false);  // No Z
}

// =============================================================================
// HirModule Tests
// =============================================================================

TEST_CASE("HirModule construction and accessors", "[hir]") {
    HirModule hir;
    hir.num_qubits = 4;

    REQUIRE(hir.num_ops() == 0);
    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.global_weight == std::complex<double>(1.0, 0.0));

    // Add some operations: 2 T gates, 1 T_dag gate, 1 measurement
    hir.ops.push_back(HeisenbergOp::make_tgate(X(0), 0, false));        // T
    hir.ops.push_back(HeisenbergOp::make_tgate(X(1), 0, false, true));  // T_dag
    hir.ops.push_back(HeisenbergOp::make_measure(X(0), Z(0), false, MeasRecordIdx{0}));
    hir.ops.push_back(HeisenbergOp::make_tgate(X(2), 0, false));  // T

    REQUIRE(hir.num_ops() == 4);
    REQUIRE(hir.num_t_gates() == 3);  // All T_GATE ops (regardless of is_dagger)
}

TEST_CASE("HirModule with noise sites", "[hir]") {
    HirModule hir;
    hir.num_qubits = 2;

    // Add a noise site
    NoiseSite site;
    site.channels.push_back({X(0), 0, 0.1});
    hir.noise_sites.push_back(std::move(site));

    hir.ops.push_back(HeisenbergOp::make_noise(NoiseSiteIdx{0}));
    REQUIRE(hir.noise_sites.size() == 1);
    REQUIRE(hir.ops[0].noise_site_idx() == NoiseSiteIdx{0});
}

TEST_CASE("Tableau composition via then() and inverse()", "[hir]") {
    // after.then(before.inverse()) extracts the operator inserted between
    // `before` and `after`. Here `before = H` and `after = H; CNOT`, so the
    // composition should equal CNOT alone.
    stim::Tableau<kStimWidth> before(2);
    before.prepend_H_XZ(0);

    stim::Tableau<kStimWidth> after(2);
    after.prepend_H_XZ(0);
    after.prepend_ZCX(0, 1);

    auto inv_before = before.inverse();
    auto composed = after.then(inv_before);

    // CNOT: X_0 -> X_0 X_1, Z_1 -> Z_0 Z_1, X_1 -> X_1, Z_0 -> Z_0
    REQUIRE(composed.xs[0].xs[0] == true);
    REQUIRE(composed.xs[0].xs[1] == true);  // X_0 -> X_0 X_1
    REQUIRE(composed.xs[1].xs[1] == true);
    REQUIRE(composed.xs[1].xs[0] == false);  // X_1 -> X_1 (no change)
    REQUIRE(composed.zs[0].zs[0] == true);
    REQUIRE(composed.zs[0].zs[1] == false);  // Z_0 -> Z_0 (no change)
    REQUIRE(composed.zs[1].zs[0] == true);
    REQUIRE(composed.zs[1].zs[1] == true);  // Z_1 -> Z_0 Z_1
}
