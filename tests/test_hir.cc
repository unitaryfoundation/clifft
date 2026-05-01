// HIR (Heisenberg IR) unit tests

#include "clifft/frontend/hir.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <tuple>
#include <utility>

using namespace clifft;
using clifft::test::MaskBuf;
using clifft::test::pauli_masks;
using clifft::test::X;
using clifft::test::Z;

// =============================================================================
// HirModule builder + accessor tests
// =============================================================================

// HeisenbergOp size is pinned to 16 bytes by static_assert in hir.h. We do
// not duplicate that assertion here.

TEST_CASE("HirModule::append_tgate stores Pauli in arena", "[hir]") {
    SECTION("T gate with Z on qubit 0") {
        HirModule hir(64, 1);
        auto [destab, stab] = pauli_masks("Z");
        auto& op = hir.append_tgate(MaskBuf(destab), MaskBuf(stab), /*sign=*/false);

        REQUIRE(op.op_type() == OpType::T_GATE);
        REQUIRE(hir.destab_mask(op) == 0);   // No X
        REQUIRE(hir.stab_mask(op) == Z(0));  // Z on qubit 0
        REQUIRE(hir.sign(op) == false);
        REQUIRE(op.is_dagger() == false);
    }

    SECTION("T_dag gate with X on qubit 1, negative sign") {
        HirModule hir(64, 1);
        auto& op = hir.append_tgate(MaskBuf(X(1)), MaskBuf(0), /*sign=*/true, /*dagger=*/true);

        REQUIRE(op.op_type() == OpType::T_GATE);
        REQUIRE(hir.destab_mask(op) == X(1));
        REQUIRE(hir.stab_mask(op) == 0);
        REQUIRE(hir.sign(op) == true);
        REQUIRE(op.is_dagger() == true);
    }

    SECTION("T gate with Y on qubit 2 (both X and Z bits set)") {
        HirModule hir(64, 1);
        auto [destab, stab] = pauli_masks("Y__");
        auto& op = hir.append_tgate(MaskBuf(destab), MaskBuf(stab), false);

        REQUIRE(hir.destab_mask(op) == X(2));
        REQUIRE(hir.stab_mask(op) == Z(2));
    }
}

TEST_CASE("HirModule::set_pauli replaces masks and sign", "[hir]") {
    HirModule hir(64, 1);
    auto& op = hir.append_tgate(MaskBuf(0), MaskBuf(Z(0)), /*sign=*/false);
    REQUIRE(hir.stab_mask(op) == Z(0));
    REQUIRE(hir.sign(op) == false);

    hir.set_pauli(op, MaskBuf(X(1)), MaskBuf(Z(2)), true);
    REQUIRE(hir.destab_mask(op) == X(1));
    REQUIRE(hir.stab_mask(op) == Z(2));
    REQUIRE(hir.sign(op) == true);
    REQUIRE(op.op_type() == OpType::T_GATE);
    REQUIRE(op.is_dagger() == false);
}

TEST_CASE("HirModule mask access supports bitwise inspection", "[hir]") {
    HirModule hir(64, 2);
    hir.append_tgate(MaskBuf(X(1) | X(3)), MaskBuf(Z(0) | Z(2)), false);  // X1 X3 Z0 Z2
    hir.append_tgate(MaskBuf(X(2) | X(3)), MaskBuf(Z(0) | Z(1)), false);  // X2 X3 Z0 Z1

    auto destab1 = hir.destab_mask(hir.ops[0]);
    auto stab1 = hir.stab_mask(hir.ops[0]);
    auto destab2 = hir.destab_mask(hir.ops[1]);
    auto stab2 = hir.stab_mask(hir.ops[1]);

    // destab1 & stab2 = X1 X3 & Z0 Z1 = bit 1 only
    int popcount_d1_s2 = 0;
    for (uint32_t i = 0; i < destab1.num_words(); ++i) {
        popcount_d1_s2 += std::popcount(destab1.words[i] & stab2.words[i]);
    }
    REQUIRE(popcount_d1_s2 == 1);

    // stab1 & destab2 = Z0 Z2 & X2 X3 = bit 2 only
    int popcount_s1_d2 = 0;
    for (uint32_t i = 0; i < stab1.num_words(); ++i) {
        popcount_s1_d2 += std::popcount(stab1.words[i] & destab2.words[i]);
    }
    REQUIRE(popcount_s1_d2 == 1);

    // XOR: X3 cancels, X1 and X2 remain
    REQUIRE((destab1.words[0] ^ destab2.words[0]) == (X(1) | X(2)));
}

TEST_CASE("HirModule::append_measure", "[hir]") {
    SECTION("Measurement with record index") {
        HirModule hir(64, 1);
        auto& op = hir.append_measure(MaskBuf(X(0)), MaskBuf(0), /*sign=*/false, MeasRecordIdx{5});

        REQUIRE(op.op_type() == OpType::MEASURE);
        REQUIRE(hir.destab_mask(op) == X(0));
        REQUIRE(hir.stab_mask(op) == 0);
        REQUIRE(op.meas_record_idx() == MeasRecordIdx{5});
    }

    SECTION("Multi-qubit measurement") {
        HirModule hir(64, 1);
        auto& op = hir.append_measure(MaskBuf(X(0) | X(1)), MaskBuf(0), false, MeasRecordIdx{10});
        REQUIRE(op.meas_record_idx() == MeasRecordIdx{10});
    }
}

TEST_CASE("HirModule::append_conditional", "[hir]") {
    HirModule hir(64, 1);
    auto& op =
        hir.append_conditional(MaskBuf(X(0)), MaskBuf(0), /*sign=*/false, ControllingMeasIdx{7});

    REQUIRE(op.op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(hir.destab_mask(op) == X(0));
    REQUIRE(op.controlling_meas() == ControllingMeasIdx{7});
}

TEST_CASE("pauli_masks helper", "[hir][helper]") {
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
        auto [d, s] = pauli_masks("XYZ");
        REQUIRE(d == (X(2) | X(1)));
        REQUIRE(s == (Z(1) | Z(0)));

        std::tie(d, s) = pauli_masks("IXZI");
        REQUIRE(d == X(2));
        REQUIRE(s == Z(1));
    }
}

// =============================================================================
// stim::Tableau API used by the Front-End
// =============================================================================

TEST_CASE("stim::Tableau identity and Hadamard", "[hir]") {
    stim::Tableau<kStimWidth> tab(4);

    REQUIRE(tab.num_qubits == 4);

    auto x0 = tab.xs[0];
    REQUIRE(x0.xs[0] == true);
    REQUIRE(x0.zs[0] == false);
    REQUIRE(x0.sign == false);

    auto z0 = tab.zs[0];
    REQUIRE(z0.xs[0] == false);
    REQUIRE(z0.zs[0] == true);
    REQUIRE(z0.sign == false);

    tab.prepend_H_XZ(0);
    auto x0_after = tab.xs[0];
    auto z0_after = tab.zs[0];

    REQUIRE(x0_after.xs[0] == false);
    REQUIRE(x0_after.zs[0] == true);
    REQUIRE(z0_after.xs[0] == true);
    REQUIRE(z0_after.zs[0] == false);
}

// =============================================================================
// HirModule integration
// =============================================================================

TEST_CASE("HirModule construction and accessors", "[hir]") {
    HirModule hir(4, 4);

    REQUIRE(hir.num_ops() == 0);
    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.global_weight == std::complex<double>(1.0, 0.0));

    hir.append_tgate(MaskBuf(X(0)), MaskBuf(0), false);        // T
    hir.append_tgate(MaskBuf(X(1)), MaskBuf(0), false, true);  // T_dag
    hir.append_measure(MaskBuf(X(0)), MaskBuf(Z(0)), false, MeasRecordIdx{0});
    hir.append_tgate(MaskBuf(X(2)), MaskBuf(0), false);  // T

    REQUIRE(hir.num_ops() == 4);
    REQUIRE(hir.num_t_gates() == 3);
}

TEST_CASE("HirModule with noise sites", "[hir]") {
    HirModule hir(2, 0);

    NoiseSite site;
    site.channels.push_back({PauliBitMask(X(0)), PauliBitMask{}, 0.1});
    hir.noise_sites.push_back(std::move(site));

    hir.append_noise(NoiseSiteIdx{0});
    REQUIRE(hir.noise_sites.size() == 1);
    REQUIRE(hir.ops[0].noise_site_idx() == NoiseSiteIdx{0});
}

TEST_CASE("Tableau composition via then() and inverse()", "[hir]") {
    stim::Tableau<kStimWidth> before(2);
    before.prepend_H_XZ(0);

    stim::Tableau<kStimWidth> after(2);
    after.prepend_H_XZ(0);
    after.prepend_ZCX(0, 1);

    auto inv_before = before.inverse();
    auto composed = after.then(inv_before);

    REQUIRE(composed.xs[0].xs[0] == true);
    REQUIRE(composed.xs[0].xs[1] == true);
    REQUIRE(composed.xs[1].xs[1] == true);
    REQUIRE(composed.xs[1].xs[0] == false);
    REQUIRE(composed.zs[0].zs[0] == true);
    REQUIRE(composed.zs[0].zs[1] == false);
    REQUIRE(composed.zs[1].zs[0] == true);
    REQUIRE(composed.zs[1].zs[1] == true);
}
