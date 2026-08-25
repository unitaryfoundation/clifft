// Optimizer unit tests

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/numeric.h"

#include "test_helpers.h"

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace clifft;
using clifft::test::X;
using clifft::test::Z;

// Helper: parse a .stim circuit string through the front-end to produce HIR.
static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

static size_t count_ops(const HirModule& hir, OpType type) {
    return static_cast<size_t>(
        std::count_if(hir.ops.begin(), hir.ops.end(),
                      [type](const HeisenbergOp& op) { return op.op_type() == type; }));
}

// =============================================================================
// Peephole Fusion Pass -- front-end generated HIR
//
// These tests parse real circuit strings through the front-end to produce HIR,
// then run the optimizer. This makes the test circuits readable and exercises
// the full parse -> trace -> optimize pipeline.
//
// T+T fusion now absorbs the resulting S gate into downstream ops (no
// CLIFFORD_PHASE node is emitted). Both T gates are deleted.
// =============================================================================

TEST_CASE("Peephole: T plus T absorbed as virtual S", "[optimizer]") {
    auto hir = hir_from("T 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // S is absorbed offline -- no ops remain
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
    REQUIRE(pass.cancellations() == 0);
}

TEST_CASE("Peephole: T_dag plus T_dag absorbed as virtual S_dag", "[optimizer]") {
    auto hir = hir_from("T_DAG 0\nT_DAG 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
}

TEST_CASE("Peephole: T plus T_dag cancels to identity", "[optimizer]") {
    auto hir = hir_from("T 0\nT_DAG 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 1);
    REQUIRE(pass.fusions() == 0);
}

TEST_CASE("Peephole: T_dag plus T cancels to identity", "[optimizer]") {
    auto hir = hir_from("T_DAG 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: T slides past classical DETECTOR and commuting MEASURE", "[optimizer]") {
    // DETECTOR is classical (always transparent).
    // M 0 measures Z(0) -- same basis as T's Z(0) axis, so they commute.
    auto hir = hir_from("T 0\nM 0\nDETECTOR rec[-1]\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T slides past DETECTOR and MEASURE; T+T absorbed as S
    REQUIRE(hir.ops.size() == 2);  // MEASURE + DETECTOR
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::DETECTOR);
    REQUIRE(pass.fusions() == 1);
}

TEST_CASE("Peephole: anti-commuting MEASURE blocks T", "[optimizer]") {
    // H rotates the measurement basis: M after H measures X, not Z.
    // X-basis measure anti-commutes with Z-axis T.
    auto hir = hir_from("T 0\nH 0\nM 0\nH 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[2].op_type() == OpType::T_GATE);
}

TEST_CASE("Peephole: anti-commuting T blocks fusion", "[optimizer]") {
    // H between the Ts rotates the second T to X-axis, which anti-commutes
    // with the first T's Z-axis.
    auto hir = hir_from("T 0\nH 0\nT 0\nH 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // The X-axis T blocks the two Z-axis Ts from fusing
    REQUIRE(hir.ops.size() == 3);
}

TEST_CASE("Peephole: commuting T on different qubit does not block", "[optimizer]") {
    auto hir = hir_from("T 0\nT 1\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // Z(0) and Z(1) have disjoint support -- commute
    // T+T on q0 absorbed as S; only T on q1 remains
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
}

TEST_CASE("Peephole: NOISE channel blocks when anti-commuting", "[optimizer]") {
    // X_ERROR on qubit 0 anti-commutes with Z-axis T on qubit 0
    auto hir = hir_from("T 0\nX_ERROR(0.01) 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 3);
}

TEST_CASE("Peephole: NOISE channel does not block when commuting", "[optimizer]") {
    // Z_ERROR on qubit 0 commutes with Z-axis T on qubit 0
    auto hir = hir_from("T 0\nZ_ERROR(0.01) 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T+T absorbed as S; only NOISE remains
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
}

TEST_CASE("Peephole: terminal phase is removed across Pauli noise", "[optimizer]") {
    auto hir = hir_from("H 0\nR_Z(0.02) 0\nX_ERROR(0.01) 0\nM 0");
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: terminal phases are removed from broadcast layers", "[optimizer]") {
    auto hir = hir_from(
        "H 0 1 2\n"
        "R_Z(0.02) 0 1 2\n"
        "X_ERROR(0.01) 0 1 2\n"
        "M 0 1 2");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 6);
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(hir.ops[i].op_type() == OpType::NOISE);
        REQUIRE(hir.ops[i + 3].op_type() == OpType::MEASURE);
    }
    REQUIRE(pass.cancellations() == 3);
}

TEST_CASE("Peephole: terminal multi-Pauli phase is removed across correlated noise",
          "[optimizer]") {
    auto hir = hir_from(
        "H 0 1\n"
        "R_PAULI(0.02) Z0*Z1\n"
        "CORRELATED_ERROR(0.01) X0\n"
        "MPP Z0*Z1");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: terminal T is removed before measurement-reset", "[optimizer]") {
    auto hir = hir_from("H 0\nT 0\nY_ERROR(0.01) 0\nMR 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[2].op_type() == OpType::CONDITIONAL_PAULI);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: terminal phases cross disjoint measure-reset corrections", "[optimizer]") {
    const char* circuits[] = {
        "R 0 1\nH 0 1\nR_Z(0.3) 0 1\nX_ERROR(0.01) 0 1\nMR 0 1",
        "R 0 1\nH 0 1\nR_Z(0.3) 0 1\nMR 1 0",
        "R 0 1\nH 0 1\nR_Z(0.3) 0 1\nMR 0\nMR 1",
        "R 0 1\nH 0 1\nR_Z(0.3) 0 1\nMR 1\nMR 0",
        "R 0 1\nR_X(0.3) 0 1\nMRX 0 1",
        "R 0 1\nR_Y(0.3) 0 1\nMRY 0 1",
    };

    for (const char* circuit : circuits) {
        DYNAMIC_SECTION(circuit) {
            auto hir = hir_from(circuit);
            REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 2);

            PeepholeFusionPass pass;
            pass.run(hir);

            REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 0);
            REQUIRE(pass.cancellations() == 2);
            REQUIRE(clifft::sampling::plan_sampling(hir).peak_active_width == 0);
        }
    }
}

TEST_CASE("Peephole: terminal phase belonging only to a later reset target is removed",
          "[optimizer]") {
    auto hir = hir_from("R 0 1\nH 1\nR_Z(0.3) 1\nMR 0 1");
    REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 1);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 0);
    REQUIRE(pass.cancellations() == 1);
    REQUIRE(clifft::sampling::plan_sampling(hir).peak_active_width == 0);
}

TEST_CASE("Peephole: disjoint feedback controlled by a crossed measurement is transparent",
          "[optimizer]") {
    auto hir = hir_from(
        "R 0 1 2\n"
        "H 1\n"
        "R_Z(0.3) 1\n"
        "M 2\n"
        "DETECTOR rec[-1]\n"
        "CX rec[-1] 0\n"
        "M 1");
    REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 1);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 0);
    REQUIRE(count_ops(hir, OpType::CONDITIONAL_PAULI) == 4);
    REQUIRE(count_ops(hir, OpType::DETECTOR) == 1);
}

TEST_CASE("Peephole: reset phase elimination preserves inverted records and annotations",
          "[optimizer]") {
    auto hir = hir_from(
        "R 0 1\n"
        "H 0 1\n"
        "R_Z(0.3) 0 1\n"
        "MR !0 !1\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 0);
    REQUIRE(count_ops(hir, OpType::READOUT_NOISE) == 2);
    REQUIRE(count_ops(hir, OpType::DETECTOR) == 1);
    REQUIRE(count_ops(hir, OpType::OBSERVABLE) == 1);
}

TEST_CASE("Peephole: terminal phase elimination respects barriers", "[optimizer]") {
    SECTION("anti-commuting measurement") {
        auto hir = hir_from("R_Z(0.02) 0\nMX 0\nM 0");
        PeepholeFusionPass pass;
        pass.run(hir);
        REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    }

    SECTION("anti-commuting phase") {
        auto hir = hir_from("R_Z(0.02) 0\nR_X(0.03) 0\nM 0");
        PeepholeFusionPass pass;
        pass.run(hir);
        REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    }

    SECTION("anti-commuting conditional Pauli") {
        auto hir = hir_from("M 1\nR_Z(0.02) 0\nCX rec[-1] 0\nM 0");
        PeepholeFusionPass pass;
        pass.run(hir);
        REQUIRE(hir.ops[1].op_type() == OpType::PHASE_ROTATION);
    }

    SECTION("same-support commuting conditional Pauli") {
        auto hir = hir_from("M 1\nR_Z(0.02) 0\nCZ rec[-1] 0\nM 0");
        PeepholeFusionPass pass;
        pass.run(hir);
        REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 1);
    }

    SECTION("expectation value") {
        auto hir = hir_from("R_Z(0.02) 0\nEXP_VAL Z0\nM 0");
        PeepholeFusionPass pass;
        pass.run(hir);
        REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    }
}

TEST_CASE("Peephole: different axes do not fuse", "[optimizer]") {
    // H between the two Ts rotates the second to X-axis
    auto hir = hir_from("T 0\nH 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // Z(0) and X(0) are different axes and anti-commute
    REQUIRE(hir.ops.size() == 2);
}

TEST_CASE("Peephole: mirror circuit fully cancels", "[optimizer]") {
    // T0 T1 T1_dag T0_dag -- a mirror pattern
    // Both pairs cancel within a single while-loop iteration.
    auto hir = hir_from("T 0\nT 1\nT_DAG 1\nT_DAG 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 2);
}

TEST_CASE("Peephole: OBSERVABLE is transparent", "[optimizer]") {
    // OBSERVABLE_INCLUDE is a classical annotation -- T slides past it
    auto hir = hir_from("T 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T slides past OBSERVABLE and commuting MEASURE; T+T absorbed
    REQUIRE(hir.ops.size() == 2);  // MEASURE + OBSERVABLE
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::OBSERVABLE);
}

TEST_CASE("Peephole: CONDITIONAL_PAULI blocks when anti-commuting", "[optimizer]") {
    auto hir = hir_from("T 0\nM 1\nCX rec[-1] 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // Conditional X(0) anti-commutes with Z(0) T gates
    REQUIRE(hir.ops.size() == 4);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[3].op_type() == OpType::T_GATE);
}

TEST_CASE("Peephole: CONDITIONAL_PAULI allows when commuting", "[optimizer]") {
    // CZ rec[-1] 0 produces a conditional Z on qubit 0.
    // Z commutes with Z-axis T gates.
    auto hir = hir_from("T 0\nM 1\nCZ rec[-1] 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // Conditional Z(0) commutes with Z(0) T gates; T+T absorbed
    REQUIRE(hir.ops.size() == 2);  // MEASURE + CONDITIONAL
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::CONDITIONAL_PAULI);
}

TEST_CASE("Peephole: READOUT_NOISE is transparent", "[optimizer]") {
    // MZ(p) produces a MEASURE + READOUT_NOISE pair. T slides past both.
    auto hir = hir_from("T 0\nMZ(0.01) 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T slides past READOUT_NOISE and commuting Z-basis MEASURE; T+T absorbed
    REQUIRE(hir.ops.size() == 2);  // MEASURE + READOUT_NOISE
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[1].op_type() == OpType::READOUT_NOISE);
}

TEST_CASE("Peephole: empty HIR is a no-op", "[optimizer]") {
    HirModule hir(0, /*pauli_capacity=*/16);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 0);
    REQUIRE(pass.fusions() == 0);
}

TEST_CASE("Peephole: single T gate unchanged", "[optimizer]") {
    auto hir = hir_from("T 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
}

TEST_CASE("Peephole: multi-qubit Pauli axis T plus T absorbed", "[optimizer]") {
    // CX 0 1 entangles the qubits. T on qubit 1 then acts on a ZZ Pauli axis.
    // Two such Ts fuse and the resulting S is absorbed offline.
    auto hir = hir_from("CX 0 1\nT 1\nT 1");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
}

TEST_CASE("Peephole: three T gates produce one T after S absorption", "[optimizer]") {
    auto hir = hir_from("T 0\nT 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // First two fuse to S (absorbed), leaving one T gate.
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
}

// =============================================================================
// Peephole Fusion Pass -- manually constructed HIR
// =============================================================================

TEST_CASE("Peephole: sign inversion makes T behave as T_dag", "[optimizer]") {
    // T with negative sign has eff=-1, same as T_dag with positive sign
    // So T(sign=true) + T(sign=false) should cancel
    HirModule hir(1, /*pauli_capacity=*/16);

    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/false);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: sign inversion makes same-direction absorb", "[optimizer]") {
    // T(sign=true) has eff=-1, T_dag(sign=false) has eff=-1
    // total = -2 -> fuse to S_dag (absorbed)
    HirModule hir(1, /*pauli_capacity=*/16);

    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/false, /*dagger=*/true);

    PeepholeFusionPass pass;
    pass.run(hir);

    // S_dag absorbed, no ops remain
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
}

TEST_CASE("Peephole: S absorption propagates through downstream T", "[optimizer]") {
    // T 0; T 0; T 0 -> first two fuse to S (absorbed into 3rd T), leaving one T.
    // The absorbed S conjugates the third T's Pauli mask.
    // Since Z commutes with Z (same axis), the third T is unchanged.
    auto hir = hir_from("T 0\nT 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.stab_mask(hir.ops[0]) == Z(0));
}

TEST_CASE("Peephole: PHASE_ROTATION demotes to absorbed S and T gates", "[optimizer]") {
    // 0.5 half-turns = S gate -> absorbed (no ops remain)
    HirModule hir_s(1, 1);
    clifft::test::append_phase_rotation(hir_s, 0, Z(0), false, 0.5);
    PeepholeFusionPass pass_s;
    pass_s.run(hir_s);
    REQUIRE(hir_s.ops.empty());
    REQUIRE(pass_s.fusions() == 1);

    // 1.5 half-turns = S_dag gate -> absorbed (no ops remain)
    HirModule hir_sdag(1, 1);
    clifft::test::append_phase_rotation(hir_sdag, 0, Z(0), false, 1.5);
    PeepholeFusionPass pass_sdag;
    pass_sdag.run(hir_sdag);
    REQUIRE(hir_sdag.ops.empty());
    REQUIRE(pass_sdag.fusions() == 1);

    // 0.25 half-turns = T gate -> demoted to T_GATE
    auto hir_t = hir_from("R_Z(0.25) 0");
    PeepholeFusionPass pass_t;
    pass_t.run(hir_t);
    REQUIRE(hir_t.ops.size() == 1);
    REQUIRE(hir_t.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir_t.ops[0].is_dagger() == false);

    // 1.75 half-turns = T_dag gate -> demoted to T_GATE
    auto hir_tdag = hir_from("R_Z(1.75) 0");
    PeepholeFusionPass pass_tdag;
    pass_tdag.run(hir_tdag);
    REQUIRE(hir_tdag.ops.size() == 1);
    REQUIRE(hir_tdag.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir_tdag.ops[0].is_dagger() == true);
}

TEST_CASE("Peephole: rotation canonicalization uses the shared tolerance", "[optimizer]") {
    constexpr double inside = 0.5 + 0.5 * kRotationCanonicalizationTolerance;
    HirModule hir_inside(1, 1);
    clifft::test::append_phase_rotation(hir_inside, 0, Z(0), false, inside);
    PeepholeFusionPass inside_pass;
    inside_pass.run(hir_inside);
    CHECK(hir_inside.ops.empty());
    CHECK(inside_pass.fusions() == 1);

    constexpr double outside = 0.5 + 2.0 * kRotationCanonicalizationTolerance;
    HirModule hir_outside(1, 1);
    clifft::test::append_phase_rotation(hir_outside, 0, Z(0), false, outside);
    PeepholeFusionPass outside_pass;
    outside_pass.run(hir_outside);
    REQUIRE(hir_outside.ops.size() == 1);
    CHECK(hir_outside.ops[0].op_type() == OpType::PHASE_ROTATION);
    CHECK(hir_outside.ops[0].alpha() == outside);
    CHECK(outside_pass.fusions() == 0);
    CHECK(outside_pass.cancellations() == 0);
}

TEST_CASE("Peephole: Pauli-valued rotation updates downstream signs and frame", "[optimizer]") {
    HirModule hir(1, 2);
    hir.final_tableau.emplace(1);
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 1.0);
    clifft::test::append_measure(hir, X(0), 0, false, MeasRecordIdx{0});
    const HirModule reference = hir_from("Z 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    CHECK(hir.sign(hir.ops[0]));
    CHECK(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: fused rotations absorb a Pauli residue", "[optimizer]") {
    HirModule hir(1, 2);
    hir.final_tableau.emplace(1);
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 0.3);
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 0.7);
    const HirModule reference = hir_from("Z 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    CHECK(hir.ops.empty());
    CHECK(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: Pauli absorption composes with an existing Clifford frame", "[optimizer]") {
    HirModule hir = hir_from("H 0\nR_Z(0.3) 0\nR_Z(0.7) 0");
    const HirModule reference = hir_from("H 0\nZ 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    CHECK(hir.ops.empty());
    CHECK(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: Pauli residue handles signs periods and tolerance", "[optimizer]") {
    struct RotationCase {
        double alpha;
        bool sign;
    };
    const RotationCase cases[] = {
        {1.0, false},
        {-1.0, false},
        {3.0, false},
        {1.0, true},
        {1.0 + 0.5 * kRotationCanonicalizationTolerance, false},
    };
    const HirModule reference = hir_from("Z 0");

    for (const auto& test_case : cases) {
        CAPTURE(test_case.alpha, test_case.sign);
        HirModule hir(1, 1);
        hir.final_tableau.emplace(1);
        clifft::test::append_phase_rotation(hir, 0, Z(0), test_case.sign, test_case.alpha);

        PeepholeFusionPass pass;
        pass.run(hir);

        CHECK(hir.ops.empty());
        CHECK(pass.fusions() == 1);
        REQUIRE(hir.final_tableau == reference.final_tableau);
    }

    constexpr double outside = 1.0 + 2.0 * kRotationCanonicalizationTolerance;
    HirModule outside_hir(1, 1);
    outside_hir.final_tableau.emplace(1);
    clifft::test::append_phase_rotation(outside_hir, 0, Z(0), false, outside);
    PeepholeFusionPass outside_pass;
    outside_pass.run(outside_hir);

    REQUIRE(outside_hir.ops.size() == 1);
    CHECK(outside_hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    CHECK(outside_hir.ops[0].alpha() == outside);
    CHECK(outside_pass.fusions() == 0);
}

TEST_CASE("Peephole: Pauli absorption updates instrument masks", "[optimizer]") {
    HirModule hir(1, 3);
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 1.0);

    InstrumentSite site;
    site.destination_flip_mask =
        hir.claim_side_mask([](MutablePauliMaskView slot) { slot.x().bit_set(0, true); });
    hir.instrument_sites.push_back(site);
    hir.append_instrument(InstrumentSiteIdx{0},
                          [](MutablePauliMaskView slot) { slot.x().bit_set(0, true); });

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::INSTRUMENT);
    CHECK(hir.sign(hir.ops[0]));
    const PauliMaskView flip = hir.pauli_masks.at(hir.instrument_sites[0].destination_flip_mask);
    CHECK(flip.sign());
}

TEST_CASE("Peephole: Pauli absorption supports wide multi-word axes", "[optimizer]") {
    HirModule hir(130, 1);
    hir.final_tableau.emplace(130);
    hir.append_phase_rotation(1.0, [](MutablePauliMaskView slot) {
        slot.x().bit_set(0, true);
        slot.z().bit_set(64, true);
        slot.x().bit_set(129, true);
        slot.z().bit_set(129, true);
    });
    const HirModule reference = hir_from("X 0\nZ 64\nY 129");

    PeepholeFusionPass pass;
    pass.run(hir);

    CHECK(hir.ops.empty());
    CHECK(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: square-root product rotations use fusion and absorption", "[optimizer]") {
    HirModule hir_spp(2, 1);
    hir_spp.final_tableau.emplace(2);
    clifft::test::append_phase_rotation(hir_spp, 0, X(0) | X(1), false, 0.5);

    PeepholeFusionPass pass_spp;
    pass_spp.run(hir_spp);
    REQUIRE(hir_spp.ops.empty());

    HirModule hir_cancel(2, 2);
    hir_cancel.final_tableau.emplace(2);
    clifft::test::append_phase_rotation(hir_cancel, 0, X(0) | X(1), false, 0.5);
    clifft::test::append_phase_rotation(hir_cancel, 0, X(0) | X(1), false, -0.5);

    PeepholeFusionPass pass_cancel;
    pass_cancel.run(hir_cancel);
    REQUIRE(hir_cancel.ops.empty());
    REQUIRE(pass_cancel.cancellations() == 1);
}

TEST_CASE("Peephole: S absorption conjugates anti-commuting downstream measure", "[optimizer]") {
    // T 0; T 0; H 0; M 0
    // T+T fuses to S on Z(0). The S is absorbed downstream.
    // The downstream MEASURE (after H) measures X(0), which anti-commutes
    // with Z(0). S conjugation: S_Z^dag X S_Z = Y. So the measure axis
    // should change from X to Y.
    auto hir = hir_from("T 0\nT 0\nH 0\nM 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    // After conjugation, X(0) -> Y(0): both X and Z bits set
    auto y_destab = X(0);
    auto y_stab = Z(0);
    REQUIRE(hir.destab_mask(hir.ops[0]) == y_destab);
    REQUIRE(hir.stab_mask(hir.ops[0]) == y_stab);
}

TEST_CASE("Peephole: S absorption conjugates noise and conditional Pauli", "[optimizer]") {
    // T 0; T 0; X_ERROR(0.1) 0; MX 0; CX rec[-1] 0
    //
    // T+T fuses to S on virtual Z(0). Downstream:
    //   X_ERROR -> noise channel on X(0), anti-commutes with Z(0)
    //   MX 0    -> measures X(0), anti-commutes with Z(0)
    //   CX      -> conditional X(0), anti-commutes with Z(0)
    //
    // S_Z^dag X S_Z = -Y, so all three must become Y(0) with sign=true.
    auto hir = hir_from("T 0\nT 0\nX_ERROR(0.1) 0\nMX 0\nCX rec[-1] 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T gates eradicated; NOISE + MEASURE + CONDITIONAL remain
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
    REQUIRE(hir.ops[2].op_type() == OpType::CONDITIONAL_PAULI);

    // Check noise channel conjugation: X(0) -> Y(0)
    auto site_idx = static_cast<uint32_t>(hir.ops[0].noise_site_idx());
    const auto& ch = hir.noise_sites[site_idx].channels[0];
    auto ch_view = hir.noise_channel_masks.at(ch.mask);
    CHECK(ch_view.x().bit_get(0));  // X bit set
    CHECK(ch_view.z().bit_get(0));  // Z bit set -> Y

    // Check MEASURE conjugation: X(0) -> Y(0) with sign
    CHECK(hir.destab_mask(hir.ops[1]).bit_get(0));
    CHECK(hir.stab_mask(hir.ops[1]).bit_get(0));
    CHECK(hir.sign(hir.ops[1]) == true);  // -Y

    // Check CONDITIONAL_PAULI conjugation: X(0) -> Y(0) with sign
    CHECK(hir.destab_mask(hir.ops[2]).bit_get(0));
    CHECK(hir.stab_mask(hir.ops[2]).bit_get(0));
    CHECK(hir.sign(hir.ops[2]) == true);  // -Y
}

// =============================================================================
// Negative-sign T fusion must preserve the state ray.
// =============================================================================

TEST_CASE("Peephole: negative-sign T plus T preserves the state ray", "[optimizer]") {
    // X conjugates Z -> -Z, so both T gates see -Z axis (sign=true).
    // T(-Z) = exp(i*pi/4) * T_dag(+Z), so T(-Z)+T(-Z) = exp(i*pi/2) * S_dag(+Z) = i * S_dag.
    HirModule hir(1, /*pauli_capacity=*/16);
    hir.final_tableau.emplace(1);

    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    const HirModule reference = hir_from("S_DAG 0");
    PeepholeFusionPass pass;
    pass.run(hir);

    // Both T gates absorbed (S absorbed into tableau)
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: negative-sign T_dag plus T_dag preserves the state ray", "[optimizer]") {
    // T_dag(-Z) = exp(-i*pi/4) * T(+Z), two of them: exp(-i*pi/2) * S(+Z) = -i * S.
    HirModule hir(1, /*pauli_capacity=*/16);
    hir.final_tableau.emplace(1);

    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true, /*dagger=*/true);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true, /*dagger=*/true);
    const HirModule reference = hir_from("S 0");
    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: mixed-sign T cancellation preserves the state ray", "[optimizer]") {
    // T(+Z) + T(-Z): effective_angles sum to 0 (cancellation), but
    // T(-Z) = exp(i*pi/4) * T_dag(+Z), so the physical result is
    // T(+Z) * exp(i*pi/4) * T_dag(+Z) = exp(i*pi/4) * I.
    HirModule hir(1, /*pauli_capacity=*/16);
    hir.final_tableau.emplace(1);

    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/false);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    const HirModule reference = hir_from("I 0");
    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.cancellations() == 1);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Peephole: S absorption creates negative T that subsequently fuses", "[optimizer]") {
    // T 0; T 0; H 0; T 0; T 0
    // First pair fuses to S on Z(0). H changes frame.
    // S_Z absorption conjugates the downstream T gates on X(0) to Y(0) with sign=true.
    // Those two negative-sign T gates must then fuse correctly with proper phase.
    auto hir = hir_from("T 0\nT 0\nH 0\nT 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // All 4 T gates should be absorbed (2 fusions)
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 2);
}

// --- Pass registry tripwire tests ---
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pass_registry.h"

TEST_CASE("Pass registry: all entries resolve via factory") {
    for (size_t i = 0; i < clifft::kNumRegisteredPasses; ++i) {
        const auto& info = clifft::kRegisteredPasses[i];
        auto pass = clifft::make_hir_pass(info.name);
        REQUIRE(pass != nullptr);
    }
}

TEST_CASE("Pass registry: default managers use registry") {
    auto hpm = clifft::default_hir_pass_manager();
    // Smoke test: run on a trivial circuit
    auto circuit = clifft::parse("H 0\nCNOT 0 1\nM 0\nM 1");
    auto hir = clifft::trace(circuit);
    hpm.run(hir);
    REQUIRE(hir.num_qubits == 2);
}

TEST_CASE("Pass registry: trajectory compatibility requires both guarantees") {
    constexpr clifft::PassInfo record_only{
        .name = "record-only",
        .default_enabled = true,
        .record_order = clifft::kPreservesRecordOrder,
        .instrument_prefix = clifft::kMayChangeInstrumentPrefix,
    };
    constexpr clifft::PassInfo prefix_only{
        .name = "prefix-only",
        .default_enabled = true,
        .record_order = clifft::kBreaksRecordOrder,
        .instrument_prefix = clifft::kPreservesInstrumentPrefix,
    };
    static_assert(!clifft::is_trajectory_compatible(record_only));
    static_assert(!clifft::is_trajectory_compatible(prefix_only));

    const std::vector<std::string_view> prefix_stable = {"PeepholeFusionPass"};
    const std::vector<std::string_view> may_change_prefix = {
        "StatevectorSqueezePass", "RemoveNoisePass", "DropNonUnitaryPass"};

    for (const auto& info : clifft::kRegisteredPasses) {
        const bool expected_stable =
            std::find(prefix_stable.begin(), prefix_stable.end(), info.name) != prefix_stable.end();
        const bool expected_unstable = std::find(may_change_prefix.begin(), may_change_prefix.end(),
                                                 info.name) != may_change_prefix.end();
        REQUIRE(expected_stable != expected_unstable);
        CHECK(info.instrument_prefix.preserved == expected_stable);
        CHECK(clifft::is_trajectory_compatible(info) ==
              (info.record_order.preserved && expected_stable));
    }
}

TEST_CASE("Pass registry: JSON round-trip is valid") {
    std::string json = clifft::pass_registry_json();
    REQUIRE(json.front() == '[');
    REQUIRE(json.back() == ']');
    REQUIRE(json.find("PeepholeFusionPass") != std::string::npos);
    REQUIRE(json.find("StatevectorSqueezePass") != std::string::npos);
    REQUIRE(json.find("RemoveNoisePass") != std::string::npos);
    REQUIRE(json.find("DropNonUnitaryPass") != std::string::npos);
    REQUIRE(json.find("preserves_instrument_prefix") != std::string::npos);
}

// =============================================================================
// EXP_VAL barrier tests
// =============================================================================

TEST_CASE("Commutation: EXP_VAL blocks T_GATE swap", "[optimizer][exp_val]") {
    auto hir = hir_from("T 0\nEXP_VAL Z0");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: EXP_VAL blocks MEASURE swap", "[optimizer][exp_val]") {
    auto hir = hir_from("EXP_VAL Z0\nM 0");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: EXP_VAL blocks NOISE swap", "[optimizer][exp_val]") {
    auto hir = hir_from("EXP_VAL Z0\nX_ERROR(0.1) 0");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: EXP_VAL blocks commuting NOISE swap", "[optimizer][exp_val]") {
    auto hir = hir_from("EXP_VAL Z0\nZ_ERROR(0.1) 0");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: EXP_VAL blocks disjoint NOISE swap", "[optimizer][exp_val]") {
    auto hir = hir_from("EXP_VAL Z0\nZ_ERROR(0.1) 1");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: NOISE also cannot cross EXP_VAL from the left", "[optimizer][exp_val]") {
    auto hir = hir_from("Z_ERROR(0.1) 0\nEXP_VAL Z0");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Commutation: EXP_VAL blocks even commuting Paulis", "[optimizer][exp_val]") {
    // Z0 and Z1 commute, but EXP_VAL is a positional barrier
    auto hir = hir_from("T 0\nEXP_VAL Z1");
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
}

TEST_CASE("Peephole: T-gates do not fuse across EXP_VAL", "[optimizer][exp_val]") {
    // T 0 ... EXP_VAL Z0 ... T 0 should NOT fuse
    auto hir = hir_from("T 0\nEXP_VAL Z0\nT 0");

    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[2].op_type() == OpType::T_GATE);

    PeepholeFusionPass pass;
    pass.run(hir);

    // Both T gates and the EXP_VAL must survive
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(pass.cancellations() == 0);
    REQUIRE(pass.fusions() == 0);
}

TEST_CASE("Squeeze: measurement does not bubble past EXP_VAL", "[optimizer][exp_val]") {
    // T 0 activates qubit 0; EXP_VAL probes; M 0 measures.
    // Without the barrier, the squeeze pass would try to bubble M leftward.
    auto hir = hir_from("T 0\nEXP_VAL Z0\nM 0");

    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[2].op_type() == OpType::MEASURE);

    StatevectorSqueezePass pass;
    pass.run(hir);

    // Order must be preserved: T, EXP_VAL, MEASURE
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[2].op_type() == OpType::MEASURE);
}

TEST_CASE("Squeeze: measurement does not bubble past NOISE and EXP_VAL", "[optimizer][exp_val]") {
    auto hir = hir_from("T 0\nZ_ERROR(0.1) 1\nEXP_VAL Z0\nM 0");

    REQUIRE(hir.ops.size() == 4);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[2].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[3].op_type() == OpType::MEASURE);

    StatevectorSqueezePass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 4);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[2].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[3].op_type() == OpType::MEASURE);
}

TEST_CASE("Peephole: virtual S conjugation updates EXP_VAL masks", "[optimizer][exp_val]") {
    // Circuit: T 0, T 0, EXP_VAL X0
    // T+T fuses to virtual S on Z0. The S conjugation must update the
    // downstream EXP_VAL's Pauli mask: S^dag X S = Y, so the rewound
    // X0 should become Y0 (both destab and stab bits set on qubit 0).
    auto hir = hir_from("T 0\nT 0\nEXP_VAL X0");

    REQUIRE(hir.ops.size() == 3);

    PeepholeFusionPass pass;
    pass.run(hir);

    // T+T fused into virtual S (absorbed), leaving only EXP_VAL
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::EXP_VAL);
    REQUIRE(pass.fusions() == 1);

    // S^dag X S = Y: both X and Z bits set on qubit 0
    REQUIRE(hir.destab_mask(hir.ops[0]) == X(0));
    REQUIRE(hir.stab_mask(hir.ops[0]) == Z(0));
}

TEST_CASE("Peephole: commuting NOISE does not bypass EXP_VAL barrier", "[optimizer][exp_val]") {
    auto hir = hir_from("T 0\nEXP_VAL Z0\nZ_ERROR(0.01) 1\nT 0");

    REQUIRE(hir.ops.size() == 4);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 4);
    REQUIRE(hir.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir.ops[1].op_type() == OpType::EXP_VAL);
    REQUIRE(hir.ops[2].op_type() == OpType::NOISE);
    REQUIRE(hir.ops[3].op_type() == OpType::T_GATE);
    REQUIRE(pass.cancellations() == 0);
    REQUIRE(pass.fusions() == 0);
}

// =============================================================================
// Projective correctness of S absorption -- dense stim oracle
// =============================================================================

namespace {

using PauliChannel = std::vector<double>;

PauliString basis_pauli(size_t n, size_t basis_index) {
    PauliString result(static_cast<uint32_t>(n));
    for (uint32_t q = 0; q < n; ++q) {
        const uint32_t code = static_cast<uint32_t>((basis_index >> (2 * q)) & 3U);
        result.set_pauli(q, (code & 1U) != 0, (code & 2U) != 0);
    }
    result.set_sign(false);
    return result;
}

size_t basis_index(PauliStringView pauli) {
    size_t result = 0;
    for (uint32_t q = 0; q < pauli.num_qubits(); ++q) {
        const size_t code = static_cast<size_t>(pauli.x().bit_get(q)) |
                            (static_cast<size_t>(pauli.z().bit_get(q)) << 1U);
        result |= code << (2 * q);
    }
    return result;
}

PauliString op_axis(const HirModule& hir, const HeisenbergOp& op) {
    PauliString result(hir.num_qubits);
    const MaskView x = hir.destab_mask(op);
    const MaskView z = hir.stab_mask(op);
    for (uint32_t q = 0; q < hir.num_qubits; ++q) {
        result.set_pauli(q, x.bit_get(q), z.bit_get(q));
    }
    result.set_sign(hir.sign(op));
    return result;
}

PauliChannel pauli_channel_hir_value(const HirModule& hir) {
    const size_t basis_count = size_t{1} << (2 * hir.num_qubits);
    PauliChannel channel(basis_count * basis_count, 0.0);
    for (size_t input_index = 0; input_index < basis_count; ++input_index) {
        std::vector<double> coefficients(basis_count, 0.0);
        coefficients[input_index] = 1.0;

        for (const HeisenbergOp& op : hir.ops) {
            REQUIRE((op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION));
            PauliString axis = op_axis(hir, op);
            double alpha;
            if (op.op_type() == OpType::T_GATE) {
                alpha = op.is_dagger() ? 1.75 : 0.25;
            } else {
                alpha = hir.sign(op) ? -op.alpha() : op.alpha();
                axis.set_sign(false);
            }
            const double cosine = std::cos(alpha * std::numbers::pi);
            const double sine = std::sin(alpha * std::numbers::pi);
            std::vector<double> rotated(basis_count, 0.0);
            for (size_t term = 0; term < basis_count; ++term) {
                if (coefficients[term] == 0.0) {
                    continue;
                }
                const PauliString pauli = basis_pauli(hir.num_qubits, term);
                if (axis.view().commutes(pauli.view())) {
                    rotated[term] += coefficients[term];
                    continue;
                }
                rotated[term] += coefficients[term] * cosine;
                PauliString product = axis;
                product.right_multiply(pauli.view());
                uint32_t y_phase = 0;
                for (uint32_t q = 0; q < product.num_qubits(); ++q) {
                    y_phase += product.x().bit_get(q) && product.z().bit_get(q);
                }
                y_phase &= 3U;
                const uint32_t phase_delta = (product.phase() - y_phase) & 3U;
                REQUIRE((phase_delta == 1U || phase_delta == 3U));
                const double product_sign = phase_delta == 1U ? 1.0 : -1.0;
                rotated[basis_index(product.view())] += coefficients[term] * sine * product_sign;
            }
            coefficients = std::move(rotated);
        }

        for (size_t term = 0; term < basis_count; ++term) {
            if (coefficients[term] == 0.0) {
                continue;
            }
            const PauliString pauli = basis_pauli(hir.num_qubits, term);
            const PauliString mapped = hir.final_tableau->apply(pauli.view());
            channel[input_index * basis_count + basis_index(mapped.view())] +=
                coefficients[term] * (mapped.sign() ? -1.0 : 1.0);
        }
    }
    return channel;
}

void require_channel_equivalence(const PauliChannel& actual, const PauliChannel& expected,
                                 double tolerance) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], tolerance));
    }
}

}  // namespace

TEST_CASE("Peephole: pass preserves projective HIR value on random circuits", "[optimizer]") {
    // The Pauli channel discards global phase while retaining the full
    // conjugation action that the pass must preserve. The gate mix is chosen
    // so the trials collectively reach every S absorption call site
    // (T+T fusion, rotation fusion to S/S_dag, standalone S-angle demotion)
    // and their interaction with sign normalization and downstream
    // conjugation; the aggregate fusion count below keeps that property
    // from silently eroding if the grammar or seed changes.
    std::mt19937_64 rng(2026);
    const char* single_qubit[] = {"H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"};
    const double angles[] = {0.25, 0.25, 0.5, 1.5, 0.75, 1.75, 0.1};
    size_t total_fusions = 0;

    for (int trial = 0; trial < 200; ++trial) {
        CAPTURE(trial);
        const size_t n = 2 + static_cast<size_t>(trial % 2);
        std::string src;
        for (int depth = 0; depth < 25; ++depth) {
            const int kind = static_cast<int>(rng() % 4);
            const int q = static_cast<int>(rng() % n);
            if (kind == 0) {
                src += std::string(single_qubit[rng() % 8]) + " " + std::to_string(q) + "\n";
            } else if (kind == 1) {
                const int q2 = static_cast<int>(rng() % n);
                if (q2 == q)
                    continue;
                src += "CX " + std::to_string(q) + " " + std::to_string(q2) + "\n";
            } else if (kind == 2) {
                src += "R_Z(" + std::to_string(angles[rng() % 7]) + ") " + std::to_string(q) + "\n";
            } else {
                src += std::string((rng() & 1) ? "T " : "T_DAG ") + std::to_string(q) + "\n";
            }
        }
        CAPTURE(src);

        auto hir = hir_from(src.c_str());
        const PauliChannel before_channel = pauli_channel_hir_value(hir);
        PeepholeFusionPass pass;
        pass.run(hir);
        const PauliChannel after_channel = pauli_channel_hir_value(hir);
        require_channel_equivalence(after_channel, before_channel, 1e-9);
        total_fusions += pass.fusions();
    }

    // Vacuity guard: the fuzz only validates S absorption if fusions occur.
    REQUIRE(total_fusions > 50);
}

TEST_CASE("Peephole: S absorption on wide multi-word Pauli axes", "[optimizer]") {
    // The reference decomposes the same Pauli-product S action into named
    // Clifford gates, so the frontend constructs its tableau independently
    // of the optimizer's multi-word update.
    const std::string prefix = "H 0\nCX 0 63\nS 64\nH 69\n";
    const std::string basis_change = "H 0\nH_YZ 63\nH 69\nCX 63 0\nCX 64 0\nCX 69 0\n";
    const std::string uncompute = "CX 69 0\nCX 64 0\nCX 63 0\nH 69\nH_YZ 63\nH 0\n";

    struct FusionCase {
        const char* t_gate;
        const char* axis;
        const char* fused_gate;
    };
    const FusionCase cases[] = {
        {"TPP", "X0*Y63*Z64*X69", "S"},
        {"TPP_DAG", "X0*Y63*Z64*X69", "S_DAG"},
        {"TPP", "!X0*Y63*Z64*X69", "S_DAG"},
        {"TPP_DAG", "!X0*Y63*Z64*X69", "S"},
    };

    for (const auto& test_case : cases) {
        CAPTURE(test_case.t_gate, test_case.axis, test_case.fused_gate);
        const std::string source = prefix + test_case.t_gate + " " + test_case.axis + "\n" +
                                   test_case.t_gate + " " + test_case.axis + "\n";
        const std::string reference_source =
            prefix + basis_change + test_case.fused_gate + " 0\n" + uncompute;

        auto hir = hir_from(source.c_str());
        const auto reference = hir_from(reference_source.c_str());
        PeepholeFusionPass pass;
        pass.run(hir);

        REQUIRE(pass.fusions() == 1);
        REQUIRE(hir.ops.empty());
        REQUIRE(reference.ops.empty());
        REQUIRE(hir.final_tableau == reference.final_tableau);
    }
}
