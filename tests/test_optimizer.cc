// Optimizer unit tests

#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/frontend/hir.h"
#include "ucc/optimizer/peephole.h"
#include "ucc/util/constants.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>

using namespace ucc;
using ucc::test::X;
using ucc::test::Z;

// Helper: parse a .stim circuit string through the front-end to produce HIR.
static HirModule hir_from(const char* text) {
    return ucc::trace(ucc::parse(text));
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
    HirModule hir;
    hir.num_qubits = 0;

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
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/false));

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 0);
    REQUIRE(pass.cancellations() == 1);
}

TEST_CASE("Peephole: sign inversion makes same-direction absorb", "[optimizer]") {
    // T(sign=true) has eff=-1, T_dag(sign=false) has eff=-1
    // total = -2 -> fuse to S_dag (absorbed)
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/false, /*dagger=*/true));

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
    REQUIRE(hir.ops[0].stab_mask() == Z(0));
}

TEST_CASE("Peephole: T plus T fusion leaves global_weight unchanged", "[optimizer]") {
    auto hir = hir_from("T 0\nT 0");
    auto initial_weight = hir.global_weight;

    PeepholeFusionPass pass;
    pass.run(hir);

    // For positive-sign Pauli, the canonical S phase is carried by the
    // tableau absorption -- global_weight is not modified.
    REQUIRE(std::abs(hir.global_weight - initial_weight) < 1e-12);
}

TEST_CASE("Peephole: T_dag plus T_dag fusion leaves global_weight unchanged", "[optimizer]") {
    auto hir = hir_from("T_DAG 0\nT_DAG 0");
    auto initial_weight = hir.global_weight;

    PeepholeFusionPass pass;
    pass.run(hir);

    // Same as above: positive-sign Pauli, phase carried by tableau.
    REQUIRE(std::abs(hir.global_weight - initial_weight) < 1e-12);
}

TEST_CASE("Peephole: PHASE_ROTATION demotes to absorbed S and T gates", "[optimizer]") {
    // 0.5 half-turns = S gate -> absorbed (no ops remain)
    auto hir_s = hir_from("R_Z(0.5) 0");
    PeepholeFusionPass pass_s;
    pass_s.run(hir_s);
    REQUIRE(hir_s.ops.empty());
    REQUIRE(pass_s.fusions() == 1);

    // 1.5 half-turns = S_dag gate -> absorbed (no ops remain)
    auto hir_sdag = hir_from("R_Z(1.5) 0");
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
    REQUIRE(hir.ops[0].destab_mask() == y_destab);
    REQUIRE(hir.ops[0].stab_mask() == y_stab);
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
    CHECK(ch.destab_mask.bit_get(0));  // X bit set
    CHECK(ch.stab_mask.bit_get(0));    // Z bit set -> Y

    // Check MEASURE conjugation: X(0) -> Y(0) with sign
    CHECK(hir.ops[1].destab_mask().bit_get(0));
    CHECK(hir.ops[1].stab_mask().bit_get(0));
    CHECK(hir.ops[1].sign() == true);  // -Y

    // Check CONDITIONAL_PAULI conjugation: X(0) -> Y(0) with sign
    CHECK(hir.ops[2].destab_mask().bit_get(0));
    CHECK(hir.ops[2].stab_mask().bit_get(0));
    CHECK(hir.ops[2].sign() == true);  // -Y
}

// =============================================================================
// Negative-sign T fusion: Bug regression tests
//
// When the front-end encounters T after X (which conjugates Z -> -Z), the
// T gate has sign=true. Fusing two such negative-sign T gates must preserve
// the global phase exp(i*pi/4) per negative T gate. These tests catch
// the global phase loss that occurs when the optimizer ignores sign during
// T+T fusion.
// =============================================================================

TEST_CASE("Peephole: negative-sign T plus T preserves global phase", "[optimizer]") {
    // X conjugates Z -> -Z, so both T gates see -Z axis (sign=true).
    // T(-Z) = exp(i*pi/4) * T_dag(+Z), so T(-Z)+T(-Z) = exp(i*pi/2) * S_dag(+Z) = i * S_dag.
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    auto initial_weight = hir.global_weight;

    PeepholeFusionPass pass;
    pass.run(hir);

    // Both T gates absorbed (S absorbed into tableau)
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);

    // The normalization of two negative-sign T gates produces a net phase.
    // T(-Z) -> exp(i*pi/4) * T_dag(+Z), so two of them: exp(i*pi/2) = i.
    // Verify global_weight changed by exactly i relative to initial.
    auto ratio = hir.global_weight / initial_weight;
    CHECK_THAT(ratio.real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(ratio.imag(), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("Peephole: negative-sign T_dag plus T_dag preserves global phase", "[optimizer]") {
    // T_dag(-Z) = exp(-i*pi/4) * T(+Z), two of them: exp(-i*pi/2) * S(+Z) = -i * S.
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true, /*dagger=*/true));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true, /*dagger=*/true));
    auto initial_weight = hir.global_weight;

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.fusions() == 1);

    // exp(-i*pi/2) = -i
    auto ratio = hir.global_weight / initial_weight;
    CHECK_THAT(ratio.real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(ratio.imag(), Catch::Matchers::WithinAbs(-1.0, 1e-12));
}

TEST_CASE("Peephole: mixed-sign T cancellation preserves global phase", "[optimizer]") {
    // T(+Z) + T(-Z): effective_angles sum to 0 (cancellation), but
    // T(-Z) = exp(i*pi/4) * T_dag(+Z), so the physical result is
    // T(+Z) * exp(i*pi/4) * T_dag(+Z) = exp(i*pi/4) * I.
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/false));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    auto initial_weight = hir.global_weight;

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
    REQUIRE(pass.cancellations() == 1);

    // exp(i*pi/4)
    auto ratio = hir.global_weight / initial_weight;
    CHECK_THAT(ratio.real(), Catch::Matchers::WithinAbs(kExpIPiOver4.real(), 1e-12));
    CHECK_THAT(ratio.imag(), Catch::Matchers::WithinAbs(kExpIPiOver4.imag(), 1e-12));
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
#include "ucc/optimizer/pass_factory.h"
#include "ucc/optimizer/pass_registry.h"

TEST_CASE("Pass registry: all entries resolve via factory") {
    for (size_t i = 0; i < ucc::kNumRegisteredPasses; ++i) {
        const auto& info = ucc::kRegisteredPasses[i];
        if (info.kind == ucc::PassKind::HIR) {
            auto pass = ucc::make_hir_pass(info.name);
            REQUIRE(pass != nullptr);
        } else {
            auto pass = ucc::make_bytecode_pass(info.name);
            REQUIRE(pass != nullptr);
        }
    }
}

TEST_CASE("Pass registry: default managers use registry") {
    auto hpm = ucc::default_hir_pass_manager();
    auto bpm = ucc::default_bytecode_pass_manager();

    // Smoke test: run on a trivial circuit
    auto circuit = ucc::parse("H 0\nCNOT 0 1\nM 0\nM 1");
    auto hir = ucc::trace(circuit);
    hpm.run(hir);
    auto prog = ucc::lower(hir);
    bpm.run(prog);
    REQUIRE(prog.num_qubits == 2);
}

TEST_CASE("Pass registry: JSON round-trip is valid") {
    std::string json = ucc::pass_registry_json();
    REQUIRE(json.front() == '[');
    REQUIRE(json.back() == ']');
    REQUIRE(json.find("PeepholeFusionPass") != std::string::npos);
    REQUIRE(json.find("SingleAxisFusionPass") != std::string::npos);
    REQUIRE(json.find("RemoveNoisePass") != std::string::npos);
}
