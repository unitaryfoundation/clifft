// Optimizer unit tests

#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/frontend/hir.h"
#include "ucc/optimizer/peephole.h"

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
// =============================================================================

TEST_CASE("Peephole: T plus T fuses to S", "[optimizer]") {
    auto hir = hir_from("T 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[0].is_dagger() == false);
    REQUIRE(hir.ops[0].destab_mask() == 0);
    REQUIRE(hir.ops[0].stab_mask() == Z(0));
    REQUIRE(pass.fusions() == 1);
    REQUIRE(pass.cancellations() == 0);
}

TEST_CASE("Peephole: T_dag plus T_dag fuses to S_dag", "[optimizer]") {
    auto hir = hir_from("T_DAG 0\nT_DAG 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[0].is_dagger() == true);
    REQUIRE(hir.ops[0].destab_mask() == 0);
    REQUIRE(hir.ops[0].stab_mask() == Z(0));
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

    // T slides past DETECTOR (classical) and MEASURE Z (commutes with Z)
    REQUIRE(hir.ops.size() == 3);  // CLIFFORD_PHASE + MEASURE + DETECTOR
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[0].is_dagger() == false);
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
    REQUIRE(hir.ops.size() == 2);  // S on q0 + T on q1
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
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

    REQUIRE(hir.ops.size() == 2);  // S + NOISE
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
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

    // T slides past OBSERVABLE and commuting MEASURE
    REQUIRE(hir.ops.size() == 3);  // S + MEASURE + OBSERVABLE
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
}

TEST_CASE("Peephole: CONDITIONAL_PAULI blocks when anti-commuting", "[optimizer]") {
    // CX rec[-1] 1 produces a conditional X on qubit 1.
    // The Ts are on qubit 0 (Z-axis), but the conditional is on qubit 1.
    // They have disjoint support so they commute.
    // To get an anti-commuting conditional, we need CX rec[-1] 0,
    // which is conditional X on qubit 0 -- X anti-commutes with Z.
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

    // Conditional Z(0) commutes with Z(0) T gates, MEASURE is on q1
    REQUIRE(hir.ops.size() == 3);  // S + MEASURE + CONDITIONAL
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
}

TEST_CASE("Peephole: READOUT_NOISE is transparent", "[optimizer]") {
    // MZ(p) produces a MEASURE + READOUT_NOISE pair. T slides past both.
    auto hir = hir_from("T 0\nMZ(0.01) 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // T slides past READOUT_NOISE and commuting Z-basis MEASURE
    REQUIRE(hir.ops.size() == 3);  // S + MEASURE + READOUT_NOISE
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
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

TEST_CASE("Peephole: multi-qubit Pauli axis fusion", "[optimizer]") {
    // CX 0 1 entangles the qubits. T on qubit 1 then acts on a ZZ Pauli axis.
    // Two such Ts should fuse to S on the same 2-qubit axis.
    auto hir = hir_from("CX 0 1\nT 1\nT 1");

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    auto zz_stab = Z(0) | Z(1);
    REQUIRE(hir.ops[0].stab_mask() == zz_stab);
}

TEST_CASE("Peephole: three T gates fuse to S plus T", "[optimizer]") {
    auto hir = hir_from("T 0\nT 0\nT 0");

    PeepholeFusionPass pass;
    pass.run(hir);

    // First two fuse to S. The pass does not fuse S+T (only T+T).
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[1].op_type() == OpType::T_GATE);
}

// =============================================================================
// Peephole Fusion Pass -- manually constructed HIR
//
// These tests construct HIR nodes directly to exercise algebraic edge cases
// (sign flags, specific mask combinations) that are difficult to produce from
// circuit strings. The front-end's Clifford frame rewinding makes it non-obvious
// which circuit would yield a particular sign or mask, so direct construction
// keeps the test intent clear.
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

TEST_CASE("Peephole: sign inversion makes same-direction fuse", "[optimizer]") {
    // T(sign=true) has eff=-1, T_dag(sign=false) has eff=-1
    // total = -2 -> fuse to S_dag
    HirModule hir;
    hir.num_qubits = 1;

    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/true));
    hir.ops.push_back(HeisenbergOp::make_tgate(0, Z(0), /*sign=*/false, /*dagger=*/true));

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir.ops[0].is_dagger() == true);  // total=-2 -> S_dag
}

TEST_CASE("Peephole: PHASE_ROTATION demotes to Clifford and T gates", "[optimizer]") {
    // 0.5 half-turns is S gate
    auto hir_s = hir_from("R_Z(0.5) 0");
    PeepholeFusionPass pass_s;
    pass_s.run(hir_s);
    REQUIRE(hir_s.ops.size() == 1);
    REQUIRE(hir_s.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir_s.ops[0].is_dagger() == false);

    // 1.5 half-turns is S_dag gate
    auto hir_sdag = hir_from("R_Z(1.5) 0");
    PeepholeFusionPass pass_sdag;
    pass_sdag.run(hir_sdag);
    REQUIRE(hir_sdag.ops.size() == 1);
    REQUIRE(hir_sdag.ops[0].op_type() == OpType::CLIFFORD_PHASE);
    REQUIRE(hir_sdag.ops[0].is_dagger() == true);

    // 0.25 half-turns is T gate
    auto hir_t = hir_from("R_Z(0.25) 0");
    PeepholeFusionPass pass_t;
    pass_t.run(hir_t);
    REQUIRE(hir_t.ops.size() == 1);
    REQUIRE(hir_t.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir_t.ops[0].is_dagger() == false);

    // 1.75 half-turns is T_dag gate
    auto hir_tdag = hir_from("R_Z(1.75) 0");
    PeepholeFusionPass pass_tdag;
    pass_tdag.run(hir_tdag);
    REQUIRE(hir_tdag.ops.size() == 1);
    REQUIRE(hir_tdag.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(hir_tdag.ops[0].is_dagger() == true);
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
