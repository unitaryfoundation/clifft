// Bytecode optimization pass unit tests

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/expand_t_pass.h"
#include "ucc/optimizer/multi_gate_pass.h"
#include "ucc/optimizer/noise_block_pass.h"
#include "ucc/optimizer/swap_meas_pass.h"
#include "ucc/svm/svm.h"

#include <catch2/catch_test_macros.hpp>

using namespace ucc;

// Helper: build a minimal CompiledModule with just bytecode.
static CompiledModule make_module(std::vector<Instruction> bc) {
    CompiledModule m;
    m.bytecode = std::move(bc);
    return m;
}

// Helper: set source_map from a list-of-lists + active_k values.
static void set_source_map(CompiledModule& m, const std::vector<std::vector<uint32_t>>& lists,
                           const std::vector<uint32_t>& k_hist = {}) {
    m.source_map = SourceMap();
    for (size_t i = 0; i < lists.size(); ++i) {
        uint32_t k = (i < k_hist.size()) ? k_hist[i] : 0;
        m.source_map.append(lists[i], k);
    }
}

// Helper: read source_map back as list-of-lists for assertions.
static std::vector<std::vector<uint32_t>> get_source_map(const CompiledModule& m) {
    std::vector<std::vector<uint32_t>> result;
    for (size_t i = 0; i < m.source_map.size(); ++i) {
        auto lines = m.source_map.lines_for(i);
        result.emplace_back(lines.begin(), lines.end());
    }
    return result;
}

// =============================================================================
// NoiseBlockPass
// =============================================================================

TEST_CASE("NoiseBlockPass: single noise instruction kept as-is", "[bytecode-pass]") {
    auto m = make_module({make_noise(0)});
    NoiseBlockPass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_NOISE);
}

TEST_CASE("NoiseBlockPass: contiguous noise sites coalesced", "[bytecode-pass]") {
    auto m = make_module({
        make_noise(0),
        make_noise(1),
        make_noise(2),
        make_noise(3),
        make_noise(4),
    });
    NoiseBlockPass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_NOISE_BLOCK);
    CHECK(m.bytecode[0].pauli.cp_mask_idx == 0);    // start_site
    CHECK(m.bytecode[0].pauli.condition_idx == 5);  // count
}

TEST_CASE("NoiseBlockPass: non-contiguous sites form separate blocks", "[bytecode-pass]") {
    auto m = make_module({
        make_noise(0),
        make_noise(1),
        make_noise(2),
        make_frame_h(0),  // interrupts the noise block
        make_noise(3),
        make_noise(4),
    });
    NoiseBlockPass().run(m);
    REQUIRE(m.bytecode.size() == 3);
    CHECK(m.bytecode[0].opcode == Opcode::OP_NOISE_BLOCK);
    CHECK(m.bytecode[0].pauli.cp_mask_idx == 0);
    CHECK(m.bytecode[0].pauli.condition_idx == 3);
    CHECK(m.bytecode[1].opcode == Opcode::OP_FRAME_H);
    CHECK(m.bytecode[2].opcode == Opcode::OP_NOISE_BLOCK);
    CHECK(m.bytecode[2].pauli.cp_mask_idx == 3);
    CHECK(m.bytecode[2].pauli.condition_idx == 2);
}

TEST_CASE("NoiseBlockPass: non-consecutive site indices break block", "[bytecode-pass]") {
    auto m = make_module({
        make_noise(0), make_noise(1),
        make_noise(5),  // gap in site index
    });
    NoiseBlockPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_NOISE_BLOCK);
    CHECK(m.bytecode[0].pauli.cp_mask_idx == 0);
    CHECK(m.bytecode[0].pauli.condition_idx == 2);
    CHECK(m.bytecode[1].opcode == Opcode::OP_NOISE);
    CHECK(m.bytecode[1].pauli.cp_mask_idx == 5);
}

TEST_CASE("NoiseBlockPass: preserves source map", "[bytecode-pass]") {
    CompiledModule m;
    m.bytecode = {
        make_noise(0),
        make_noise(1),
        make_noise(2),
        make_frame_h(0),
    };
    set_source_map(m, {{10}, {11}, {12}, {20}}, {0, 0, 0, 1});

    NoiseBlockPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    auto sm = get_source_map(m);
    REQUIRE(sm.size() == 2);
    CHECK(sm[0] == std::vector<uint32_t>{10, 11, 12});
    CHECK(sm[1] == std::vector<uint32_t>{20});
    auto kh = m.source_map.active_k_history();
    REQUIRE(kh.size() == 2);
    CHECK(kh[0] == 0);
    CHECK(kh[1] == 1);
}

TEST_CASE("NoiseBlockPass: empty bytecode is a no-op", "[bytecode-pass]") {
    auto m = make_module({});
    NoiseBlockPass().run(m);
    CHECK(m.bytecode.empty());
}

// =============================================================================
// OP_NOISE_BLOCK VM execution
// =============================================================================

TEST_CASE("OP_NOISE_BLOCK: equivalent to individual OP_NOISE at low noise",
          "[bytecode-pass][svm]") {
    // Low-noise deterministic-trajectory equivalence test (same seed).
    // Use a small circuit with noise sites that have very low probability.
    auto circuit = ucc::parse(
        "H 0\n"
        "DEPOLARIZE1(0.0001) 0\n"
        "DEPOLARIZE1(0.0001) 0\n"
        "DEPOLARIZE1(0.0001) 0\n"
        "M 0\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);

    // Count OP_NOISE instructions before optimization
    size_t noise_count = 0;
    for (const auto& instr : prog_original.bytecode) {
        if (instr.opcode == Opcode::OP_NOISE)
            ++noise_count;
    }
    REQUIRE(noise_count >= 3);

    // Run noise block pass
    auto prog_optimized = prog_original;  // copy
    NoiseBlockPass().run(prog_optimized);

    // Verify it was actually coalesced
    size_t block_count = 0;
    size_t remaining_noise = 0;
    for (const auto& instr : prog_optimized.bytecode) {
        if (instr.opcode == Opcode::OP_NOISE_BLOCK)
            ++block_count;
        if (instr.opcode == Opcode::OP_NOISE)
            ++remaining_noise;
    }
    CHECK(block_count >= 1);
    CHECK(prog_optimized.bytecode.size() < prog_original.bytecode.size());

    // Both must produce identical results with deterministic seed
    uint64_t seed = 12345;
    auto res_orig = ucc::sample(prog_original, 1000, seed);
    auto res_opt = ucc::sample(prog_optimized, 1000, seed);
    REQUIRE(res_orig.measurements == res_opt.measurements);
}

TEST_CASE("OP_NOISE_BLOCK: deterministic equivalence with full noise", "[bytecode-pass][svm]") {
    // A circuit with enough noise to trigger actual errors
    auto circuit = ucc::parse(
        "H 0\n"
        "CNOT 0 1\n"
        "DEPOLARIZE1(0.01) 0 1\n"
        "DEPOLARIZE2(0.01) 0 1\n"
        "M 0 1\n"
        "DETECTOR rec[-1] rec[-2]\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);
    auto prog_optimized = prog_original;
    NoiseBlockPass().run(prog_optimized);

    uint64_t seed = 42;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_opt = ucc::sample(prog_optimized, 5000, seed);
    REQUIRE(res_orig.measurements == res_opt.measurements);
    REQUIRE(res_orig.detectors == res_opt.detectors);
}

// =============================================================================
// BytecodePassManager
// =============================================================================

TEST_CASE("BytecodePassManager: runs passes in order", "[bytecode-pass]") {
    auto m = make_module({
        make_noise(0),
        make_noise(1),
        make_noise(2),
    });

    BytecodePassManager pm;
    pm.add_pass(std::make_unique<NoiseBlockPass>());
    pm.run(m);

    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_NOISE_BLOCK);
}

// =============================================================================
// MultiGatePass
// =============================================================================

TEST_CASE("MultiGatePass: single CNOT kept as-is", "[bytecode-pass]") {
    auto m = make_module({make_array_cnot(/*ctrl=*/0, /*tgt=*/3)});
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_CNOT);
}

TEST_CASE("MultiGatePass: contiguous CNOTs sharing target fused", "[bytecode-pass]") {
    // CNOT(ctrl=0, tgt=3), CNOT(ctrl=1, tgt=3), CNOT(ctrl=2, tgt=3)
    auto m = make_module({
        make_array_cnot(0, 3),
        make_array_cnot(1, 3),
        make_array_cnot(2, 3),
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_MULTI_CNOT);
    CHECK(m.bytecode[0].axis_1 == 3);                // target
    CHECK(m.bytecode[0].multi_gate.mask == 0b0111);  // ctrl_mask = axes 0,1,2
}

TEST_CASE("MultiGatePass: different targets break CNOT run", "[bytecode-pass]") {
    auto m = make_module({
        make_array_cnot(0, 3), make_array_cnot(1, 3), make_array_cnot(2, 4),  // different target
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_MULTI_CNOT);
    CHECK(m.bytecode[0].axis_1 == 3);
    CHECK(m.bytecode[0].multi_gate.mask == 0b011);  // ctrl axes 0,1
    CHECK(m.bytecode[1].opcode == Opcode::OP_ARRAY_CNOT);
}

TEST_CASE("MultiGatePass: contiguous CZs sharing control fused", "[bytecode-pass]") {
    auto m = make_module({
        make_array_cz(5, 0),
        make_array_cz(5, 1),
        make_array_cz(5, 2),
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_MULTI_CZ);
    CHECK(m.bytecode[0].axis_1 == 5);                // control
    CHECK(m.bytecode[0].multi_gate.mask == 0b0111);  // target_mask = axes 0,1,2
}

TEST_CASE("MultiGatePass: single CZ kept as-is", "[bytecode-pass]") {
    auto m = make_module({make_array_cz(5, 0)});
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_CZ);
}

TEST_CASE("MultiGatePass: preserves source map", "[bytecode-pass]") {
    CompiledModule m;
    m.bytecode = {
        make_array_cnot(0, 3),
        make_array_cnot(1, 3),
        make_array_cnot(2, 3),
        make_array_h(0),
    };
    set_source_map(m, {{10}, {11}, {12}, {20}}, {5, 5, 5, 5});

    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    auto sm = get_source_map(m);
    REQUIRE(sm.size() == 2);
    CHECK(sm[0] == std::vector<uint32_t>{10, 11, 12});
    CHECK(sm[1] == std::vector<uint32_t>{20});
    auto kh = m.source_map.active_k_history();
    REQUIRE(kh.size() == 2);
    CHECK(kh[0] == 5);
    CHECK(kh[1] == 5);
}

TEST_CASE("MultiGatePass: mixed CNOT and CZ runs", "[bytecode-pass]") {
    auto m = make_module({
        make_array_cnot(0, 3),
        make_array_cnot(1, 3),
        make_array_cz(5, 0),
        make_array_cz(5, 1),
        make_array_cz(5, 2),
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_MULTI_CNOT);
    CHECK(m.bytecode[1].opcode == Opcode::OP_ARRAY_MULTI_CZ);
}

TEST_CASE("MultiGatePass: duplicate CNOT controls cancel via XOR", "[bytecode-pass]") {
    // Two identical CNOTs with same control and target should cancel completely.
    auto m = make_module({
        make_array_cnot(0, 3),
        make_array_cnot(0, 3),
    });
    MultiGatePass().run(m);
    CHECK(m.bytecode.empty());
}

TEST_CASE("MultiGatePass: triple CNOT leaves single gate", "[bytecode-pass]") {
    // Three identical CNOTs: XOR yields a single surviving gate.
    auto m = make_module({
        make_array_cnot(0, 3),
        make_array_cnot(0, 3),
        make_array_cnot(0, 3),
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_CNOT);
    CHECK(m.bytecode[0].axis_1 == 0);
    CHECK(m.bytecode[0].axis_2 == 3);
}

TEST_CASE("MultiGatePass: duplicate CZ targets cancel via XOR", "[bytecode-pass]") {
    auto m = make_module({
        make_array_cz(5, 0),
        make_array_cz(5, 0),
    });
    MultiGatePass().run(m);
    CHECK(m.bytecode.empty());
}

TEST_CASE("MultiGatePass: mixed duplicates partial cancel", "[bytecode-pass]") {
    // CNOT(0,3), CNOT(1,3), CNOT(0,3): ctrl 0 cancels, only ctrl 1 survives.
    auto m = make_module({
        make_array_cnot(0, 3),
        make_array_cnot(1, 3),
        make_array_cnot(0, 3),
    });
    MultiGatePass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_CNOT);
    CHECK(m.bytecode[0].axis_1 == 1);
    CHECK(m.bytecode[0].axis_2 == 3);
}

TEST_CASE("MultiGatePass: empty bytecode is a no-op", "[bytecode-pass]") {
    auto m = make_module({});
    MultiGatePass().run(m);
    CHECK(m.bytecode.empty());
}

// =============================================================================
// OP_ARRAY_MULTI_CNOT and OP_ARRAY_MULTI_CZ VM execution correctness
// =============================================================================

TEST_CASE("MULTI_CNOT: equivalent to sequential CNOTs", "[bytecode-pass][svm]") {
    // T gates activate the qubits into the statevector array.
    // The measurement MPP Z0*Z1*Z2*Z3 forces compress_pauli to handle a pure Z-string.
    // It emits CNOTs that fold all Z bits onto a single pivot, natively generating
    // a sequence of contiguous ARRAY_CNOTs sharing a target.
    auto circuit = ucc::parse(
        "H 0\nH 1\nH 2\nH 3\n"
        "T 0\nT 1\nT 2\nT 3\n"
        "MPP Z0*Z1*Z2*Z3\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);

    size_t cnot_count = 0;
    for (const auto& instr : prog_original.bytecode)
        if (instr.opcode == Opcode::OP_ARRAY_CNOT)
            ++cnot_count;

    auto prog_fused = prog_original;
    MultiGatePass().run(prog_fused);

    // Unconditionally require the backend emitted the sequence
    REQUIRE(cnot_count >= 2);

    size_t multi_count = 0;
    for (const auto& instr : prog_fused.bytecode)
        if (instr.opcode == Opcode::OP_ARRAY_MULTI_CNOT)
            ++multi_count;

    // Assert the pass actually fused them
    REQUIRE(multi_count >= 1);

    uint64_t seed = 99;
    auto res_orig = ucc::sample(prog_original, 1000, seed);
    auto res_fused = ucc::sample(prog_fused, 1000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
}

TEST_CASE("MULTI_CZ: equivalent to sequential CZs", "[bytecode-pass][svm]") {
    // T gates activate the qubits.
    // The measurement MPP X0*Z1*Z2*Z3 forces compress_pauli to handle an X-pivot
    // with Z-residues. It clears the Z-residues by natively emitting a burst
    // of contiguous ARRAY_CZs sharing a control.
    auto circuit = ucc::parse(
        "H 0\nH 1\nH 2\nH 3\n"
        "T 0\nT 1\nT 2\nT 3\n"
        "MPP X0*Z1*Z2*Z3\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);

    size_t cz_count = 0;
    for (const auto& instr : prog_original.bytecode)
        if (instr.opcode == Opcode::OP_ARRAY_CZ)
            ++cz_count;

    auto prog_fused = prog_original;
    MultiGatePass().run(prog_fused);

    // Unconditionally require the backend emitted the sequence
    REQUIRE(cz_count >= 2);

    size_t multi_count = 0;
    for (const auto& instr : prog_fused.bytecode)
        if (instr.opcode == Opcode::OP_ARRAY_MULTI_CZ)
            ++multi_count;

    // Assert the pass actually fused them
    REQUIRE(multi_count >= 1);

    uint64_t seed = 77;
    auto res_orig = ucc::sample(prog_original, 1000, seed);
    auto res_fused = ucc::sample(prog_fused, 1000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
}

TEST_CASE("MULTI_CNOT and MULTI_CZ: d5 circuit deterministic equivalence", "[bytecode-pass][svm]") {
    // Use a surface-code-like circuit with noise to stress-test.
    auto circuit = ucc::parse(
        "H 0\n"
        "CNOT 0 1\n"
        "CNOT 0 2\n"
        "CNOT 0 3\n"
        "CNOT 0 4\n"
        "DEPOLARIZE1(0.001) 0 1 2 3 4\n"
        "H 0\n"
        "CZ 0 1\n"
        "CZ 0 2\n"
        "CZ 0 3\n"
        "DEPOLARIZE2(0.001) 0 1\n"
        "M 0 1 2 3 4\n"
        "DETECTOR rec[-1] rec[-2]\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);

    auto prog_fused = prog_original;
    NoiseBlockPass().run(prog_fused);
    MultiGatePass().run(prog_fused);

    // Also run NoiseBlockPass on original for fair comparison
    NoiseBlockPass().run(prog_original);

    uint64_t seed = 12345;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_fused = ucc::sample(prog_fused, 5000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
    REQUIRE(res_orig.detectors == res_fused.detectors);
}

TEST_CASE("PassManager: NoiseBlock then MultiGate", "[bytecode-pass]") {
    auto circuit = ucc::parse(
        "H 0\n"
        "CNOT 0 1\n"
        "CNOT 0 2\n"
        "DEPOLARIZE1(0.01) 0 1 2\n"
        "M 0 1 2\n");
    auto hir = ucc::trace(circuit);
    auto prog = ucc::lower(hir);

    BytecodePassManager bpm;
    bpm.add_pass(std::make_unique<NoiseBlockPass>());
    bpm.add_pass(std::make_unique<MultiGatePass>());
    bpm.run(prog);

    // Verify both passes applied (no crash, reduced instructions)
    bool has_noise_block = false;
    bool has_multi_cnot = false;
    for (const auto& instr : prog.bytecode) {
        if (instr.opcode == Opcode::OP_NOISE_BLOCK)
            has_noise_block = true;
        if (instr.opcode == Opcode::OP_ARRAY_MULTI_CNOT)
            has_multi_cnot = true;
    }
    // At least one of these should be present (depending on circuit structure)
    CHECK((has_noise_block || has_multi_cnot));
}
// =============================================================================
// ExpandTPass
// =============================================================================

TEST_CASE("ExpandTPass: EXPAND + T fused into EXPAND_T", "[bytecode-pass]") {
    auto m = make_module({
        make_expand(3),
        make_phase_t(3),
    });
    ExpandTPass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_EXPAND_T);
    CHECK(m.bytecode[0].axis_1 == 3);
}

TEST_CASE("ExpandTPass: EXPAND + T_DAG fused into EXPAND_T_DAG", "[bytecode-pass]") {
    auto m = make_module({
        make_expand(5),
        make_phase_t_dag(5),
    });
    ExpandTPass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_EXPAND_T_DAG);
    CHECK(m.bytecode[0].axis_1 == 5);
}

TEST_CASE("ExpandTPass: mismatched axes not fused", "[bytecode-pass]") {
    auto m = make_module({
        make_expand(3),
        make_phase_t(4),  // different axis
    });
    ExpandTPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_EXPAND);
    CHECK(m.bytecode[1].opcode == Opcode::OP_PHASE_T);
}

TEST_CASE("ExpandTPass: standalone EXPAND kept", "[bytecode-pass]") {
    auto m = make_module({
        make_expand(3),
        make_frame_h(0),
    });
    ExpandTPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_EXPAND);
}

TEST_CASE("ExpandTPass: preserves source map", "[bytecode-pass]") {
    CompiledModule m;
    m.bytecode = {make_expand(3), make_phase_t(3), make_frame_h(0)};
    set_source_map(m, {{10}, {11}, {20}}, {3, 4, 4});

    ExpandTPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    auto sm = get_source_map(m);
    REQUIRE(sm.size() == 2);
    CHECK(sm[0] == std::vector<uint32_t>{10, 11});
    CHECK(sm[1] == std::vector<uint32_t>{20});
    auto kh = m.source_map.active_k_history();
    REQUIRE(kh.size() == 2);
    CHECK(kh[0] == 4);
}

TEST_CASE("ExpandTPass: empty bytecode", "[bytecode-pass]") {
    auto m = make_module({});
    ExpandTPass().run(m);
    CHECK(m.bytecode.empty());
}

// =============================================================================
// ExpandT VM execution correctness
// =============================================================================

TEST_CASE("EXPAND_T: equivalent to EXPAND then T", "[bytecode-pass][svm]") {
    auto circuit = ucc::parse(
        "H 0\n"
        "T 0\n"
        "H 0\n"
        "M 0\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);
    auto prog_fused = prog_original;
    ExpandTPass().run(prog_fused);

    // Verify fusion happened
    bool has_expand_t = false;
    for (const auto& instr : prog_fused.bytecode)
        if (instr.opcode == Opcode::OP_EXPAND_T || instr.opcode == Opcode::OP_EXPAND_T_DAG)
            has_expand_t = true;
    CHECK(has_expand_t);

    uint64_t seed = 42;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_fused = ucc::sample(prog_fused, 5000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
}

TEST_CASE("EXPAND_T: multi-T circuit deterministic equivalence", "[bytecode-pass][svm]") {
    auto circuit = ucc::parse(
        "H 0\n"
        "T 0\n"
        "T 0\n"
        "T 0\n"
        "CNOT 0 1\n"
        "T 1\n"
        "H 0\n"
        "H 1\n"
        "M 0 1\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);
    auto prog_fused = prog_original;
    ExpandTPass().run(prog_fused);

    uint64_t seed = 123;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_fused = ucc::sample(prog_fused, 5000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
}

// =============================================================================
// SwapMeasPass
// =============================================================================

TEST_CASE("SwapMeasPass: SWAP + MEAS_ACTIVE_INTERFERE fused", "[bytecode-pass]") {
    auto m = make_module({
        make_array_swap(0, 3),
        make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, 3, 42, true),
    });
    SwapMeasPass().run(m);
    REQUIRE(m.bytecode.size() == 1);
    CHECK(m.bytecode[0].opcode == Opcode::OP_SWAP_MEAS_INTERFERE);
    CHECK(m.bytecode[0].axis_1 == 0);
    CHECK(m.bytecode[0].axis_2 == 3);
    CHECK(m.bytecode[0].classical.classical_idx == 42);
    CHECK((m.bytecode[0].flags & Instruction::FLAG_SIGN) != 0);
}

TEST_CASE("SwapMeasPass: mismatched meas axis not fused", "[bytecode-pass]") {
    auto m = make_module({
        make_array_swap(0, 3),
        make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, 2, 42, false),  // axis != axis_2
    });
    SwapMeasPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    CHECK(m.bytecode[0].opcode == Opcode::OP_ARRAY_SWAP);
    CHECK(m.bytecode[1].opcode == Opcode::OP_MEAS_ACTIVE_INTERFERE);
}

TEST_CASE("SwapMeasPass: SWAP + non-interfere meas not fused", "[bytecode-pass]") {
    auto m = make_module({
        make_array_swap(0, 3),
        make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, 3, 42, false),
    });
    SwapMeasPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
}

TEST_CASE("SwapMeasPass: preserves source map", "[bytecode-pass]") {
    CompiledModule m;
    m.bytecode = {
        make_array_swap(0, 3),
        make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, 3, 42, false),
        make_frame_h(0),
    };
    set_source_map(m, {{10}, {11}, {20}}, {4, 3, 3});

    SwapMeasPass().run(m);
    REQUIRE(m.bytecode.size() == 2);
    auto sm = get_source_map(m);
    REQUIRE(sm.size() == 2);
    CHECK(sm[0] == std::vector<uint32_t>{10, 11});
    CHECK(sm[1] == std::vector<uint32_t>{20});
    auto kh = m.source_map.active_k_history();
    REQUIRE(kh.size() == 2);
    CHECK(kh[0] == 3);
}

TEST_CASE("SwapMeasPass: empty bytecode", "[bytecode-pass]") {
    auto m = make_module({});
    SwapMeasPass().run(m);
    CHECK(m.bytecode.empty());
}

// =============================================================================
// SwapMeasPass VM execution correctness
// =============================================================================

TEST_CASE("SWAP_MEAS_INTERFERE: equivalent to separate SWAP + MEAS", "[bytecode-pass][svm]") {
    // Circuit that generates SWAP + MEAS_ACTIVE_INTERFERE patterns.
    // Multiple T gates force active measurements with SWAPs.
    auto circuit = ucc::parse(
        "H 0\n"
        "CNOT 0 1\n"
        "T 0\n"
        "T 1\n"
        "H 0\n"
        "H 1\n"
        "M 0 1\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);
    auto prog_fused = prog_original;
    SwapMeasPass().run(prog_fused);

    uint64_t seed = 42;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_fused = ucc::sample(prog_fused, 5000, seed);
    REQUIRE(res_orig.measurements == res_fused.measurements);
}

TEST_CASE("All passes combined: deterministic equivalence", "[bytecode-pass][svm]") {
    auto circuit = ucc::parse(
        "H 0\n"
        "CNOT 0 1\n"
        "CNOT 0 2\n"
        "CNOT 0 3\n"
        "T 0\n"
        "T 1\n"
        "DEPOLARIZE1(0.001) 0 1 2 3\n"
        "H 0\n"
        "H 1\n"
        "CZ 0 1\n"
        "CZ 0 2\n"
        "H 0\n"
        "H 1\n"
        "H 2\n"
        "H 3\n"
        "M 0 1 2 3\n"
        "DETECTOR rec[-1] rec[-2]\n");
    auto hir = ucc::trace(circuit);
    auto prog_original = ucc::lower(hir);

    auto prog_optimized = prog_original;
    BytecodePassManager bpm;
    bpm.add_pass(std::make_unique<NoiseBlockPass>());
    bpm.add_pass(std::make_unique<MultiGatePass>());
    bpm.add_pass(std::make_unique<ExpandTPass>());
    bpm.add_pass(std::make_unique<SwapMeasPass>());
    bpm.run(prog_optimized);

    CHECK(prog_optimized.bytecode.size() < prog_original.bytecode.size());

    uint64_t seed = 12345;
    auto res_orig = ucc::sample(prog_original, 5000, seed);
    auto res_opt = ucc::sample(prog_optimized, 5000, seed);
    REQUIRE(res_orig.measurements == res_opt.measurements);
    REQUIRE(res_orig.detectors == res_opt.detectors);
}
