// Tests for TileAxisFusionPass and OP_ARRAY_U4 execution.
//
// Test categories (following the review guidance):
// 1. Asymmetric axis trap: CNOT(ctrl=hi, tgt=lo) vs CNOT(ctrl=lo, tgt=hi)
// 2. 16-state Pauli FSM exhaustion: all input frame combinations
// 3. SIMD cache-line boundaries: lo=2 hi=3, lo=3 hi=5 at high rank
// 4. Boundary halting: tiles correctly break at EXPAND, measurement, noise
// 5. Integration: full pipeline with get_statevector oracle

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/single_axis_fusion_pass.h"
#include "ucc/optimizer/tile_axis_fusion_pass.h"
#include "ucc/svm/svm.h"
#include "ucc/util/constants.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace ucc;
using ucc::test::check_complex;

namespace {

constexpr double kTol = 1e-12;

CompiledModule make_program(std::vector<Instruction> bytecode, uint32_t peak_rank,
                            uint32_t num_meas = 0) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = num_meas;
    mod.total_meas_slots = num_meas;
    return mod;
}

// Fill state with a known non-trivial pattern.
void fill_pattern(SchrodingerState& state) {
    uint64_t n = 1ULL << state.active_k;
    for (uint64_t i = 0; i < n; ++i) {
        double re = static_cast<double>(i + 1) / static_cast<double>(n + 1);
        double im = static_cast<double>(n - i) / static_cast<double>(2 * (n + 1));
        state.v()[i] = {re, im};
    }
}

// Execute unfused bytecode and return the resulting statevector.
std::vector<std::complex<double>> run_unfused(const std::vector<Instruction>& bc, uint32_t rank,
                                              uint64_t seed = 42) {
    auto prog = make_program(bc, rank);
    SchrodingerState state(rank, 0, 0, 0, seed);
    state.active_k = rank;
    fill_pattern(state);
    execute(prog, state);

    uint64_t n = 1ULL << rank;
    std::vector<std::complex<double>> result(n);
    for (uint64_t i = 0; i < n; ++i)
        result[i] = state.v()[i];
    return result;
}

// Execute fused bytecode (apply TileAxisFusionPass) and return the resulting statevector.
std::vector<std::complex<double>> run_fused(const std::vector<Instruction>& bc, uint32_t rank,
                                            uint64_t seed = 42) {
    auto prog = make_program(bc, rank);

    TileAxisFusionPass pass;
    pass.run(prog);

    SchrodingerState state(rank, 0, 0, 0, seed);
    state.active_k = rank;
    fill_pattern(state);
    execute(prog, state);

    uint64_t n = 1ULL << rank;
    std::vector<std::complex<double>> result(n);
    for (uint64_t i = 0; i < n; ++i)
        result[i] = state.v()[i];
    return result;
}

// Compare two statevectors element-wise.
void check_statevectors_equal(const std::vector<std::complex<double>>& a,
                              const std::vector<std::complex<double>>& b, double tol = kTol) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CAPTURE(i);
        check_complex(a[i], b[i], tol);
    }
}

// Set specific frame bits on a state and run program.
std::vector<std::complex<double>> run_with_frame(const std::vector<Instruction>& bc, uint32_t rank,
                                                 uint8_t px_lo, uint8_t pz_lo, uint8_t px_hi,
                                                 uint8_t pz_hi, uint16_t axis_lo, uint16_t axis_hi,
                                                 bool fuse, uint64_t seed = 42) {
    auto prog = make_program(bc, rank);
    if (fuse) {
        TileAxisFusionPass pass;
        pass.run(prog);
    }
    SchrodingerState state(rank, 0, 0, 0, seed);
    state.active_k = rank;
    fill_pattern(state);
    state.p_x.bit_set(axis_lo, px_lo != 0);
    state.p_z.bit_set(axis_lo, pz_lo != 0);
    state.p_x.bit_set(axis_hi, px_hi != 0);
    state.p_z.bit_set(axis_hi, pz_hi != 0);
    execute(prog, state);

    uint64_t n = 1ULL << rank;
    std::vector<std::complex<double>> result(n);
    for (uint64_t i = 0; i < n; ++i)
        result[i] = state.v()[i];
    return result;
}

// Get gamma after execution.
std::complex<double> run_and_get_gamma(const std::vector<Instruction>& bc, uint32_t rank,
                                       uint8_t px_lo, uint8_t pz_lo, uint8_t px_hi, uint8_t pz_hi,
                                       uint16_t axis_lo, uint16_t axis_hi, bool fuse,
                                       uint64_t seed = 42) {
    auto prog = make_program(bc, rank);
    if (fuse) {
        TileAxisFusionPass pass;
        pass.run(prog);
    }
    SchrodingerState state(rank, 0, 0, 0, seed);
    state.active_k = rank;
    fill_pattern(state);
    state.p_x.bit_set(axis_lo, px_lo != 0);
    state.p_z.bit_set(axis_lo, pz_lo != 0);
    state.p_x.bit_set(axis_hi, px_hi != 0);
    state.p_z.bit_set(axis_hi, pz_hi != 0);
    execute(prog, state);
    return state.gamma();
}

// Get output frame bits after execution.
std::tuple<bool, bool, bool, bool> run_and_get_frame(const std::vector<Instruction>& bc,
                                                     uint32_t rank, uint8_t px_lo, uint8_t pz_lo,
                                                     uint8_t px_hi, uint8_t pz_hi, uint16_t axis_lo,
                                                     uint16_t axis_hi, bool fuse,
                                                     uint64_t seed = 42) {
    auto prog = make_program(bc, rank);
    if (fuse) {
        TileAxisFusionPass pass;
        pass.run(prog);
    }
    SchrodingerState state(rank, 0, 0, 0, seed);
    state.active_k = rank;
    fill_pattern(state);
    state.p_x.bit_set(axis_lo, px_lo != 0);
    state.p_z.bit_set(axis_lo, pz_lo != 0);
    state.p_x.bit_set(axis_hi, px_hi != 0);
    state.p_z.bit_set(axis_hi, pz_hi != 0);
    execute(prog, state);
    return {state.p_x.bit_get(axis_lo), state.p_z.bit_get(axis_lo), state.p_x.bit_get(axis_hi),
            state.p_z.bit_get(axis_hi)};
}

}  // namespace

// =============================================================================
// Category 1: Asymmetric Axis Trap - CNOT direction matters
// =============================================================================

TEST_CASE("U4 fusion: CNOT ctrl=lo tgt=hi vs ctrl=hi tgt=lo") {
    // 3-op tile: CNOT + H_on_lo + CZ  (enough to trigger fusion)
    constexpr uint32_t rank = 4;
    uint16_t lo = 1, hi = 2;

    // CNOT(lo, hi): control on low axis, target on high axis
    auto bc_ctrl_lo =
        std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(lo), make_array_cz(lo, hi)};

    // CNOT(hi, lo): control on high axis, target on low axis
    auto bc_ctrl_hi =
        std::vector<Instruction>{make_array_cnot(hi, lo), make_array_h(lo), make_array_cz(lo, hi)};

    // Unfused reference
    auto ref_lo = run_unfused(bc_ctrl_lo, rank);
    auto ref_hi = run_unfused(bc_ctrl_hi, rank);

    // Fused
    auto fused_lo = run_fused(bc_ctrl_lo, rank);
    auto fused_hi = run_fused(bc_ctrl_hi, rank);

    // Each should match its own reference
    check_statevectors_equal(fused_lo, ref_lo);
    check_statevectors_equal(fused_hi, ref_hi);

    // The two should differ (different circuits)
    bool differ = false;
    for (size_t i = 0; i < ref_lo.size(); ++i) {
        if (std::abs(ref_lo[i] - ref_hi[i]) > kTol) {
            differ = true;
            break;
        }
    }
    CHECK(differ);
}

TEST_CASE("U4 fusion: CNOT axis ordering with axes 2 and 5") {
    constexpr uint32_t rank = 6;
    uint16_t lo = 2, hi = 5;

    auto bc_a = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(hi),
                                         make_array_cnot(hi, lo)};
    auto bc_b = std::vector<Instruction>{make_array_cnot(hi, lo), make_array_h(hi),
                                         make_array_cnot(lo, hi)};

    check_statevectors_equal(run_fused(bc_a, rank), run_unfused(bc_a, rank));
    check_statevectors_equal(run_fused(bc_b, rank), run_unfused(bc_b, rank));
}

// =============================================================================
// Category 2: 16-State Pauli FSM Exhaustion
// =============================================================================

TEST_CASE("U4 fusion: all 16 Pauli frame input states produce correct results") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 0, hi = 1;

    // A non-trivial tile: CNOT + H_lo + S_hi + CZ + T_lo
    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(lo), make_array_s(hi),
                                       make_array_cz(lo, hi), make_phase_t(lo)};

    for (uint8_t in_state = 0; in_state < 16; ++in_state) {
        CAPTURE(in_state);
        uint8_t px_lo = in_state & 1;
        uint8_t pz_lo = (in_state >> 1) & 1;
        uint8_t px_hi = (in_state >> 2) & 1;
        uint8_t pz_hi = (in_state >> 3) & 1;

        auto ref = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, false);
        auto fused = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, true);
        check_statevectors_equal(fused, ref);

        // Also check gamma matches
        auto gamma_ref = run_and_get_gamma(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, false);
        auto gamma_fused = run_and_get_gamma(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, true);
        check_complex(gamma_fused, gamma_ref);

        // And frame output matches
        auto frame_ref = run_and_get_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, false);
        auto frame_fused = run_and_get_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, true);
        CHECK(frame_fused == frame_ref);
    }
}

TEST_CASE("U4 fusion: all 16 frames with SWAP-based tile") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 1, hi = 3;

    // Tile with SWAP: CNOT + SWAP + H_hi + CZ
    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_swap(lo, hi),
                                       make_array_h(hi), make_array_cz(lo, hi)};

    for (uint8_t in_state = 0; in_state < 16; ++in_state) {
        CAPTURE(in_state);
        uint8_t px_lo = in_state & 1;
        uint8_t pz_lo = (in_state >> 1) & 1;
        uint8_t px_hi = (in_state >> 2) & 1;
        uint8_t pz_hi = (in_state >> 3) & 1;

        auto ref = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, false);
        auto fused = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, true);
        check_statevectors_equal(fused, ref);
    }
}

// =============================================================================
// Category 3: SIMD Cache-Line Boundaries - different axis placements
// =============================================================================

TEST_CASE("U4 fusion: axes 0 and 1 - contiguous within register") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 0, hi = 1;

    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(lo), make_array_s(hi),
                                       make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: axes 0 and 4 - mixed striding") {
    constexpr uint32_t rank = 5;
    uint16_t lo = 0, hi = 4;

    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(hi),
                                       make_array_cz(lo, hi), make_array_s(lo)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: axes 3 and 5 - fully strided 3D loops") {
    constexpr uint32_t rank = 6;
    uint16_t lo = 3, hi = 5;

    auto bc = std::vector<Instruction>{make_array_cnot(hi, lo), make_array_h(lo),
                                       make_array_cz(lo, hi), make_phase_t(hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: axes 2 and 3 at high rank 10 - AVX-512 structured stride") {
    constexpr uint32_t rank = 10;
    uint16_t lo = 2, hi = 3;

    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(lo), make_array_s(hi),
                                       make_array_cz(lo, hi), make_phase_t(lo)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: axes 3 and 7 at high rank 10") {
    constexpr uint32_t rank = 10;
    uint16_t lo = 3, hi = 7;

    auto bc = std::vector<Instruction>{make_array_cnot(hi, lo), make_array_s(lo), make_array_h(hi),
                                       make_array_cnot(lo, hi), make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

// =============================================================================
// Category 4: Boundary Halting
// =============================================================================

TEST_CASE("U4 fusion: tile breaks at EXPAND") {
    // Setup: 2 ops on {0,1}, then EXPAND, then 2 more ops on {0,1}.
    // Neither group hits the threshold of 3, so nothing should be fused.
    constexpr uint32_t rank = 4;

    auto bc = std::vector<Instruction>{
        make_array_cnot(0, 1), make_array_h(0),  // 2 ops (below threshold)
        make_expand(2),                          // breaks tile
        make_array_cnot(0, 1), make_array_h(0),  // 2 ops (below threshold)
    };

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    // No OP_ARRAY_U4 should appear
    bool has_u4 = false;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4 = true;
    }
    CHECK_FALSE(has_u4);
}

TEST_CASE("U4 fusion: tile breaks at third-axis op") {
    constexpr uint32_t rank = 4;

    // 3-op tile on {0,1}, interrupted by H on axis 2
    auto bc = std::vector<Instruction>{
        make_array_cnot(0, 1),
        make_array_h(0),
        make_array_h(2),  // breaks tile
        make_array_cz(0, 1),
    };

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    // Should not fuse (2 ops then break, then 1 op)
    bool has_u4 = false;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4 = true;
    }
    CHECK_FALSE(has_u4);
}

TEST_CASE("U4 fusion: tile breaks at frame op touching one tile axis") {
    constexpr uint32_t rank = 4;

    // FRAME_CZ(0, 2) touches axis 0 (in tile) and axis 2 (outside tile)
    auto bc = std::vector<Instruction>{
        make_array_cnot(0, 1),
        make_array_h(0),
        make_frame_cz(0, 2),  // breaks tile: touches axis 0 but not {0,1}
        make_array_cz(0, 1),
    };

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    bool has_u4 = false;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4 = true;
    }
    CHECK_FALSE(has_u4);
}

TEST_CASE("U4 fusion: does not fuse runs with fewer than 3 array ops") {
    constexpr uint32_t rank = 4;

    // Only 2 array ops
    auto bc = std::vector<Instruction>{make_array_cnot(0, 1), make_array_cz(0, 1)};

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    bool has_u4 = false;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4 = true;
    }
    CHECK_FALSE(has_u4);
}

TEST_CASE("U4 fusion: frame ops on tile axes are absorbed and do not count toward threshold") {
    constexpr uint32_t rank = 4;

    // 2 array ops + 1 frame op on tile axis = still only 2 array ops, below threshold
    auto bc = std::vector<Instruction>{make_array_cnot(0, 1), make_frame_h(0), make_array_cz(0, 1)};

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    bool has_u4 = false;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4 = true;
    }
    CHECK_FALSE(has_u4);

    // But 3 array ops with interspersed frame ops SHOULD fuse
    auto bc2 = std::vector<Instruction>{make_array_cnot(0, 1), make_frame_h(0), make_array_h(1),
                                        make_array_cz(0, 1)};

    auto prog2 = make_program(bc2, rank);
    TileAxisFusionPass pass2;
    pass2.run(prog2);

    bool has_u4_2 = false;
    for (const auto& inst : prog2.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            has_u4_2 = true;
    }
    CHECK(has_u4_2);

    // And the result should match unfused
    check_statevectors_equal(run_fused(bc2, rank), run_unfused(bc2, rank));
}

// =============================================================================
// Category 5: Integration with full pipeline
// =============================================================================

TEST_CASE("U4 fusion: QV-like tile sequence matches unfused") {
    // A realistic QV tile: CNOT + H_lo + H_hi + CZ + T_lo + T_hi +
    // CNOT + S_lo + CZ + H_hi + CNOT + CZ + S_hi + H_lo
    constexpr uint32_t rank = 6;
    uint16_t lo = 1, hi = 4;

    auto bc = std::vector<Instruction>{
        make_array_cnot(lo, hi), make_array_h(lo), make_array_h(hi),        make_array_cz(lo, hi),
        make_phase_t(lo),        make_phase_t(hi), make_array_cnot(hi, lo), make_array_s(lo),
        make_array_cz(lo, hi),   make_array_h(hi), make_array_cnot(lo, hi), make_array_cz(lo, hi),
        make_array_s(hi),        make_array_h(lo)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: tile with T-dagger and PHASE_ROT") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 0, hi = 2;

    // Continuous rotation angle
    double re = std::cos(0.3);
    double im = std::sin(0.3);

    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_phase_t_dag(hi),
                                       make_phase_rot(lo, re, im), make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: multiple disjoint tiles in one bytecode") {
    constexpr uint32_t rank = 6;

    // Tile 1 on {0, 1}, then an op on axis 3, then Tile 2 on {2, 4}
    auto bc = std::vector<Instruction>{
        // Tile 1: {0, 1}
        make_array_cnot(0, 1),
        make_array_h(0),
        make_array_cz(0, 1),
        // Break: different axis
        make_array_h(3),
        // Tile 2: {2, 4}
        make_array_cnot(2, 4),
        make_array_s(4),
        make_array_cz(2, 4),
    };

    auto prog = make_program(bc, rank);
    TileAxisFusionPass pass;
    pass.run(prog);

    // Should have exactly 2 OP_ARRAY_U4 instructions
    int u4_count = 0;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            ++u4_count;
    }
    CHECK(u4_count == 2);

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: frame ops on tile axes absorbed correctly") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 0, hi = 1;

    // CNOT + FRAME_S(lo) + H(hi) + CZ  -- frame S on tile axis absorbed
    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_frame_s(lo), make_array_h(hi),
                                       make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));

    // Verify all 16 frame states
    for (uint8_t in_state = 0; in_state < 16; ++in_state) {
        CAPTURE(in_state);
        uint8_t px_lo = in_state & 1;
        uint8_t pz_lo = (in_state >> 1) & 1;
        uint8_t px_hi = (in_state >> 2) & 1;
        uint8_t pz_hi = (in_state >> 3) & 1;

        auto ref = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, false);
        auto fused = run_with_frame(bc, rank, px_lo, pz_lo, px_hi, pz_hi, lo, hi, true);
        check_statevectors_equal(fused, ref);
    }
}

TEST_CASE("U4 fusion: full pipeline statevector oracle") {
    // Use the compile + get_statevector path to validate against the oracle.
    // A simple circuit with gates that will form a tile.
    const char* circuit = R"(
        H 0
        H 1
        CX 0 1
        H 0
        S 1
        CZ 0 1
        T 0
        T 1
        CX 1 0
        H 1
    )";

    auto circuit_obj = parse(circuit);
    auto hir = trace(circuit_obj);

    // Compile without fusion
    auto prog_nofuse = lower(hir);
    SchrodingerState state_nofuse(prog_nofuse.peak_rank, prog_nofuse.total_meas_slots, 0, 0, 42);
    execute(prog_nofuse, state_nofuse);
    auto sv_ref = get_statevector(prog_nofuse, state_nofuse);

    // Compile with TileAxisFusionPass
    auto prog_fuse = lower(hir);
    TileAxisFusionPass tile_pass;
    tile_pass.run(prog_fuse);
    SchrodingerState state_fuse(prog_fuse.peak_rank, prog_fuse.total_meas_slots, 0, 0, 42);
    execute(prog_fuse, state_fuse);
    auto sv_fuse = get_statevector(prog_fuse, state_fuse);

    check_statevectors_equal(sv_fuse, sv_ref, 1e-10);
}

TEST_CASE("U4 fusion: S-dagger on tile axis") {
    constexpr uint32_t rank = 4;
    uint16_t lo = 0, hi = 2;

    auto bc = std::vector<Instruction>{make_array_cnot(lo, hi), make_array_s_dag(lo),
                                       make_array_h(hi), make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}

TEST_CASE("U4 fusion: pass ordering - U4 before U2 leaves isolated 1Q chains for U2") {
    constexpr uint32_t rank = 5;

    // A tile on {0,1} followed by isolated 1Q ops on axis 3
    auto bc = std::vector<Instruction>{
        make_array_cnot(0, 1), make_array_h(0), make_array_cz(0, 1),
        make_array_h(3),       make_array_s(3), make_array_s(3),
    };

    auto prog = make_program(bc, rank);

    // Run U4 first, then U2
    TileAxisFusionPass tile_pass;
    tile_pass.run(prog);
    SingleAxisFusionPass u2_pass;
    u2_pass.run(prog);

    // Should have 1 OP_ARRAY_U4 and 1 OP_ARRAY_U2
    int u4_count = 0, u2_count = 0;
    for (const auto& inst : prog.bytecode) {
        if (inst.opcode == Opcode::OP_ARRAY_U4)
            ++u4_count;
        if (inst.opcode == Opcode::OP_ARRAY_U2)
            ++u2_count;
    }
    CHECK(u4_count == 1);
    CHECK(u2_count == 1);
}

TEST_CASE("U4 fusion: long tile run at rank 10") {
    // A long tile run to stress the AVX-512 structured stride path
    constexpr uint32_t rank = 10;
    uint16_t lo = 2, hi = 5;

    auto bc =
        std::vector<Instruction>{make_array_cnot(lo, hi), make_array_h(lo),        make_array_s(hi),
                                 make_array_cz(lo, hi),   make_phase_t(lo),        make_phase_t(hi),
                                 make_array_cnot(hi, lo), make_array_h(hi),        make_array_s(lo),
                                 make_array_cz(lo, hi),   make_array_cnot(lo, hi), make_array_h(lo),
                                 make_array_cz(lo, hi)};

    check_statevectors_equal(run_fused(bc, rank), run_unfused(bc, rank));
}
