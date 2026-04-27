// Tests for the SingleAxisFusionPass and OP_ARRAY_U2 execution.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/expand_t_pass.h"
#include "clifft/optimizer/single_axis_fusion_pass.h"
#include "clifft/svm/svm.h"
#include "clifft/util/constants.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <vector>

using namespace clifft;
using Catch::Matchers::WithinAbs;
using clifft::test::check_complex;

// =============================================================================
// Helpers
// =============================================================================

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

// Execute raw bytecode on a state with active_k already set to peak_rank
// (simulates mid-execution where qubits are already active).
SchrodingerState run_raw(CompiledModule& prog, uint8_t init_px = 0, uint8_t init_pz = 0,
                         uint64_t seed = 42) {
    SchrodingerState state(
        {.peak_rank = prog.peak_rank, .num_measurements = prog.num_measurements, .seed = seed});
    state.active_k = prog.peak_rank;
    state.p_x.bit_set(0, init_px != 0);
    state.p_z.bit_set(0, init_pz != 0);
    execute(prog, state);
    return state;
}

}  // namespace

// =============================================================================
// 4-State Oracle: verify fused == unfused for all Pauli frame inputs
// =============================================================================

TEST_CASE("U2 fusion: 4-state oracle for H-T-S sequence") {
    // Build a 1-qubit program: ARRAY_H(0), ARRAY_T(0), ARRAY_S(0)
    auto unfused = make_program({make_array_h(0), make_array_t(0), make_array_s(0)}, 1);

    // Apply fusion pass
    auto fused = make_program({make_array_h(0), make_array_t(0), make_array_s(0)}, 1);
    SingleAxisFusionPass().run(fused);

    REQUIRE(fused.bytecode.size() == 1);
    REQUIRE(fused.bytecode[0].opcode == Opcode::OP_ARRAY_U2);
    REQUIRE(fused.bytecode[0].axis_1 == 0);

    // Test all 4 incoming Pauli frame states
    for (uint8_t in_state = 0; in_state < 4; ++in_state) {
        uint8_t px = in_state & 1;
        uint8_t pz = (in_state >> 1) & 1;

        CAPTURE(in_state, px, pz);

        auto ref = run_raw(unfused, px, pz);
        auto opt = run_raw(fused, px, pz);

        // Compare amplitudes
        check_complex(opt.v()[0], ref.v()[0], kTol);
        check_complex(opt.v()[1], ref.v()[1], kTol);

        // Compare gamma
        check_complex(opt.gamma(), ref.gamma(), kTol);

        // Compare frame state
        CHECK(opt.p_x.bit_get(0) == ref.p_x.bit_get(0));
        CHECK(opt.p_z.bit_get(0) == ref.p_z.bit_get(0));
    }
}

TEST_CASE("U2 fusion: 4-state oracle for Rz-H-Rz sequence") {
    double angle1 = 0.3;
    double re1 = std::cos(angle1), im1 = std::sin(angle1);
    double angle2 = -0.7;
    double re2 = std::cos(angle2), im2 = std::sin(angle2);

    auto unfused = make_program(
        {make_array_rot(0, re1, im1), make_array_h(0), make_array_rot(0, re2, im2)}, 1);

    auto fused = make_program(
        {make_array_rot(0, re1, im1), make_array_h(0), make_array_rot(0, re2, im2)}, 1);
    SingleAxisFusionPass().run(fused);

    REQUIRE(fused.bytecode.size() == 1);
    REQUIRE(fused.bytecode[0].opcode == Opcode::OP_ARRAY_U2);

    for (uint8_t in_state = 0; in_state < 4; ++in_state) {
        uint8_t px = in_state & 1;
        uint8_t pz = (in_state >> 1) & 1;
        CAPTURE(in_state);

        auto ref = run_raw(unfused, px, pz);
        auto opt = run_raw(fused, px, pz);

        check_complex(opt.v()[0], ref.v()[0], kTol);
        check_complex(opt.v()[1], ref.v()[1], kTol);
        check_complex(opt.gamma(), ref.gamma(), kTol);
        CHECK(opt.p_x.bit_get(0) == ref.p_x.bit_get(0));
        CHECK(opt.p_z.bit_get(0) == ref.p_z.bit_get(0));
    }
}

TEST_CASE("U2 fusion: 4-state oracle for S-H-Rz-H-Rz sequence") {
    double re1 = std::cos(1.2), im1 = std::sin(1.2);
    double re2 = std::cos(-0.5), im2 = std::sin(-0.5);

    auto unfused = make_program({make_array_s(0), make_array_h(0), make_array_rot(0, re1, im1),
                                 make_array_h(0), make_array_rot(0, re2, im2)},
                                1);

    auto fused = make_program({make_array_s(0), make_array_h(0), make_array_rot(0, re1, im1),
                               make_array_h(0), make_array_rot(0, re2, im2)},
                              1);
    SingleAxisFusionPass().run(fused);

    REQUIRE(fused.bytecode.size() == 1);
    REQUIRE(fused.bytecode[0].opcode == Opcode::OP_ARRAY_U2);

    for (uint8_t in_state = 0; in_state < 4; ++in_state) {
        uint8_t px = in_state & 1;
        uint8_t pz = (in_state >> 1) & 1;
        CAPTURE(in_state);

        auto ref = run_raw(unfused, px, pz);
        auto opt = run_raw(fused, px, pz);

        check_complex(opt.v()[0], ref.v()[0], kTol);
        check_complex(opt.v()[1], ref.v()[1], kTol);
        check_complex(opt.gamma(), ref.gamma(), kTol);
        CHECK(opt.p_x.bit_get(0) == ref.p_x.bit_get(0));
        CHECK(opt.p_z.bit_get(0) == ref.p_z.bit_get(0));
    }
}

// =============================================================================
// Pass Boundary Tests: heuristics and interleaving
// =============================================================================

TEST_CASE("U2 fusion: does not fuse single array op") {
    auto mod = make_program({make_array_h(0)}, 1);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 1);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_H);
}

TEST_CASE("U2 fusion: does not fuse isolated ARRAY_ROT") {
    auto mod = make_program({make_array_rot(0, 1.0, 0.0)}, 1);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 1);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_ROT);
}

TEST_CASE("U2 fusion: terminates at two-qubit gate boundary") {
    // H(0), T(0) -> NOT fusible (only 2 array ops, no ROT)
    // CNOT(0,1) -> boundary
    // H(1), T(1), S(1) -> fusible (3 array ops)
    auto mod = make_program({make_array_h(0), make_array_t(0), make_array_cnot(0, 1),
                             make_array_h(1), make_array_t(1), make_array_s(1)},
                            2);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 4);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_H);  // H(0) unfused
    CHECK(mod.bytecode[1].opcode == Opcode::OP_ARRAY_T);  // T(0) unfused
    CHECK(mod.bytecode[2].opcode == Opcode::OP_ARRAY_CNOT);
    CHECK(mod.bytecode[3].opcode == Opcode::OP_ARRAY_U2);  // H(1)+T(1)+S(1) fused
    CHECK(mod.bytecode[3].axis_1 == 1);
}

TEST_CASE("U2 fusion: terminates at EXPAND boundary") {
    auto mod = make_program({make_array_h(0), make_array_rot(0, 0.5, 0.866), make_expand(1),
                             make_array_h(1), make_array_rot(1, 0.707, 0.707)},
                            2);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 3);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_U2);  // H(0)+Rz(0) fused
    CHECK(mod.bytecode[1].opcode == Opcode::OP_EXPAND);
    CHECK(mod.bytecode[2].opcode == Opcode::OP_ARRAY_U2);  // H(1)+Rz(1) fused
}

TEST_CASE("U2 fusion: different axes break runs") {
    // H(0), Rz(1) -> axis changes, each is length 1, no fusion
    auto mod = make_program({make_array_h(0), make_array_rot(1, 0.5, 0.866)}, 2);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 2);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_H);
    CHECK(mod.bytecode[1].opcode == Opcode::OP_ARRAY_ROT);
}

TEST_CASE("U2 fusion: frame-only ops contribute to run but need array partner") {
    // FRAME_S(0) alone -> 0 array ops, no fusion
    auto mod = make_program({make_frame_s(0)}, 1);
    SingleAxisFusionPass().run(mod);
    REQUIRE(mod.bytecode.size() == 1);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_FRAME_S);

    // FRAME_S(0) + ARRAY_H(0) -> 1 array op + 1 frame op = fuse (array_count >= 2? No, only 1).
    // Actually: frame_s has 0 array ops, array_h has 1. Total array = 1. Not fused.
    auto mod2 = make_program({make_frame_s(0), make_array_h(0)}, 1);
    SingleAxisFusionPass().run(mod2);
    REQUIRE(mod2.bytecode.size() == 2);
    CHECK(mod2.bytecode[0].opcode == Opcode::OP_FRAME_S);
    CHECK(mod2.bytecode[1].opcode == Opcode::OP_ARRAY_H);

    // FRAME_S(0) + ARRAY_H(0) + ARRAY_ROT(0) -> 2 array ops -> fuse!
    double re = std::cos(0.5), im = std::sin(0.5);
    auto mod3 = make_program({make_frame_s(0), make_array_h(0), make_array_rot(0, re, im)}, 1);
    SingleAxisFusionPass().run(mod3);
    REQUIRE(mod3.bytecode.size() == 1);
    CHECK(mod3.bytecode[0].opcode == Opcode::OP_ARRAY_U2);
}

TEST_CASE("U2 fusion: measurement terminates run") {
    auto meas = make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, 0, 0, false);
    auto mod = make_program({make_array_h(0), make_array_rot(0, 0.5, 0.866), meas}, 1, 1);
    SingleAxisFusionPass().run(mod);

    REQUIRE(mod.bytecode.size() == 2);
    CHECK(mod.bytecode[0].opcode == Opcode::OP_ARRAY_U2);
    CHECK(mod.bytecode[1].opcode == Opcode::OP_MEAS_ACTIVE_DIAGONAL);
}

// =============================================================================
// Multi-qubit integration: fuse around entangling gates
// =============================================================================

TEST_CASE("U2 fusion: Rz-H-Rz blocks around CNOT on 2 qubits") {
    // Mimics a QV layer: Rz-H-Rz on q0, CNOT(0,1), Rz-H-Rz on q1
    double r1 = std::cos(0.3), i1 = std::sin(0.3);
    double r2 = std::cos(-0.7), i2 = std::sin(-0.7);
    double r3 = std::cos(1.1), i3 = std::sin(1.1);
    double r4 = std::cos(-0.2), i4 = std::sin(-0.2);

    auto unfused =
        make_program({make_array_rot(0, r1, i1), make_array_h(0), make_array_rot(0, r2, i2),
                      make_array_cnot(0, 1), make_array_rot(1, r3, i3), make_array_h(1),
                      make_array_rot(1, r4, i4)},
                     2);

    auto fused =
        make_program({make_array_rot(0, r1, i1), make_array_h(0), make_array_rot(0, r2, i2),
                      make_array_cnot(0, 1), make_array_rot(1, r3, i3), make_array_h(1),
                      make_array_rot(1, r4, i4)},
                     2);
    SingleAxisFusionPass().run(fused);

    REQUIRE(fused.bytecode.size() == 3);
    CHECK(fused.bytecode[0].opcode == Opcode::OP_ARRAY_U2);
    CHECK(fused.bytecode[1].opcode == Opcode::OP_ARRAY_CNOT);
    CHECK(fused.bytecode[2].opcode == Opcode::OP_ARRAY_U2);

    // Verify mathematical equivalence (qubits already active)
    SchrodingerState ref_state({.peak_rank = 2, .num_measurements = 0, .seed = 42});
    ref_state.active_k = 2;
    execute(unfused, ref_state);

    SchrodingerState opt_state({.peak_rank = 2, .num_measurements = 0, .seed = 42});
    opt_state.active_k = 2;
    execute(fused, opt_state);

    for (uint64_t j = 0; j < 4; ++j) {
        CAPTURE(j);
        check_complex(opt_state.v()[j], ref_state.v()[j], kTol);
    }
    check_complex(opt_state.gamma(), ref_state.gamma(), kTol);
}

// =============================================================================
// End-to-end: compile from stim text with and without fusion
// =============================================================================

TEST_CASE("U2 fusion: end-to-end statevector equivalence for Clifford+T circuit") {
    // Small circuit with known interference patterns
    std::string circuit_text =
        "H 0\n"
        "T 0\n"
        "H 0\n"
        "S 0\n"
        "H 0\n"
        "T 1\n"
        "H 1\n"
        "CX 0 1\n"
        "T 0\n"
        "H 0\n"
        "T 1\n"
        "H 1\n";

    Circuit c = parse(circuit_text);
    HirModule hir = trace(c);
    CompiledModule unfused = lower(hir);

    Circuit c2 = parse(circuit_text);
    HirModule hir2 = trace(c2);
    CompiledModule fused = lower(hir2);
    SingleAxisFusionPass().run(fused);

    // Verify bytecode got shorter
    REQUIRE(fused.bytecode.size() < unfused.bytecode.size());

    // Verify statevector equivalence
    SchrodingerState ref_state(
        {.peak_rank = unfused.peak_rank, .num_measurements = unfused.num_measurements, .seed = 42});
    execute(unfused, ref_state);
    auto ref_sv = get_statevector(unfused, ref_state);

    SchrodingerState opt_state(
        {.peak_rank = fused.peak_rank, .num_measurements = fused.num_measurements, .seed = 42});
    execute(fused, opt_state);
    auto opt_sv = get_statevector(fused, opt_state);

    REQUIRE(ref_sv.size() == opt_sv.size());
    for (size_t j = 0; j < ref_sv.size(); ++j) {
        CAPTURE(j);
        check_complex(opt_sv[j], ref_sv[j], 1e-10);
    }
}

TEST_CASE("U2 fusion: randomized Clifford+T fuzzer") {
    // Use the test LCG to generate random circuits
    constexpr int kNumQubits = 4;
    constexpr int kDepth = 40;

    for (uint64_t trial_seed = 100; trial_seed < 108; ++trial_seed) {
        CAPTURE(trial_seed);

        uint64_t lcg = trial_seed;
        std::string circuit_text;
        const char* gates_1q[] = {"H", "S", "S_DAG", "T", "T_DAG"};
        const char* gates_2q[] = {"CX", "CY", "CZ"};

        for (int d = 0; d < kDepth; ++d) {
            uint64_t r = clifft::test::test_lcg(lcg);
            if (r % 3 == 0 && kNumQubits > 1) {
                uint64_t r2 = clifft::test::test_lcg(lcg);
                int q1 = static_cast<int>(r2 % kNumQubits);
                uint64_t r3 = clifft::test::test_lcg(lcg);
                int q2 = static_cast<int>(r3 % (kNumQubits - 1));
                if (q2 >= q1)
                    ++q2;
                circuit_text += std::string(gates_2q[r2 / 3 % 3]) + " " + std::to_string(q1) + " " +
                                std::to_string(q2) + "\n";
            } else {
                uint64_t r2 = clifft::test::test_lcg(lcg);
                int q = static_cast<int>(r2 % kNumQubits);
                circuit_text += std::string(gates_1q[r2 / 5 % 5]) + " " + std::to_string(q) + "\n";
            }
        }

        Circuit c = parse(circuit_text);
        HirModule hir = trace(c);
        CompiledModule unfused = lower(hir);

        Circuit c2 = parse(circuit_text);
        HirModule hir2 = trace(c2);
        CompiledModule fused = lower(hir2);
        SingleAxisFusionPass().run(fused);

        SchrodingerState ref_state({.peak_rank = unfused.peak_rank,
                                    .num_measurements = unfused.num_measurements,
                                    .seed = 42});
        execute(unfused, ref_state);
        auto ref_sv = get_statevector(unfused, ref_state);

        SchrodingerState opt_state(
            {.peak_rank = fused.peak_rank, .num_measurements = fused.num_measurements, .seed = 42});
        execute(fused, opt_state);
        auto opt_sv = get_statevector(fused, opt_state);

        REQUIRE(ref_sv.size() == opt_sv.size());
        for (size_t j = 0; j < ref_sv.size(); ++j) {
            check_complex(opt_sv[j], ref_sv[j], 1e-9);
        }
    }
}
