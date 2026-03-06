#include "ucc/backend/backend.h"
#include "ucc/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <vector>

using namespace ucc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using ucc::test::check_complex;
using ucc::test::kInvSqrt2;
using ucc::test::test_lcg;

// =============================================================================
// Helpers
// =============================================================================

namespace {

CompiledModule make_program(std::vector<Instruction> bytecode, uint32_t peak_rank,
                            uint32_t num_meas = 0, uint32_t num_det = 0, uint32_t num_obs = 0) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = num_meas;
    mod.num_detectors = num_det;
    mod.num_observables = num_obs;
    return mod;
}

Instruction make_meas_dormant_static(uint16_t v, uint32_t classical_idx) {
    return make_meas(Opcode::OP_MEAS_DORMANT_STATIC, v, classical_idx, false);
}

Instruction make_meas_dormant_random(uint16_t v, uint32_t classical_idx) {
    return make_meas(Opcode::OP_MEAS_DORMANT_RANDOM, v, classical_idx, false);
}

Instruction make_meas_active_diagonal(uint16_t v, uint32_t classical_idx) {
    return make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, v, classical_idx, false);
}

Instruction make_meas_active_interfere(uint16_t v, uint32_t classical_idx) {
    return make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE, v, classical_idx, false);
}

}  // namespace

// Semantic helpers for constructing Pauli frame bitmasks as bitwords.
// These differ from the uint64_t helpers in test_helpers.h because
// SchrodingerState::p_x / p_z use stim::bitword<kStimWidth>.
static stim::bitword<kStimWidth> X(uint16_t q) {
    return stim::bitword<kStimWidth>(uint64_t{1} << q);
}
static stim::bitword<kStimWidth> Z(uint16_t q) {
    return stim::bitword<kStimWidth>(uint64_t{1} << q);
}
static const stim::bitword<kStimWidth> NONE{uint64_t{0}};

constexpr double kTol = 1e-12;

// =============================================================================
// Frame Opcode Tests
// =============================================================================

TEST_CASE("RISC Frame: CNOT updates p_x and p_z") {
    SchrodingerState state(2, 0);

    // Set p_x[0] = 1 (X error on qubit 0)
    state.p_x = X(0);
    state.p_z = NONE;

    auto prog = make_program({make_frame_cnot(0, 1)}, 2);
    execute(prog, state);

    // CNOT(0,1): p_x[1] ^= p_x[0] -> p_x = 0b11
    // p_z[0] ^= p_z[1] -> unchanged
    CHECK(state.p_x == (X(0) | X(1)));
    CHECK(state.p_z == NONE);
}

TEST_CASE("RISC Frame: CNOT propagates Z backward") {
    SchrodingerState state(2, 0);

    state.p_x = NONE;
    state.p_z = Z(1);  // Z on qubit 1

    auto prog = make_program({make_frame_cnot(0, 1)}, 2);
    execute(prog, state);

    // CNOT(0,1): p_z[0] ^= p_z[1] -> p_z = 0b11
    CHECK(state.p_x == NONE);
    CHECK(state.p_z == (Z(0) | Z(1)));
}

TEST_CASE("RISC Frame: CZ with both X errors negates gamma") {
    SchrodingerState state(2, 0);

    state.p_x = X(0) | X(1);  // X on both
    state.p_z = NONE;

    auto prog = make_program({make_frame_cz(0, 1)}, 2);
    execute(prog, state);

    // CZ: phase = -1 since both p_x bits are set
    check_complex(state.gamma(), {-1.0, 0.0});
    // p_z[1] ^= p_x[0] = 1, p_z[0] ^= p_x[1] = 1
    CHECK(state.p_z == (Z(0) | Z(1)));
}

TEST_CASE("RISC Frame: CZ with one X error - no phase") {
    SchrodingerState state(2, 0);

    state.p_x = X(0);  // X on qubit 0 only
    state.p_z = NONE;

    auto prog = make_program({make_frame_cz(0, 1)}, 2);
    execute(prog, state);

    check_complex(state.gamma(), {1.0, 0.0});
    // p_z[1] ^= p_x[0] = 1, p_z[0] ^= p_x[1] = 0
    CHECK(state.p_z == Z(1));
}

TEST_CASE("RISC Frame: H swaps p_x and p_z") {
    SchrodingerState state(2, 0);

    state.p_x = X(0);  // X on qubit 0
    state.p_z = NONE;

    auto prog = make_program({make_frame_h(0)}, 2);
    execute(prog, state);

    // H: swap p_x[0] <-> p_z[0], no phase (only one bit set)
    CHECK(state.p_x == NONE);
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC Frame: H on Y error negates gamma") {
    SchrodingerState state(2, 0);

    // Y = iXZ, so both bits set
    state.p_x = X(0);
    state.p_z = Z(0);

    auto prog = make_program({make_frame_h(0)}, 2);
    execute(prog, state);

    // H(Y)H = -Y, so gamma negated. Bits stay the same (swap of 1,1 = 1,1)
    CHECK(state.p_x == X(0));
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {-1.0, 0.0});
}

TEST_CASE("RISC Frame: S on X error multiplies gamma by i") {
    SchrodingerState state(2, 0);

    state.p_x = X(0);
    state.p_z = NONE;

    auto prog = make_program({make_frame_s(0)}, 2);
    execute(prog, state);

    // S: p_x[0]=1 -> gamma *= i, p_z[0] ^= p_x[0] = 1
    CHECK(state.p_x == X(0));
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {0.0, 1.0});
}

TEST_CASE("RISC Frame: S on no X error - no phase change") {
    SchrodingerState state(2, 0);

    state.p_x = NONE;
    state.p_z = Z(0);

    auto prog = make_program({make_frame_s(0)}, 2);
    execute(prog, state);

    // S: p_x[0]=0 -> no phase, p_z[0] ^= 0 = unchanged
    CHECK(state.p_x == NONE);
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC Frame: S_DAG on X error multiplies gamma by -i") {
    SchrodingerState state(2, 0);

    state.p_x = X(0);
    state.p_z = NONE;

    auto prog = make_program({make_frame_s_dag(0)}, 2);
    execute(prog, state);

    // S_dag: p_x[0]=1 -> gamma *= -i, p_z[0] ^= p_x[0] = 1
    CHECK(state.p_x == X(0));
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {0.0, -1.0});
}

TEST_CASE("RISC Frame: S_DAG on no X error - no phase change") {
    SchrodingerState state(2, 0);

    state.p_x = NONE;
    state.p_z = Z(0);

    auto prog = make_program({make_frame_s_dag(0)}, 2);
    execute(prog, state);

    // S_dag: p_x[0]=0 -> no phase, p_z[0] ^= 0 = unchanged
    CHECK(state.p_x == NONE);
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC Frame: SWAP exchanges bits") {
    SchrodingerState state(3, 0);

    state.p_x = X(0) | X(2);  // X on 0,2
    state.p_z = Z(1);         // Z on 1

    auto prog = make_program({make_frame_swap(0, 2)}, 3);
    execute(prog, state);

    // Swap bits 0 and 2: p_x stays 0b101 (symmetric), p_z unchanged
    CHECK(state.p_x == (X(0) | X(2)));
    CHECK(state.p_z == Z(1));
}

TEST_CASE("RISC Frame: SWAP asymmetric case") {
    SchrodingerState state(3, 0);

    state.p_x = X(0);  // X on 0
    state.p_z = Z(2);  // Z on 2

    auto prog = make_program({make_frame_swap(0, 1)}, 3);
    execute(prog, state);

    // Swap bits 0 and 1 in both p_x and p_z
    CHECK(state.p_x == X(1));  // X moved from 0 to 1
    CHECK(state.p_z == Z(2));  // Z on 2 unchanged
}

// =============================================================================
// Array Opcode Tests
// =============================================================================

TEST_CASE("RISC Array: EXPAND doubles array") {
    SchrodingerState state(3, 0);
    // Start with active_k = 0, v[0] = 1.0

    auto prog = make_program({make_expand(0)}, 3);
    execute(prog, state);

    // After expand(0): active_k = 1, v = [1, 1], gamma = 1/sqrt(2)
    CHECK(state.active_k == 1);
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {1.0, 0.0});
    CHECK_THAT(state.gamma().real(), WithinAbs(1.0 / std::sqrt(2.0), kTol));
    CHECK_THAT(state.gamma().imag(), WithinAbs(0.0, kTol));
}

TEST_CASE("RISC Array: Two EXPANDs create 4-element array") {
    SchrodingerState state(3, 0);

    auto prog = make_program({make_expand(0), make_expand(1)}, 3);
    execute(prog, state);

    CHECK(state.active_k == 2);
    // v = [1, 1, 1, 1], gamma = 1/2
    for (int j = 0; j < 4; ++j) {
        check_complex(state.v()[j], {1.0, 0.0});
    }
    CHECK_THAT(state.gamma().real(), WithinAbs(0.5, kTol));
}

TEST_CASE("RISC Array: CNOT on 2-qubit state") {
    SchrodingerState state(2, 0);
    state.active_k = 2;
    // Set up |10> state: v = [0, 0, 1, 0]
    state.v()[0] = {0.0, 0.0};
    state.v()[1] = {0.0, 0.0};
    state.v()[2] = {1.0, 0.0};  // |10>
    state.v()[3] = {0.0, 0.0};

    auto prog = make_program({make_array_cnot(0, 1)}, 2);
    execute(prog, state);

    // CNOT(0,1) on |10>: bit 0 is control. Index 2 = 0b10, control bit 0 is 0.
    // So |10> is unchanged (control bit = 0).
    check_complex(state.v()[2], {1.0, 0.0});
}

TEST_CASE("RISC Array: CNOT flips target when control is 1") {
    SchrodingerState state(2, 0);
    state.active_k = 2;
    // |01> state: v[1] = 1 (qubit 0 = 1, qubit 1 = 0)
    state.v()[0] = {0.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.v()[2] = {0.0, 0.0};
    state.v()[3] = {0.0, 0.0};

    auto prog = make_program({make_array_cnot(0, 1)}, 2);
    execute(prog, state);

    // CNOT(0,1): control = bit 0, target = bit 1
    // |01> has bit 0 = 1, so flip bit 1: |01> -> |11>
    check_complex(state.v()[0], {0.0, 0.0});
    check_complex(state.v()[1], {0.0, 0.0});
    check_complex(state.v()[2], {0.0, 0.0});
    check_complex(state.v()[3], {1.0, 0.0});  // |11> = index 3
}

TEST_CASE("RISC Array: CZ negates 11 component") {
    SchrodingerState state(2, 0);
    state.active_k = 2;
    // Equal superposition: |00> + |01> + |10> + |11>
    for (int j = 0; j < 4; ++j) {
        state.v()[j] = {1.0, 0.0};
    }

    auto prog = make_program({make_array_cz(0, 1)}, 2);
    execute(prog, state);

    // CZ negates only |11> component
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {1.0, 0.0});
    check_complex(state.v()[2], {1.0, 0.0});
    check_complex(state.v()[3], {-1.0, 0.0});
}

TEST_CASE("RISC Array: SWAP exchanges axes") {
    SchrodingerState state(2, 0);
    state.active_k = 2;
    // |01> state
    state.v()[0] = {0.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.v()[2] = {0.0, 0.0};
    state.v()[3] = {0.0, 0.0};

    auto prog = make_program({make_array_swap(0, 1)}, 2);
    execute(prog, state);

    // SWAP(0,1) on |01> -> |10>
    check_complex(state.v()[0], {0.0, 0.0});
    check_complex(state.v()[1], {0.0, 0.0});
    check_complex(state.v()[2], {1.0, 0.0});  // |10>
    check_complex(state.v()[3], {0.0, 0.0});
}

TEST_CASE("RISC Array: S_DAG applies -i to 1-component") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};  // |0>
    state.v()[1] = {1.0, 0.0};  // |1>

    auto prog = make_program({make_array_s_dag(0)}, 2);
    execute(prog, state);

    // diag(1, -i): |0> unchanged, |1> *= -i
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {0.0, -1.0});
}

TEST_CASE("RISC Array: S then S_DAG cancels") {
    SchrodingerState state(2, 0);
    state.active_k = 2;
    state.v()[0] = {0.5, 0.0};
    state.v()[1] = {0.0, 0.5};
    state.v()[2] = {-0.5, 0.0};
    state.v()[3] = {0.0, -0.5};

    auto prog = make_program({make_array_s(0), make_array_s_dag(0)}, 2);
    execute(prog, state);

    // S then S_dag = identity on array and frame
    check_complex(state.v()[0], {0.5, 0.0});
    check_complex(state.v()[1], {0.0, 0.5});
    check_complex(state.v()[2], {-0.5, 0.0});
    check_complex(state.v()[3], {0.0, -0.5});
    check_complex(state.gamma(), {1.0, 0.0});
}

// =============================================================================
// Phase T/T_dag Tests
// =============================================================================

TEST_CASE("RISC Phase: T on active axis - no frame error") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    // |+> state: v = [1, 1], gamma = 1/sqrt(2)
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto prog = make_program({make_phase_t(0)}, 2);
    execute(prog, state);

    // T applies e^{i*pi/4} to |1> component (no frame error)
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {kInvSqrt2, kInvSqrt2});
    // Gamma unchanged
    CHECK_THAT(state.gamma().real(), WithinAbs(1.0 / std::sqrt(2.0), kTol));
    CHECK_THAT(state.gamma().imag(), WithinAbs(0.0, kTol));
}

TEST_CASE("RISC Phase: T on active axis - with X frame error") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.p_x = X(0);  // X error on qubit 0

    auto prog = make_program({make_phase_t(0)}, 2);
    execute(prog, state);

    // With p_x[0]=1: apply T_dag to array, gamma *= e^{i*pi/4}
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {kInvSqrt2, -kInvSqrt2});  // e^{-i*pi/4}
    check_complex(state.gamma(), {kInvSqrt2, kInvSqrt2});  // e^{i*pi/4}
}

TEST_CASE("RISC Phase: T_dag on active axis - no frame error") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};

    auto prog = make_program({make_phase_t_dag(0)}, 2);
    execute(prog, state);

    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {kInvSqrt2, -kInvSqrt2});  // e^{-i*pi/4}
}

TEST_CASE("RISC Phase: T then T_dag cancels") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    state.v()[0] = {0.6, 0.1};
    state.v()[1] = {0.3, -0.7};
    auto v0_orig = state.v()[0];
    auto v1_orig = state.v()[1];

    auto prog = make_program({make_phase_t(0), make_phase_t_dag(0)}, 2);
    execute(prog, state);

    check_complex(state.v()[0], v0_orig);
    check_complex(state.v()[1], v1_orig);
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC Phase: Two T gates equal S - applies i to 1-component") {
    SchrodingerState state(2, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};

    auto prog = make_program({make_phase_t(0), make_phase_t(0)}, 2);
    execute(prog, state);

    // Two T gates = S = diag(1, i)
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {0.0, 1.0});
}

// =============================================================================
// Expand + Phase: T gate on dormant qubit via H then T
// =============================================================================

TEST_CASE("RISC: EXPAND then PHASE_T - single T gate circuit") {
    SchrodingerState state(2, 0);
    // Initial state: |0>, active_k = 0, v = [1]

    auto prog = make_program({make_expand(0), make_phase_t(0)}, 2);
    execute(prog, state);

    // After EXPAND(0): v = [1, 1], gamma = 1/sqrt(2), active_k = 1
    // After PHASE_T(0): v = [1, e^{i*pi/4}]
    CHECK(state.active_k == 1);
    check_complex(state.v()[0], {1.0, 0.0});
    check_complex(state.v()[1], {kInvSqrt2, kInvSqrt2});
    CHECK_THAT(state.gamma().real(), WithinAbs(1.0 / std::sqrt(2.0), kTol));
}

// =============================================================================
// Measurement Tests
// =============================================================================

TEST_CASE("RISC Meas: Dormant static - outcome from p_x") {
    SchrodingerState state(2, 2);

    // p_x[0] = 0, p_x[1] = 1
    state.p_x = X(1);

    auto prog =
        make_program({make_meas_dormant_static(0, 0), make_meas_dormant_static(1, 1)}, 2, 2);
    execute(prog, state);

    CHECK(state.meas_record[0] == 0);  // p_x[0] = 0
    CHECK(state.meas_record[1] == 1);  // p_x[1] = 1
}

TEST_CASE("RISC Meas: Active diagonal on definite 0-state") {
    // 1-qubit active state |0>: v = [1, 0]
    SchrodingerState state(2, 1);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {0.0, 0.0};

    auto prog = make_program({make_meas_active_diagonal(0, 0)}, 2, 1);
    execute(prog, state);

    // Definite outcome: b=0 (all amplitude in lower half)
    CHECK(state.meas_record[0] == 0);
    CHECK(state.active_k == 0);
    // v[0] should be 1.0 (the kept amplitude)
    check_complex(state.v()[0], {1.0, 0.0});
    // gamma = 1/sqrt(1.0) = 1.0
    CHECK_THAT(std::abs(state.gamma()), WithinAbs(1.0, kTol));
}

TEST_CASE("RISC Meas: Active diagonal on definite 1-state") {
    SchrodingerState state(2, 1);
    state.active_k = 1;
    state.v()[0] = {0.0, 0.0};
    state.v()[1] = {1.0, 0.0};

    auto prog = make_program({make_meas_active_diagonal(0, 0)}, 2, 1);
    execute(prog, state);

    // Definite outcome: b=1, m = 1 XOR p_x[0](=0) = 1
    CHECK(state.meas_record[0] == 1);
    CHECK(state.active_k == 0);
    check_complex(state.v()[0], {1.0, 0.0});
}

TEST_CASE("RISC Meas: Active diagonal with p_x flips outcome") {
    SchrodingerState state(2, 1);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};  // All amplitude in |0>
    state.v()[1] = {0.0, 0.0};
    state.p_x = X(0);  // X error on qubit 0

    auto prog = make_program({make_meas_active_diagonal(0, 0)}, 2, 1);
    execute(prog, state);

    // b=0 deterministically, but m = b XOR p_x[0] = 0 XOR 1 = 1
    CHECK(state.meas_record[0] == 1);
    CHECK(state.active_k == 0);
}

TEST_CASE("RISC Meas: Active interfere on plus-state") {
    // |+> = (|0> + |1>)/sqrt(2)
    SchrodingerState state(2, 1);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto prog = make_program({make_meas_active_interfere(0, 0)}, 2, 1);
    execute(prog, state);

    // |+> measured in X-basis: deterministic b_x=0
    // prob_plus = |1+1|^2 = 4, prob_minus = |1-1|^2 = 0
    CHECK(state.meas_record[0] == 0);
    CHECK(state.active_k == 0);
    // Folded: v[0] = (1 + 1) * kInvSqrt2 = sqrt(2)
    // gamma *= sqrt(total / prob_bx) = sqrt(4 / 4) = 1
    // Original gamma = 1/sqrt(2), so final gamma = 1/sqrt(2)
    check_complex(state.v()[0], {std::sqrt(2.0), 0.0});
    CHECK_THAT(std::abs(state.gamma()), WithinAbs(kInvSqrt2, kTol));
}

TEST_CASE("RISC Meas: Active interfere on minus-state") {
    // |-> = (|0> - |1>)/sqrt(2)
    SchrodingerState state(2, 1);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {-1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto prog = make_program({make_meas_active_interfere(0, 0)}, 2, 1);
    execute(prog, state);

    // |-> in X-basis: deterministic b_x=1
    CHECK(state.meas_record[0] == 1);
    CHECK(state.active_k == 0);
    // Folded: v[0] = (1 - (-1)) * kInvSqrt2 = sqrt(2)
    check_complex(state.v()[0], {std::sqrt(2.0), 0.0});
}

// =============================================================================
// Dormant Random Measurement Test
// =============================================================================

TEST_CASE("RISC Meas: Dormant random - statistical test") {
    // Run many shots and check 50/50 distribution
    uint32_t zeros = 0;
    uint32_t ones = 0;
    constexpr uint32_t num_trials = 10000;

    SchrodingerState state(2, 1);
    auto prog = make_program({make_meas_dormant_random(0, 0)}, 2, 1);
    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 12345 + 42);
        execute(prog, state);

        if (state.meas_record[0] == 0) {
            zeros++;
        } else {
            ones++;
        }
    }

    // Should be roughly 50/50 (allow 5% tolerance)
    double ratio = static_cast<double>(zeros) / num_trials;
    CHECK_THAT(ratio, WithinAbs(0.5, 0.05));
}

// =============================================================================
// Integration mini-tests: End-to-end bytecode sequences
// =============================================================================

TEST_CASE("RISC Integration: Expand-T-MeasDiag gives correct statistics") {
    // H|0> then T then measure Z: should give 50/50 with T phase
    // After EXPAND: v = [1, 1], gamma = 1/sqrt(2)
    // After T: v = [1, e^{i*pi/4}], gamma = 1/sqrt(2)
    // After MEAS_ACTIVE_DIAGONAL: prob_b0 = |1|^2 = 1, prob_b1 = |e^{i*pi/4}|^2 = 1
    // So 50/50 distribution

    uint32_t zeros = 0;
    uint32_t ones = 0;
    constexpr uint32_t num_trials = 10000;

    SchrodingerState state(2, 1);
    auto prog =
        make_program({make_expand(0), make_phase_t(0), make_meas_active_diagonal(0, 0)}, 2, 1);
    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 77 + 13);
        execute(prog, state);

        if (state.meas_record[0] == 0) {
            zeros++;
        } else {
            ones++;
        }
    }

    double ratio = static_cast<double>(zeros) / num_trials;
    CHECK_THAT(ratio, WithinAbs(0.5, 0.05));
}

TEST_CASE("RISC Integration: Expand-MeasInterfere on plus gives deterministic 0") {
    // EXPAND(0) creates |+>, measuring in X-basis should deterministically yield 0
    constexpr uint32_t num_trials = 100;

    SchrodingerState state(2, 1);
    auto prog = make_program({make_expand(0), make_meas_active_interfere(0, 0)}, 2, 1);
    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial);
        execute(prog, state);

        CHECK(state.meas_record[0] == 0);
    }
}

TEST_CASE("RISC Integration: Two-qubit EXPAND-CNOT-MEAS pipeline runs correctly") {
    // EXPAND(0), EXPAND(1) produces |++> = uniform superposition [1,1,1,1]/2.
    // CNOT on |++> = |++> (invariant), so measurements are independent 50/50.

    constexpr uint32_t num_trials = 1000;

    SchrodingerState state(3, 2);
    auto prog = make_program({make_expand(0), make_expand(1), make_array_cnot(0, 1),
                              make_meas_active_diagonal(1, 0), make_meas_active_diagonal(0, 1)},
                             3, 2);
    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 31);
        execute(prog, state);

        CHECK(state.active_k == 0);
    }
}

TEST_CASE("RISC Integration: Bell state via targeted initial state") {
    // Create Bell state: (|00> + |11>)/sqrt(2)
    // Start with active_k = 2 and manually set amplitudes
    SchrodingerState state(3, 2);
    state.active_k = 2;
    state.v()[0] = {1.0, 0.0};  // |00>
    state.v()[1] = {0.0, 0.0};
    state.v()[2] = {0.0, 0.0};
    state.v()[3] = {1.0, 0.0};  // |11>
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    // Measure axis 1 (k-1), then axis 0
    // Repeated trials to check correlation
    uint32_t same = 0;
    constexpr uint32_t num_trials = 1000;

    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        INFO("Trial: " << trial);
        state.active_k = 2;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {0.0, 0.0};
        state.v()[2] = {0.0, 0.0};
        state.v()[3] = {1.0, 0.0};
        state.set_gamma({1.0 / std::sqrt(2.0), 0.0});
        state.p_x = 0;
        state.p_z = 0;
        state.meas_record[0] = 0;
        state.meas_record[1] = 0;
        // Reseed
        state.reset(trial * 97 + 7);
        state.active_k = 2;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {0.0, 0.0};
        state.v()[2] = {0.0, 0.0};
        state.v()[3] = {1.0, 0.0};
        state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

        auto prog =
            make_program({make_meas_active_diagonal(1, 0), make_meas_active_diagonal(0, 1)}, 3, 2);
        execute(prog, state);

        if (state.meas_record[0] == state.meas_record[1]) {
            same++;
        }
    }

    // Bell state: outcomes must always be correlated
    CHECK(same == num_trials);
}

TEST_CASE("RISC Meas: Dormant random resets frame correctly") {
    SchrodingerState state(2, 1, 0, 0, 42);
    state.p_x = X(0);  // X error on qubit 0
    state.p_z = Z(0);  // Z error on qubit 0

    auto prog = make_program({make_meas_dormant_random(0, 0)}, 2, 1);
    execute(prog, state);

    // After measurement, p_z[0] must be 0, p_x[0] must equal the outcome
    uint8_t m = state.meas_record[0];
    CHECK(bool((state.p_x >> 0) & uint64_t{1}) == bool(m));
    CHECK((state.p_z & uint64_t{1}) == uint64_t{0});
}

// =============================================================================
// Detector Test
// =============================================================================

TEST_CASE("RISC Detector: computes parity of measurement records") {
    SchrodingerState state(2, 3, 1);
    state.meas_record[0] = 1;
    state.meas_record[1] = 0;
    state.meas_record[2] = 1;

    CompiledModule mod;
    mod.peak_rank = 2;
    mod.num_measurements = 3;
    mod.num_detectors = 1;
    mod.constant_pool.detector_targets.push_back({0, 1, 2});  // XOR of all 3

    Instruction det{};
    det.opcode = Opcode::OP_DETECTOR;
    det.pauli.cp_mask_idx = 0;    // detector target list index
    det.pauli.condition_idx = 0;  // detector record index
    mod.bytecode.push_back(det);

    execute(mod, state);

    // Parity: 1 ^ 0 ^ 1 = 0
    CHECK(state.det_record[0] == 0);
}

TEST_CASE("RISC Detector: odd parity") {
    SchrodingerState state(2, 2, 1);
    state.meas_record[0] = 1;
    state.meas_record[1] = 0;

    CompiledModule mod;
    mod.peak_rank = 2;
    mod.num_measurements = 2;
    mod.num_detectors = 1;
    mod.constant_pool.detector_targets.push_back({0, 1});

    Instruction det{};
    det.opcode = Opcode::OP_DETECTOR;
    det.pauli.cp_mask_idx = 0;
    det.pauli.condition_idx = 0;
    mod.bytecode.push_back(det);

    execute(mod, state);

    CHECK(state.det_record[0] == 1);
}

// =============================================================================
// APPLY_PAULI Tests
// =============================================================================

TEST_CASE("RISC ApplyPauli: X error flips p_x bit") {
    SchrodingerState state(4, 1);
    state.meas_record[0] = 1;  // condition_idx=0 fires
    state.p_x = NONE;
    state.p_z = NONE;

    CompiledModule mod;
    mod.num_measurements = 1;
    mod.peak_rank = 4;

    stim::PauliString<kStimWidth> ps(4);
    ps.xs[1] = true;
    mod.constant_pool.pauli_masks.push_back(ps);

    Instruction instr{};
    instr.opcode = Opcode::OP_APPLY_PAULI;
    instr.pauli.cp_mask_idx = 0;
    instr.pauli.condition_idx = 0;
    mod.bytecode.push_back(instr);

    execute(mod, state);

    CHECK(state.p_x == X(1));
    CHECK(state.p_z == NONE);
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC ApplyPauli: Z error flips p_z bit") {
    SchrodingerState state(4, 1);
    state.meas_record[0] = 1;
    state.p_x = NONE;
    state.p_z = NONE;

    CompiledModule mod;
    mod.num_measurements = 1;
    mod.peak_rank = 4;

    stim::PauliString<kStimWidth> ps(4);
    ps.zs[2] = true;
    mod.constant_pool.pauli_masks.push_back(ps);

    Instruction instr{};
    instr.opcode = Opcode::OP_APPLY_PAULI;
    instr.pauli.cp_mask_idx = 0;
    instr.pauli.condition_idx = 0;
    mod.bytecode.push_back(instr);

    execute(mod, state);

    CHECK(state.p_x == NONE);
    CHECK(state.p_z == Z(2));
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC ApplyPauli: X error on Z frame has no anticommutation phase") {
    // E=X_0 applied to P=Z_0: commutation phase is (-1)^{e_z . p_x} = (-1)^0 = +1
    SchrodingerState state(4, 1);
    state.meas_record[0] = 1;
    state.p_x = NONE;
    state.p_z = Z(0);

    CompiledModule mod;
    mod.num_measurements = 1;
    mod.peak_rank = 4;

    stim::PauliString<kStimWidth> ps(4);
    ps.xs[0] = true;
    mod.constant_pool.pauli_masks.push_back(ps);

    Instruction instr{};
    instr.opcode = Opcode::OP_APPLY_PAULI;
    instr.pauli.cp_mask_idx = 0;
    instr.pauli.condition_idx = 0;
    mod.bytecode.push_back(instr);

    execute(mod, state);

    CHECK(state.p_x == X(0));
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {1.0, 0.0});
}

TEST_CASE("RISC ApplyPauli: Z error on X frame negates gamma") {
    // E=Z_0 applied to P=X_0: commutation phase is (-1)^{e_z . p_x} = (-1)^1 = -1
    SchrodingerState state(4, 1);
    state.meas_record[0] = 1;
    state.p_x = X(0);
    state.p_z = NONE;

    CompiledModule mod;
    mod.num_measurements = 1;
    mod.peak_rank = 4;

    stim::PauliString<kStimWidth> ps(4);
    ps.zs[0] = true;
    mod.constant_pool.pauli_masks.push_back(ps);

    Instruction instr{};
    instr.opcode = Opcode::OP_APPLY_PAULI;
    instr.pauli.cp_mask_idx = 0;
    instr.pauli.condition_idx = 0;
    mod.bytecode.push_back(instr);

    execute(mod, state);

    CHECK(state.p_x == X(0));
    CHECK(state.p_z == Z(0));
    check_complex(state.gamma(), {-1.0, 0.0});
}

TEST_CASE("RISC ApplyPauli: signed PauliString negates gamma") {
    SchrodingerState state(4, 1);
    state.meas_record[0] = 1;
    state.p_x = NONE;
    state.p_z = NONE;

    CompiledModule mod;
    mod.num_measurements = 1;
    mod.peak_rank = 4;

    stim::PauliString<kStimWidth> ps(4);
    ps.xs[0] = true;
    ps.sign = true;
    mod.constant_pool.pauli_masks.push_back(ps);

    Instruction instr{};
    instr.opcode = Opcode::OP_APPLY_PAULI;
    instr.pauli.cp_mask_idx = 0;
    instr.pauli.condition_idx = 0;
    mod.bytecode.push_back(instr);

    execute(mod, state);

    CHECK(state.p_x == X(0));
    CHECK(state.p_z == NONE);
    check_complex(state.gamma(), {-1.0, 0.0});
}

// =============================================================================
// Gamma Tracking Tests
// =============================================================================

TEST_CASE("RISC Gamma: Frame S accumulates i phase") {
    SchrodingerState state(2, 0);
    state.p_x = X(0);  // X on qubit 0

    // Apply S four times: gamma should cycle i -> -1 -> -i -> 1
    auto prog =
        make_program({make_frame_s(0), make_frame_s(0), make_frame_s(0), make_frame_s(0)}, 2);
    execute(prog, state);

    // Four applications of S with X error: gamma *= i four times = i^4 = 1
    // But S also updates p_z each time. Let's trace:
    // Start: p_x=1, p_z=0. S: gamma*=i, p_z^=1 -> p_z=1
    // p_x=1, p_z=1. S: gamma*=i (now -1), p_z^=1 -> p_z=0
    // p_x=1, p_z=0. S: gamma*=i (now -i), p_z^=1 -> p_z=1
    // p_x=1, p_z=1. S: gamma*=i (now 1), p_z^=1 -> p_z=0
    check_complex(state.gamma(), {1.0, 0.0});
    CHECK(state.p_z == NONE);
}

// =============================================================================
// Multi-Shot Reset Tests (validate no stale data after reset)
// =============================================================================

TEST_CASE("RISC Reset: meas and det records do not leak between shots") {
    // Construct a program where measurement outcome depends on the RNG seed.
    // A dormant-random measurement produces outcome 0 or 1 depending on seed.
    // A detector then reads that measurement. If stale data from a previous
    // shot leaked through reset(), the detector parity would be wrong.

    CompiledModule mod;
    mod.num_qubits = 2;
    mod.peak_rank = 2;
    mod.num_measurements = 1;
    mod.total_meas_slots = 1;
    mod.num_detectors = 1;

    // Dormant random measurement at axis 0 -> meas_record[0]
    mod.bytecode.push_back(make_meas_dormant_random(0, 0));

    // Detector that reads meas_record[0] -> det_record[0]
    Instruction det_instr{};
    det_instr.opcode = Opcode::OP_DETECTOR;
    det_instr.pauli.cp_mask_idx = 0;    // index into detector_targets
    det_instr.pauli.condition_idx = 0;  // det_record[0]
    mod.bytecode.push_back(det_instr);
    mod.constant_pool.detector_targets.push_back({0});  // parity of meas[0]

    // Run 100 shots manually, verifying each shot's detector matches its
    // measurement. If reset() left stale data, some shots would mismatch.
    SchrodingerState state(mod.peak_rank, mod.total_meas_slots, mod.num_detectors, 0, 0);

    for (uint32_t shot = 0; shot < 100; ++shot) {
        if (shot > 0) {
            state.reset(shot);
        }
        execute(mod, state);

        // Detector parity should exactly equal measurement outcome
        CHECK(state.det_record[0] == state.meas_record[0]);
    }

    // Across 100 shots with different seeds, we expect a mix of 0s and 1s.
    // If all were identical, the test would be vacuous.
    // (Dormant-random with no frame bias is 50/50, so this is near-certain.)
}

TEST_CASE("RISC Reset: deterministic measurement overwrites previous shot") {
    // Shot 1: manually set p_x[0]=1 so dormant-static measurement yields 1.
    // Shot 2: reset clears p_x, so measurement should yield 0.
    // Without deterministic overwrite this would leak the stale 1.

    CompiledModule mod;
    mod.num_qubits = 2;
    mod.peak_rank = 2;
    mod.num_measurements = 1;
    mod.total_meas_slots = 1;

    mod.bytecode.push_back(make_meas_dormant_static(0, 0));

    SchrodingerState state(mod.peak_rank, mod.total_meas_slots, 0, 0, 0);

    // Shot 1: force p_x[0]=1 -> measurement yields 1
    state.p_x = X(0);
    execute(mod, state);
    CHECK(state.meas_record[0] == 1);

    // Shot 2: reset clears p_x -> measurement should yield 0
    state.reset(1);
    execute(mod, state);
    CHECK(state.meas_record[0] == 0);
}

// =============================================================================
// Array Compaction Fuzzers
// =============================================================================

// Generate a random complex number with magnitude in [0, 1].
static std::complex<double> random_complex(uint64_t& seed) {
    double re = static_cast<double>(test_lcg(seed) >> 11) * 0x1.0p-53 * 2.0 - 1.0;
    double im = static_cast<double>(test_lcg(seed) >> 11) * 0x1.0p-53 * 2.0 - 1.0;
    return {re, im};
}

// Compute physical norm: |gamma|^2 * sum(|v[i]|^2)
static double physical_norm(const SchrodingerState& state) {
    double arr_norm = 0.0;
    uint64_t size = state.v_size();
    for (uint64_t i = 0; i < size; ++i) {
        arr_norm += std::norm(state.v()[i]);
    }
    return std::norm(state.gamma()) * arr_norm;
}

TEST_CASE("Compaction fuzz: active diagonal preserves norm - k=4") {
    // Bypass compiler: manually construct k=4 state with random amplitudes,
    // measure the top axis via OP_MEAS_ACTIVE_DIAGONAL, verify norm preserved.
    uint64_t seed = 0xDEAD0001;
    SchrodingerState state(4, 1);

    for (int trial = 0; trial < 200; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(test_lcg(seed));
        state.active_k = 4;

        // Fill 16 random complex amplitudes
        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] = random_complex(seed);
            raw_norm += std::norm(state.v()[i]);
        }
        // Normalize so the physical state has norm 1
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-10));

        // Calculate theoretical branch probabilities before measurement
        double prob_b0 = 0.0;
        double prob_b1 = 0.0;
        uint64_t half = 1ULL << 3;  // k-1 = 3
        for (uint64_t i = 0; i < half; ++i) {
            prob_b0 += std::norm(state.v()[i]);
            prob_b1 += std::norm(state.v()[i + half]);
        }
        double total_prob = prob_b0 + prob_b1;
        double gamma_mag_before = std::abs(state.gamma());

        // Save frame bit before execute (measurement overwrites p_x)
        bool px_v_before = (state.p_x.val >> 3) & 1;

        // Measure top axis (k-1 = 3) in Z-basis
        auto prog = make_program({make_meas_active_diagonal(3, 0)}, 4, 1);
        execute(prog, state);

        CHECK(state.active_k == 3);
        CHECK(state.meas_record[0] <= 1);
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));

        // Verify the VM scaled gamma by the exact theoretical probability
        uint8_t m_phys = state.meas_record[0];
        uint8_t b_chosen = m_phys ^ static_cast<uint8_t>(px_v_before);
        double p_chosen = (b_chosen == 0) ? prob_b0 : prob_b1;
        double expected_gamma_scale = std::sqrt(total_prob / p_chosen);
        double actual_gamma_scale = std::abs(state.gamma()) / gamma_mag_before;
        CHECK_THAT(actual_gamma_scale, WithinRel(expected_gamma_scale, 1e-9));
    }
}

TEST_CASE("Compaction fuzz: active interfere preserves norm - k=4") {
    // Same as above but with OP_MEAS_ACTIVE_INTERFERE (X-basis fold).
    uint64_t seed = 0xDEAD0002;
    SchrodingerState state(4, 1);

    for (int trial = 0; trial < 200; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(test_lcg(seed));
        state.active_k = 4;

        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] = random_complex(seed);
            raw_norm += std::norm(state.v()[i]);
        }
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-10));

        // Calculate theoretical X-basis branch probabilities before measurement
        double prob_plus = 0.0;
        double prob_minus = 0.0;
        uint64_t half = 1ULL << 3;  // k-1 = 3
        for (uint64_t i = 0; i < half; ++i) {
            auto sum = state.v()[i] + state.v()[i + half];
            auto diff = state.v()[i] - state.v()[i + half];
            prob_plus += std::norm(sum);
            prob_minus += std::norm(diff);
        }
        double total_prob = prob_plus + prob_minus;
        double gamma_mag_before = std::abs(state.gamma());

        // Save frame bit before execute (measurement overwrites p_z)
        bool pz_v_before = (state.p_z.val >> 3) & 1;

        auto prog = make_program({make_meas_active_interfere(3, 0)}, 4, 1);
        execute(prog, state);

        CHECK(state.active_k == 3);
        CHECK(state.meas_record[0] <= 1);
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));

        // Verify the VM scaled gamma by the exact theoretical probability
        uint8_t m_phys = state.meas_record[0];
        uint8_t bx_chosen = m_phys ^ static_cast<uint8_t>(pz_v_before);
        double p_chosen = (bx_chosen == 0) ? prob_plus : prob_minus;
        double expected_gamma_scale = std::sqrt(total_prob / p_chosen);
        double actual_gamma_scale = std::abs(state.gamma()) / gamma_mag_before;
        CHECK_THAT(actual_gamma_scale, WithinRel(expected_gamma_scale, 1e-9));
    }
}

TEST_CASE("Compaction fuzz: cascaded measurements k=4 to k=0") {
    // Measure all 4 axes one at a time, verifying norm at each step.
    // Alternates diagonal and interfere to exercise both paths.
    uint64_t seed = 0xDEAD0003;
    SchrodingerState state(4, 4);

    for (int trial = 0; trial < 100; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(test_lcg(seed));
        state.active_k = 4;

        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] = random_complex(seed);
            raw_norm += std::norm(state.v()[i]);
        }
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        // Measure axis 3 (diagonal), then 2 (interfere), then 1 (diagonal), then 0 (interfere)
        auto prog =
            make_program({make_meas_active_diagonal(3, 0), make_meas_active_interfere(2, 1),
                          make_meas_active_diagonal(1, 2), make_meas_active_interfere(0, 3)},
                         4, 4);
        execute(prog, state);

        CHECK(state.active_k == 0);
        // After all measurements, norm should still be 1
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Compaction fuzz: diagonal with pre-existing Pauli frame") {
    // Verify compaction preserves norm even when p_x and p_z are non-trivial.
    uint64_t seed = 0xDEAD0004;
    SchrodingerState state(4, 1);

    for (int trial = 0; trial < 100; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(test_lcg(seed));
        state.active_k = 4;

        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] = random_complex(seed);
            raw_norm += std::norm(state.v()[i]);
        }
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        // Set random frame bits
        uint64_t px = test_lcg(seed) & 0xF;
        uint64_t pz = test_lcg(seed) & 0xF;
        state.p_x = stim::bitword<kStimWidth>(px);
        state.p_z = stim::bitword<kStimWidth>(pz);

        // Calculate theoretical branch probabilities before measurement
        double prob_b0 = 0.0;
        double prob_b1 = 0.0;
        uint64_t half = 1ULL << 3;
        for (uint64_t i = 0; i < half; ++i) {
            prob_b0 += std::norm(state.v()[i]);
            prob_b1 += std::norm(state.v()[i + half]);
        }
        double total_prob = prob_b0 + prob_b1;
        double gamma_mag_before = std::abs(state.gamma());

        // Save frame bit before execute (measurement overwrites p_x)
        bool px_v_before = (state.p_x.val >> 3) & 1;

        auto prog = make_program({make_meas_active_diagonal(3, 0)}, 4, 1);
        execute(prog, state);

        CHECK(state.active_k == 3);
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));

        // Verify exact probability scaling
        uint8_t m_phys = state.meas_record[0];
        uint8_t b_chosen = m_phys ^ static_cast<uint8_t>(px_v_before);
        double p_chosen = (b_chosen == 0) ? prob_b0 : prob_b1;
        double expected_gamma_scale = std::sqrt(total_prob / p_chosen);
        double actual_gamma_scale = std::abs(state.gamma()) / gamma_mag_before;
        CHECK_THAT(actual_gamma_scale, WithinRel(expected_gamma_scale, 1e-9));
    }
}

TEST_CASE("Compaction fuzz: interfere with pre-existing Pauli frame") {
    uint64_t seed = 0xDEAD0005;
    SchrodingerState state(4, 1);

    for (int trial = 0; trial < 100; ++trial) {
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(test_lcg(seed));
        state.active_k = 4;

        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] = random_complex(seed);
            raw_norm += std::norm(state.v()[i]);
        }
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 16; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        uint64_t px = test_lcg(seed) & 0xF;
        uint64_t pz = test_lcg(seed) & 0xF;
        state.p_x = stim::bitword<kStimWidth>(px);
        state.p_z = stim::bitword<kStimWidth>(pz);

        // Calculate theoretical X-basis branch probabilities before measurement
        double prob_plus = 0.0;
        double prob_minus = 0.0;
        uint64_t half = 1ULL << 3;
        for (uint64_t i = 0; i < half; ++i) {
            auto sum = state.v()[i] + state.v()[i + half];
            auto diff = state.v()[i] - state.v()[i + half];
            prob_plus += std::norm(sum);
            prob_minus += std::norm(diff);
        }
        double total_prob = prob_plus + prob_minus;
        double gamma_mag_before = std::abs(state.gamma());

        // Save frame bit before execute (measurement overwrites p_z)
        bool pz_v_before = (state.p_z.val >> 3) & 1;

        auto prog = make_program({make_meas_active_interfere(3, 0)}, 4, 1);
        execute(prog, state);

        CHECK(state.active_k == 3);
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));

        // Verify exact probability scaling
        uint8_t m_phys = state.meas_record[0];
        uint8_t bx_chosen = m_phys ^ static_cast<uint8_t>(pz_v_before);
        double p_chosen = (bx_chosen == 0) ? prob_plus : prob_minus;
        double expected_gamma_scale = std::sqrt(total_prob / p_chosen);
        double actual_gamma_scale = std::abs(state.gamma()) / gamma_mag_before;
        CHECK_THAT(actual_gamma_scale, WithinRel(expected_gamma_scale, 1e-9));
    }
}

// =============================================================================
// Zero-Probability Branch Stability
// =============================================================================

TEST_CASE("Zero-prob: minus state interfere is deterministic b_x=1") {
    // |-> = (|0> - |1>)/sqrt(2). X-basis measurement: prob_plus=0, prob_minus=4.
    // Must yield b_x=1 deterministically without NaN/Inf in gamma.
    SchrodingerState state(2, 1);
    for (uint32_t trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 31 + 7);
        state.active_k = 1;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {-1.0, 0.0};
        state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

        auto prog = make_program({make_meas_active_interfere(0, 0)}, 2, 1);
        execute(prog, state);

        CHECK(state.meas_record[0] == 1);  // deterministic |-> -> b_x=1
        CHECK(state.active_k == 0);
        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Zero-prob: plus state interfere is deterministic b_x=0") {
    // |+> = (|0> + |1>)/sqrt(2). prob_minus=0, must yield b_x=0.
    SchrodingerState state(2, 1);
    for (uint32_t trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 13 + 3);
        state.active_k = 1;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {1.0, 0.0};
        state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

        auto prog = make_program({make_meas_active_interfere(0, 0)}, 2, 1);
        execute(prog, state);

        CHECK(state.meas_record[0] == 0);
        CHECK(state.active_k == 0);
        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Zero-prob: zero-state diagonal is deterministic b=0") {
    // |0> state. prob_b1=0, must yield b=0.
    SchrodingerState state(2, 1);
    for (uint32_t trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 17 + 5);
        state.active_k = 1;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {0.0, 0.0};

        auto prog = make_program({make_meas_active_diagonal(0, 0)}, 2, 1);
        execute(prog, state);

        CHECK(state.meas_record[0] == 0);
        CHECK(state.active_k == 0);
        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Zero-prob: one-state diagonal is deterministic b=1") {
    // |1> state. prob_b0=0, must yield b=1.
    SchrodingerState state(2, 1);
    for (uint32_t trial = 0; trial < 50; ++trial) {
        INFO("Trial: " << trial);
        state.reset(trial * 19 + 11);
        state.active_k = 1;
        state.v()[0] = {0.0, 0.0};
        state.v()[1] = {1.0, 0.0};

        auto prog = make_program({make_meas_active_diagonal(0, 0)}, 2, 1);
        execute(prog, state);

        CHECK(state.meas_record[0] == 1);
        CHECK(state.active_k == 0);
        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Zero-prob: multi-qubit deterministic interfere") {
    // k=3 state where axis 2 is in exact |-> across all subspace combinations.
    // v[i] for i with bit 2 set = -v[i with bit 2 cleared].
    // Interfere on axis 2 must be deterministic b_x=1.
    SchrodingerState state(3, 1);
    for (uint32_t trial = 0; trial < 50; ++trial) {
        uint64_t seed = 0xBEEF0000 + trial;
        INFO("Trial: " << trial << " | Seed: " << seed);
        state.reset(seed);
        state.active_k = 3;

        // Fill lower half randomly, upper half = -lower
        double raw_norm = 0.0;
        for (uint64_t i = 0; i < 4; ++i) {
            state.v()[i] = random_complex(seed);
            state.v()[i + 4] = -state.v()[i];  // bit 2 = |-> pattern
            raw_norm += 2.0 * std::norm(state.v()[i]);
        }
        double scale = 1.0 / std::sqrt(raw_norm);
        for (uint64_t i = 0; i < 8; ++i) {
            state.v()[i] *= scale;
        }
        state.set_gamma({1.0, 0.0});

        auto prog = make_program({make_meas_active_interfere(2, 0)}, 3, 1);
        execute(prog, state);

        CHECK(state.meas_record[0] == 1);  // deterministic |->
        CHECK(state.active_k == 2);
        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK_THAT(physical_norm(state), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("RISC Integration: Amortized renormalization prevents IEEE-754 drift") {
    // Renormalization only fires on magnitude-changing opcodes (EXPAND,
    // MEAS_ACTIVE_DIAGONAL, MEAS_ACTIVE_INTERFERE). Use EXPAND to trigger it.
    // 3 physical qubits, 1 active (k=1). EXPAND promotes axis 1 -> k becomes 2.
    SchrodingerState state(3, 0);
    state.active_k = 1;

    auto prog = make_program({make_expand(1)}, 3);

    SECTION("Rescues from severe underflow - gamma approaching 0") {
        double extreme_scale = 1e150;
        state.set_gamma({1.0 / extreme_scale, 0.0});
        state.v()[0] = {extreme_scale, 0.0};
        state.v()[1] = {0.0, 0.0};

        execute(prog, state);

        CHECK_THAT(std::abs(state.gamma()), WithinAbs(1.0, 0.01));
        CHECK(std::isfinite(state.v()[0].real()));
        CHECK(std::isfinite(state.v()[1].real()));
    }

    SECTION("Rescues from severe overflow - gamma approaching Infinity") {
        double extreme_scale = 1e150;
        state.set_gamma({0.0, extreme_scale});
        state.v()[0] = {0.0, 0.0};
        state.v()[1] = {1.0 / extreme_scale, 0.0};

        execute(prog, state);

        CHECK(std::isfinite(state.gamma().real()));
        CHECK(std::isfinite(state.gamma().imag()));
        CHECK(std::isfinite(state.v()[0].real()));
        CHECK(std::isfinite(state.v()[1].real()));
    }
}

// =============================================================================
// OP_POSTSELECT Tests
// =============================================================================

TEST_CASE("POSTSELECT: discards shot when parity is non-zero") {
    // Circuit: measure qubit 0 in X basis (random outcome), then
    // post-select on detector parity of that measurement.
    // If outcome == 1, parity != 0 -> discard.
    // After postselect, there is a second measurement that should NOT execute.
    CompiledModule mod;
    mod.peak_rank = 0;
    mod.num_measurements = 2;
    mod.total_meas_slots = 2;
    mod.num_detectors = 1;

    // meas[0]: dormant random measurement on qubit 0
    mod.bytecode.push_back(make_meas_dormant_random(0, 0));

    // OP_POSTSELECT: check detector parity of meas[0]
    mod.constant_pool.detector_targets.push_back({0});
    mod.bytecode.push_back(make_postselect(0, 0));

    // meas[1]: a second dormant static measurement (should not run if discarded)
    Instruction meas1 = make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 1, true);
    mod.bytecode.push_back(meas1);

    // Run many shots: some will have meas[0]==1 (discarded), others meas[0]==0
    uint32_t discarded_count = 0;
    uint32_t survived_count = 0;

    for (uint64_t seed = 0; seed < 200; ++seed) {
        SchrodingerState state(mod.peak_rank, mod.total_meas_slots, mod.num_detectors, 0, seed);
        execute(mod, state);

        if (state.discarded) {
            ++discarded_count;
            // det_record should be 0 (written by postselect before aborting)
            CHECK(state.det_record[0] == 0);
            // meas[1] should be 0 (never executed, initialized to 0)
            CHECK(state.meas_record[1] == 0);
        } else {
            ++survived_count;
            // meas[0] must have been 0 for survival
            CHECK(state.meas_record[0] == 0);
            // meas[1] should have been executed (sign=true -> outcome=1)
            CHECK(state.meas_record[1] == 1);
        }
    }

    // With a random dormant measurement, roughly half should be discarded
    CHECK(discarded_count > 0);
    CHECK(survived_count > 0);
}

TEST_CASE("POSTSELECT: passes when parity is zero") {
    // Deterministic circuit: meas[0]=0, so postselect parity is 0.
    // All instructions after postselect should execute normally.
    CompiledModule mod;
    mod.peak_rank = 0;
    mod.num_measurements = 2;
    mod.total_meas_slots = 2;
    mod.num_detectors = 1;

    // meas[0]: dormant static, sign=false -> outcome=0
    mod.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 0, false));

    // OP_POSTSELECT: parity of meas[0] = 0 -> pass
    mod.constant_pool.detector_targets.push_back({0});
    mod.bytecode.push_back(make_postselect(0, 0));

    // meas[1]: dormant static, sign=true -> outcome=1
    mod.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 1, true));

    SchrodingerState state(mod.peak_rank, mod.total_meas_slots, mod.num_detectors, 0, 42);
    execute(mod, state);

    CHECK_FALSE(state.discarded);
    CHECK(state.meas_record[0] == 0);
    CHECK(state.meas_record[1] == 1);
    CHECK(state.det_record[0] == 0);
}

TEST_CASE("POSTSELECT: reset clears stale data after discarded shot") {
    // Verify that after a discarded shot, reset() zeroes classical records
    // so the next shot doesn't see stale data.
    CompiledModule mod;
    mod.peak_rank = 0;
    mod.num_measurements = 2;
    mod.total_meas_slots = 2;
    mod.num_detectors = 1;

    // meas[0]: dormant static, sign=true -> always outcome=1
    mod.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 0, true));

    // OP_POSTSELECT: parity of meas[0] = 1 -> always discard
    mod.constant_pool.detector_targets.push_back({0});
    mod.bytecode.push_back(make_postselect(0, 0));

    // meas[1]: should never execute due to abort
    mod.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 1, true));

    SchrodingerState state(mod.peak_rank, mod.total_meas_slots, mod.num_detectors, 0, 0);
    execute(mod, state);
    CHECK(state.discarded);
    CHECK(state.meas_record[0] == 1);  // was set before postselect
    CHECK(state.meas_record[1] == 0);  // never reached

    // Reset and run a second shot (which also gets discarded)
    state.reset(1);
    CHECK_FALSE(state.discarded);      // reset clears discarded flag
    CHECK(state.meas_record[0] == 0);  // stale data cleared
    CHECK(state.meas_record[1] == 0);  // stale data cleared

    execute(mod, state);
    CHECK(state.discarded);
}

TEST_CASE("POSTSELECT: sample returns all shots including discarded") {
    // Verify sample() still returns full arrays and does not crash.
    CompiledModule mod;
    mod.peak_rank = 0;
    mod.num_measurements = 1;
    mod.total_meas_slots = 1;
    mod.num_detectors = 1;

    // meas[0]: dormant random
    mod.bytecode.push_back(make_meas_dormant_random(0, 0));

    // OP_POSTSELECT on meas[0]
    mod.constant_pool.detector_targets.push_back({0});
    mod.bytecode.push_back(make_postselect(0, 0));

    auto result = sample(mod, 100, 42);

    // Shape should be 100 shots regardless of how many were discarded
    CHECK(result.measurements.size() == 100);
    CHECK(result.detectors.size() == 100);
}
