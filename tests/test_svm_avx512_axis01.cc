// Tests for AVX-512 axis 0/1 fast paths in SVM kernels.
//
// These paths only activate when active_k >= kMinRankFor3DLoop (9).
// Previous tests only exercised active_k <= 4, so these bugs were invisible.
// Each test constructs a state at rank 10, applies a gate on axis 0 or 1,
// then verifies the result against manually computed expected values.

#include "clifft/backend/backend.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace clifft;
using clifft::test::check_complex;

namespace {

constexpr double kTol = 1e-12;
constexpr uint32_t kHighRank = 10;  // 1024 elements, well above kMinRankFor3DLoop=9

CompiledModule make_program(std::vector<Instruction> bytecode, uint32_t peak_rank) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = 0;
    mod.num_detectors = 0;
    mod.num_observables = 0;
    return mod;
}

// Fill state with a known non-trivial pattern so we can verify transformations.
// Each amplitude gets a unique value based on its index.
void fill_state_with_pattern(SchrodingerState& state) {
    uint64_t n = 1ULL << state.active_k;
    for (uint64_t i = 0; i < n; ++i) {
        double re = static_cast<double>(i) / static_cast<double>(n);
        double im = static_cast<double>(n - i) / static_cast<double>(n * 2);
        state.v()[i] = {re, im};
    }
}

}  // namespace

// =============================================================================
// Hadamard on axis 0 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: Hadamard on axis 0 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    // Save original values for verification
    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // Apply H on axis 0
    auto prog = make_program({make_array_h(0)}, kHighRank);
    execute(prog, state);

    // Verify: H on axis 0 means for each pair (idx, idx^1):
    //   new[idx with bit0=0] = (old[idx&~1] + old[idx|1]) / sqrt(2)
    //   new[idx with bit0=1] = (old[idx&~1] - old[idx|1]) / sqrt(2)
    double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    for (uint64_t i = 0; i < n; i += 2) {
        auto a = orig[i];      // bit 0 = 0
        auto b = orig[i + 1];  // bit 0 = 1
        std::complex<double> expected_0 = (a + b) * inv_sqrt2;
        std::complex<double> expected_1 = (a - b) * inv_sqrt2;
        check_complex(state.v()[i], expected_0, kTol);
        check_complex(state.v()[i + 1], expected_1, kTol);
    }
}

TEST_CASE("AVX512 axis01: Hadamard on axis 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_h(1)}, kHighRank);
    execute(prog, state);

    // H on axis 1: pairs are (idx, idx^2) where bit 1 differs
    double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    for (uint64_t i = 0; i < n; ++i) {
        if (i & 2)
            continue;          // only check the bit1=0 half
        auto a = orig[i];      // bit 1 = 0
        auto b = orig[i | 2];  // bit 1 = 1
        std::complex<double> expected_0 = (a + b) * inv_sqrt2;
        std::complex<double> expected_1 = (a - b) * inv_sqrt2;
        check_complex(state.v()[i], expected_0, kTol);
        check_complex(state.v()[i | 2], expected_1, kTol);
    }
}

// =============================================================================
// Hadamard round-trip: H*H = I
// =============================================================================

TEST_CASE("AVX512 axis01: H squared is identity on axis 0 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // Apply H twice -- should return to original state
    // Note: exec_array_h also updates frame (p_x, p_z), so we need two
    // separate executions that each do frame + array
    auto prog = make_program({make_array_h(0), make_array_h(0)}, kHighRank);
    execute(prog, state);

    for (uint64_t i = 0; i < n; ++i) {
        check_complex(state.v()[i], orig[i], kTol);
    }
}

TEST_CASE("AVX512 axis01: H squared is identity on axis 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_h(1), make_array_h(1)}, kHighRank);
    execute(prog, state);

    for (uint64_t i = 0; i < n; ++i) {
        check_complex(state.v()[i], orig[i], kTol);
    }
}

// =============================================================================
// U2 on axis 0 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: U2 on axis 0 at rank 10") {
    // Build a U2 constant pool entry for a known unitary.
    // Use the Hadamard matrix as our U2 (frame state 0 = identity frame).
    // This lets us cross-check against the H test above.
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // Construct a program with a U2 that applies an arbitrary 2x2 matrix.
    // Use a non-trivial matrix: Ry(pi/3) = [[cos(pi/6), -sin(pi/6)],
    //                                       [sin(pi/6),  cos(pi/6)]]
    double c = std::cos(std::numbers::pi / 6.0);  // sqrt(3)/2
    double s = std::sin(std::numbers::pi / 6.0);  // 1/2
    std::complex<double> m00 = {c, 0};
    std::complex<double> m01 = {-s, 0};
    std::complex<double> m10 = {s, 0};
    std::complex<double> m11 = {c, 0};

    // Create the constant pool entry
    ConstantPool pool;
    FusedU2Node node{};
    // For frame state 0 (p_x=0, p_z=0): just apply the matrix directly
    node.matrices[0][0] = m00;
    node.matrices[0][1] = m01;
    node.matrices[0][2] = m10;
    node.matrices[0][3] = m11;
    node.gamma_multipliers[0] = {1.0, 0.0};
    node.out_states[0] = 0;
    // Fill other frame states with identity to avoid UB
    for (int fs = 1; fs < 4; ++fs) {
        node.matrices[fs][0] = {1, 0};
        node.matrices[fs][1] = {0, 0};
        node.matrices[fs][2] = {0, 0};
        node.matrices[fs][3] = {1, 0};
        node.gamma_multipliers[fs] = {1.0, 0.0};
        node.out_states[fs] = static_cast<uint8_t>(fs);
    }
    pool.fused_u2_nodes.push_back(node);

    // Build program with the constant pool
    CompiledModule mod;
    mod.bytecode = {make_array_u2(0, 0)};
    mod.peak_rank = kHighRank;
    mod.num_measurements = 0;
    mod.num_detectors = 0;
    mod.num_observables = 0;
    mod.constant_pool = std::move(pool);

    execute(mod, state);

    // Verify: U2 on axis 0 transforms pairs (i, i^1)
    for (uint64_t i = 0; i < n; i += 2) {
        auto a = orig[i];      // |0>
        auto b = orig[i + 1];  // |1>
        std::complex<double> expected_0 = m00 * a + m01 * b;
        std::complex<double> expected_1 = m10 * a + m11 * b;
        check_complex(state.v()[i], expected_0, kTol);
        check_complex(state.v()[i + 1], expected_1, kTol);
    }
}

TEST_CASE("AVX512 axis01: U2 on axis 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // T gate matrix diag(1, e^{i*pi/4}) -- complex m11 exercises real+imag paths.
    std::complex<double> m00 = {1.0, 0.0};
    std::complex<double> m01 = {0.0, 0.0};
    std::complex<double> m10 = {0.0, 0.0};
    std::complex<double> m11 = {std::cos(std::numbers::pi / 4.0), std::sin(std::numbers::pi / 4.0)};

    ConstantPool pool;
    FusedU2Node node{};
    node.matrices[0][0] = m00;
    node.matrices[0][1] = m01;
    node.matrices[0][2] = m10;
    node.matrices[0][3] = m11;
    node.gamma_multipliers[0] = {1.0, 0.0};
    node.out_states[0] = 0;
    for (int fs = 1; fs < 4; ++fs) {
        node.matrices[fs][0] = {1, 0};
        node.matrices[fs][1] = {0, 0};
        node.matrices[fs][2] = {0, 0};
        node.matrices[fs][3] = {1, 0};
        node.gamma_multipliers[fs] = {1.0, 0.0};
        node.out_states[fs] = static_cast<uint8_t>(fs);
    }
    pool.fused_u2_nodes.push_back(node);

    CompiledModule mod;
    mod.bytecode = {make_array_u2(1, 0)};
    mod.peak_rank = kHighRank;
    mod.num_measurements = 0;
    mod.num_detectors = 0;
    mod.num_observables = 0;
    mod.constant_pool = std::move(pool);

    execute(mod, state);

    // Verify: U2 on axis 1 transforms pairs (i, i^2)
    for (uint64_t i = 0; i < n; ++i) {
        if (i & 2)
            continue;
        auto a = orig[i];      // bit 1 = 0
        auto b = orig[i | 2];  // bit 1 = 1
        std::complex<double> expected_0 = m00 * a + m01 * b;
        std::complex<double> expected_1 = m10 * a + m11 * b;
        check_complex(state.v()[i], expected_0, kTol);
        check_complex(state.v()[i | 2], expected_1, kTol);
    }
}

// =============================================================================
// U2 with a fully dense unitary to catch coefficient placement errors
// =============================================================================

TEST_CASE("AVX512 axis01: U2 dense unitary on axis 0 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // Dense unitary where all 4 matrix elements are distinct and complex.
    // Use Ry(theta) * Rz(phi) for some angles to get a proper unitary.
    // Simpler: use the Hadamard-like matrix 1/sqrt2 * [[1, i], [i, 1]]
    double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    std::complex<double> m00 = {inv_sqrt2, 0};
    std::complex<double> m01 = {0, inv_sqrt2};
    std::complex<double> m10 = {0, inv_sqrt2};
    std::complex<double> m11 = {inv_sqrt2, 0};

    ConstantPool pool;
    FusedU2Node node{};
    node.matrices[0][0] = m00;
    node.matrices[0][1] = m01;
    node.matrices[0][2] = m10;
    node.matrices[0][3] = m11;
    node.gamma_multipliers[0] = {1.0, 0.0};
    node.out_states[0] = 0;
    for (int fs = 1; fs < 4; ++fs) {
        node.matrices[fs][0] = {1, 0};
        node.matrices[fs][1] = {0, 0};
        node.matrices[fs][2] = {0, 0};
        node.matrices[fs][3] = {1, 0};
        node.gamma_multipliers[fs] = {1.0, 0.0};
        node.out_states[fs] = static_cast<uint8_t>(fs);
    }
    pool.fused_u2_nodes.push_back(node);

    CompiledModule mod;
    mod.bytecode = {make_array_u2(0, 0)};
    mod.peak_rank = kHighRank;
    mod.num_measurements = 0;
    mod.num_detectors = 0;
    mod.num_observables = 0;
    mod.constant_pool = std::move(pool);

    execute(mod, state);

    for (uint64_t i = 0; i < n; i += 2) {
        auto a = orig[i];
        auto b = orig[i + 1];
        std::complex<double> expected_0 = m00 * a + m01 * b;
        std::complex<double> expected_1 = m10 * a + m11 * b;
        check_complex(state.v()[i], expected_0, kTol);
        check_complex(state.v()[i + 1], expected_1, kTol);
    }
}

// =============================================================================
// CNOT with one axis < 2 and other >= 2 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: CNOT axis 0 ctrl and axis 5 target at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cnot(0, 5)}, kHighRank);
    execute(prog, state);

    // CNOT(ctrl=0, tgt=5): flip bit 5 when bit 0 is set
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t src = i;
        if (i & 1) {  // bit 0 set -> flip bit 5
            src = i ^ (1ULL << 5);
        }
        check_complex(state.v()[i], orig[src], kTol);
    }
}

TEST_CASE("AVX512 axis01: CNOT axis 1 ctrl and axis 3 target at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cnot(1, 3)}, kHighRank);
    execute(prog, state);

    // CNOT(ctrl=1, tgt=3): flip bit 3 when bit 1 is set
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t src = i;
        if (i & 2) {  // bit 1 set -> flip bit 3
            src = i ^ (1ULL << 3);
        }
        check_complex(state.v()[i], orig[src], kTol);
    }
}

// =============================================================================
// CZ with one axis < 2 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: CZ axis 0 and axis 4 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cz(0, 4)}, kHighRank);
    execute(prog, state);

    // CZ: negate amplitude when both bits are set
    for (uint64_t i = 0; i < n; ++i) {
        auto expected = orig[i];
        if ((i & 1) && (i & (1ULL << 4))) {
            expected = -expected;
        }
        check_complex(state.v()[i], expected, kTol);
    }
}

TEST_CASE("AVX512 axis01: CZ axis 1 and axis 6 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cz(1, 6)}, kHighRank);
    execute(prog, state);

    for (uint64_t i = 0; i < n; ++i) {
        auto expected = orig[i];
        if ((i & 2) && (i & (1ULL << 6))) {
            expected = -expected;
        }
        check_complex(state.v()[i], expected, kTol);
    }
}

// =============================================================================
// SWAP with mixed axes (lo < 2, hi >= 2) at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: SWAP axis 0 and axis 4 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_swap(0, 4)}, kHighRank);
    execute(prog, state);

    // SWAP exchanges bit 0 and bit 4 in all indices
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t bit0 = (i >> 0) & 1;
        uint64_t bit4 = (i >> 4) & 1;
        uint64_t src = (i & ~((1ULL << 0) | (1ULL << 4))) | (bit0 << 4) | (bit4 << 0);
        check_complex(state.v()[i], orig[src], kTol);
    }
}

TEST_CASE("AVX512 axis01: SWAP axis 1 and axis 5 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_swap(1, 5)}, kHighRank);
    execute(prog, state);

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t bit1 = (i >> 1) & 1;
        uint64_t bit5 = (i >> 5) & 1;
        uint64_t src = (i & ~((1ULL << 1) | (1ULL << 5))) | (bit1 << 5) | (bit5 << 1);
        check_complex(state.v()[i], orig[src], kTol);
    }
}

// =============================================================================
// Phase rotation on axis 0/1 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: phase rotation on axis 0 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    // T gate: e^{i*pi/4}
    double angle = std::numbers::pi / 4.0;
    std::complex<double> phase = {std::cos(angle), std::sin(angle)};
    auto prog = make_program({make_array_rot(0, phase.real(), phase.imag())}, kHighRank);
    execute(prog, state);

    // Phase on axis 0: multiply |1> amplitudes (bit 0 = 1) by phase
    for (uint64_t i = 0; i < n; ++i) {
        auto expected = orig[i];
        if (i & 1) {
            expected *= phase;
        }
        check_complex(state.v()[i], expected, kTol);
    }
}

TEST_CASE("AVX512 axis01: phase rotation on axis 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    double angle = std::numbers::pi / 3.0;
    std::complex<double> phase = {std::cos(angle), std::sin(angle)};
    auto prog = make_program({make_array_rot(1, phase.real(), phase.imag())}, kHighRank);
    execute(prog, state);

    for (uint64_t i = 0; i < n; ++i) {
        auto expected = orig[i];
        if (i & 2) {  // bit 1 set
            expected *= phase;
        }
        check_complex(state.v()[i], expected, kTol);
    }
}

// =============================================================================
// CNOT with both axes 0,1 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: CNOT axes 0 and 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cnot(0, 1)}, kHighRank);
    execute(prog, state);

    // CNOT(ctrl=0, tgt=1): flip bit 1 when bit 0 is set
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t src = i;
        if (i & 1) {
            src = i ^ 2;
        }
        check_complex(state.v()[i], orig[src], kTol);
    }
}

// =============================================================================
// CZ with both axes 0,1 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: CZ axes 0 and 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_cz(0, 1)}, kHighRank);
    execute(prog, state);

    // CZ: negate |11> (both bits 0 and 1 set)
    for (uint64_t i = 0; i < n; ++i) {
        auto expected = orig[i];
        if ((i & 1) && (i & 2)) {
            expected = -expected;
        }
        check_complex(state.v()[i], expected, kTol);
    }
}

// =============================================================================
// SWAP axes 0 and 1 at high rank
// =============================================================================

TEST_CASE("AVX512 axis01: SWAP axes 0 and 1 at rank 10") {
    SchrodingerState state(kHighRank, 0);
    state.active_k = kHighRank;
    fill_state_with_pattern(state);

    uint64_t n = 1ULL << kHighRank;
    std::vector<std::complex<double>> orig(n);
    for (uint64_t i = 0; i < n; ++i) {
        orig[i] = state.v()[i];
    }

    auto prog = make_program({make_array_swap(0, 1)}, kHighRank);
    execute(prog, state);

    // SWAP bits 0 and 1
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t bit0 = (i >> 0) & 1;
        uint64_t bit1 = (i >> 1) & 1;
        uint64_t src = (i & ~3u) | (bit0 << 1) | (bit1 << 0);
        check_complex(state.v()[i], orig[src], kTol);
    }
}
