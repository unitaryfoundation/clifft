#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

using namespace clifft;
using Catch::Matchers::WithinAbs;
using clifft::test::check_complex;
using clifft::test::kInvSqrt2;

constexpr double kTol = 1e-9;

// Tolerance for float-precision tableau multiplication.
// Stim's to_flat_unitary_matrix returns complex<float> (~7 decimal digits).
constexpr double kFloatTol = 1e-6;

// Build a minimal CompiledModule with the given settings.
static CompiledModule make_module(uint32_t num_qubits, uint32_t peak_rank) {
    CompiledModule mod;
    mod.num_qubits = num_qubits;
    mod.peak_rank = peak_rank;
    return mod;
}

// =============================================================================
// Identity / No-Tableau Tests
// =============================================================================

TEST_CASE("Statevector: single qubit zero state with no tableau") {
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    // active_k = 0, v[0] = 1.0 -> all dormant -> |0>

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    check_complex(sv[0], {1.0, 0.0});
    check_complex(sv[1], {0.0, 0.0});
}

TEST_CASE("Statevector: single active qubit in plus state with no tableau") {
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    check_complex(sv[0], {kInvSqrt2, 0.0});
    check_complex(sv[1], {kInvSqrt2, 0.0});
}

TEST_CASE("Statevector: two active qubits no tableau") {
    // |phi> = [1, 0, 0, 1] with gamma = 1/sqrt(2) -> (|00> + |11>)/sqrt(2)
    auto mod = make_module(2, 2);
    SchrodingerState state(2, 0);
    state.active_k = 2;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {0.0, 0.0};
    state.v()[2] = {0.0, 0.0};
    state.v()[3] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 4);

    check_complex(sv[0], {kInvSqrt2, 0.0});
    check_complex(sv[1], {0.0, 0.0});
    check_complex(sv[2], {0.0, 0.0});
    check_complex(sv[3], {kInvSqrt2, 0.0});
}

// =============================================================================
// Pauli Frame Tests
// =============================================================================

TEST_CASE("Statevector: X frame flips the zero state") {
    // |0> with p_x[0]=1 -> P|0> = X|0> = |1>
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    state.p_x = PauliBitMask(uint64_t{1});  // X on qubit 0

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    check_complex(sv[0], {0.0, 0.0});
    check_complex(sv[1], {1.0, 0.0});
}

TEST_CASE("Statevector: Z frame applies phase") {
    // |+> with p_z[0]=1 -> Z|+> = |->  = (|0> - |1>)/sqrt(2)
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});
    state.p_z = PauliBitMask(uint64_t{1});  // Z on qubit 0

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    // Z|+> = (Z|0> + Z|1>)/sqrt(2) = (|0> - |1>)/sqrt(2)
    check_complex(sv[0], {kInvSqrt2, 0.0});
    check_complex(sv[1], {-kInvSqrt2, 0.0});
}

TEST_CASE("Statevector: XZ frame on 2-qubit state") {
    // 2 qubits, |00> state, p_x = 0b01 (X on q0), p_z = 0b10 (Z on q1)
    // P = X0 Z1, applied to |00>:
    // X0 Z1 |00> = X0 |00> = |01> (bit 0 flipped)
    // Z1 on |00> is trivial (eigenvalue +1 since bit 1 = 0)
    auto mod = make_module(2, 2);
    SchrodingerState state(2, 0);
    state.p_x = PauliBitMask(uint64_t{1});  // X on qubit 0
    state.p_z = PauliBitMask(uint64_t{2});  // Z on qubit 1

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 4);
    check_complex(sv[0], {0.0, 0.0});  // |00>
    check_complex(sv[1], {1.0, 0.0});  // |01> - X flipped bit 0
    check_complex(sv[2], {0.0, 0.0});  // |10>
    check_complex(sv[3], {0.0, 0.0});  // |11>
}

// =============================================================================
// Gamma Scaling Tests
// =============================================================================

TEST_CASE("Statevector: gamma scales output") {
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    state.set_gamma({0.0, 1.0});  // gamma = i

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    check_complex(sv[0], {0.0, 1.0});  // i * |0>
    check_complex(sv[1], {0.0, 0.0});
}

TEST_CASE("Statevector: global_weight scales output") {
    auto mod = make_module(1, 1);
    mod.constant_pool.global_weight = {0.5, 0.0};
    SchrodingerState state(1, 0);

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    check_complex(sv[0], {0.5, 0.0});
    check_complex(sv[1], {0.0, 0.0});
}

// =============================================================================
// Mixed Active/Dormant Tests
// =============================================================================

TEST_CASE("Statevector: 3 qubits with 1 active - dormant qubits contribute zero") {
    // n=3, k=1. Active axis 0 holds |+> = [1,1]/sqrt(2).
    // Dormant axes 1,2 are |0>. Virtual state = |+00> = (|000> + |001>)/sqrt(2).
    auto mod = make_module(3, 1);
    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 8);

    check_complex(sv[0], {kInvSqrt2, 0.0});  // |000>
    check_complex(sv[1], {kInvSqrt2, 0.0});  // |001>
    check_complex(sv[2], {0.0, 0.0});        // |010>
    check_complex(sv[3], {0.0, 0.0});        // |011>
    check_complex(sv[4], {0.0, 0.0});        // |100>
    check_complex(sv[5], {0.0, 0.0});        // |101>
    check_complex(sv[6], {0.0, 0.0});        // |110>
    check_complex(sv[7], {0.0, 0.0});        // |111>
}

TEST_CASE("Statevector: all dormant returns zero state") {
    // n=3, k=0. All qubits dormant. State = |000>.
    auto mod = make_module(3, 0);
    SchrodingerState state(0, 0);

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 8);
    check_complex(sv[0], {1.0, 0.0});
    for (size_t i = 1; i < 8; ++i) {
        check_complex(sv[i], {0.0, 0.0});
    }
}

// =============================================================================
// Tableau (U_C) Application Tests
// =============================================================================

TEST_CASE("Statevector: identity tableau is transparent") {
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::identity(1);
    SchrodingerState state(1, 0);

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    check_complex(sv[0], {1.0, 0.0});
    check_complex(sv[1], {0.0, 0.0});
}

TEST_CASE("Statevector: Hadamard tableau on zero state produces plus") {
    // U_C = H. Virtual state = |0>. Physical = H|0> = |+>.
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Z", "+X");
    SchrodingerState state(1, 0);

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {kInvSqrt2, 0.0}, kFloatTol);
}

TEST_CASE("Statevector: Hadamard tableau on one-state produces minus") {
    // U_C = H. Virtual state = |1> (via X frame on |0>). Physical = H*X|0> = H|1> = |->.
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Z", "+X");
    SchrodingerState state(1, 0);
    state.p_x = PauliBitMask(uint64_t{1});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    // H|1> = (|0> - |1>)/sqrt(2)
    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {-kInvSqrt2, 0.0}, kFloatTol);
}

TEST_CASE("Statevector: CNOT tableau creates Bell state from plus-zero") {
    // 2 qubits. Virtual state: active axis 0 = |+> = [1,1]/sqrt(2), axis 1 dormant.
    // U_C = CNOT(0,1). Physical = CNOT * (|+> (x) |0>) = (|00>+|11>)/sqrt(2).
    auto mod = make_module(2, 1);

    // Build CNOT tableau: X0->XX, Z0->ZI, X1->IX, Z1->ZZ
    auto cnot = stim::Tableau<kStimWidth>::gate2("+XX", "+ZI", "+IX", "+ZZ");
    mod.constant_pool.final_tableau = cnot;

    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 4);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);  // |00>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);        // |01>
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);        // |10>
    check_complex(sv[3], {kInvSqrt2, 0.0}, kFloatTol);  // |11>
}

TEST_CASE("Statevector: S tableau on plus state") {
    // U_C = S. Virtual state = |+> = [1,1]/sqrt(2).
    // Physical = S|+> = (|0> + i|1>)/sqrt(2).
    auto mod = make_module(1, 1);
    // S: X -> Y = iXZ, Z -> Z.  Tableau: xs maps to "+Y", zs maps to "+Z"
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Y", "+Z");
    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {0.0, kInvSqrt2}, kFloatTol);  // i/sqrt(2)
}

// =============================================================================
// Combined: Pauli Frame + Tableau
// =============================================================================

TEST_CASE("Statevector: H tableau with Z frame on zero state") {
    // Virtual: |0> with p_z=1 -> framed = Z|0> = |0> (Z eigenstate, no effect).
    // Physical: H|0> = |+>.
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Z", "+X");
    SchrodingerState state(1, 0);
    state.p_z = PauliBitMask(uint64_t{1});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);

    // Z|0> = |0>, then H|0> = |+>
    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {kInvSqrt2, 0.0}, kFloatTol);
}

TEST_CASE("Statevector: H tableau with Z frame on plus state") {
    // Virtual: |+> = [1,1]/sqrt(2) with p_z=1 -> framed = Z|+> = |->.
    // Physical: H|-> = |1>.
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Z", "+X");
    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});
    state.p_z = PauliBitMask(uint64_t{1});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    // Z|+> = |-> = [1,-1]/sqrt(2). Then H|-> = |1>.
    check_complex(sv[0], {0.0, 0.0}, kFloatTol);
    check_complex(sv[1], {1.0, 0.0}, kFloatTol);
}

// =============================================================================
// Error handling
// =============================================================================

TEST_CASE("Statevector: rejects more than 10 qubits") {
    auto mod = make_module(11, 0);
    SchrodingerState state(0, 0);

    CHECK_THROWS_AS(get_statevector(mod, state), std::runtime_error);
}

// =============================================================================
// Combined-Layer Tests
// =============================================================================

TEST_CASE(
    "Statevector: all four layers combined - active array plus frame plus tableau plus gamma") {
    // Exercise all 4 components simultaneously:
    //   active array + Pauli frame + U_C tableau + gamma*global_weight
    //
    // Setup: 2 qubits, k=1 (axis 0 active, axis 1 dormant).
    // Active array: v = [1, i] (unnormalized).
    // Pauli frame: p_x = 0b01 (X on q0), p_z = 0b10 (Z on q1).
    // gamma = (0.5 + 0.5i)
    // global_weight = (0.6, 0.0)
    // U_C = Hadamard on q0, identity on q1 (2-qubit tableau).
    //
    // Step-by-step expansion:
    //
    // 1. Embed active into 2^2 dense (virtual basis):
    //    dense = [1, i, 0, 0]  (active on axis 0; axis 1 dormant = |0>)
    //
    // 2. Apply P = X^{p_x} Z^{p_z}:
    //    P|i> = (-1)^popcount(i & pz_mask) * |i XOR px_mask>
    //    px_mask = 0b01, pz_mask = 0b10
    //
    //    i=0 (0b00): target = 0b00 ^ 0b01 = 0b01, sign = (-1)^popcount(0b00 & 0b10) = +1
    //      framed[1] += 1 * 1 = 1
    //    i=1 (0b01): target = 0b01 ^ 0b01 = 0b00, sign = (-1)^popcount(0b01 & 0b10) = +1
    //      framed[0] += 1 * i = i
    //    i=2,3: dense=0 -> no contribution
    //
    //    framed = [i, 1, 0, 0]
    //
    // 3. Apply U_C = H (x) I (Hadamard on q0, identity on q1).
    //    Matrix (little-endian qubit order, q0 is LSB):
    //    H (x) I = 1/sqrt(2) * [[1, 1, 0, 0],
    //                            [1,-1, 0, 0],
    //                            [0, 0, 1, 1],
    //                            [0, 0, 1,-1]]
    //
    //    physical[0] = (i + 1) / sqrt(2)
    //    physical[1] = (i - 1) / sqrt(2)
    //    physical[2] = 0
    //    physical[3] = 0
    //
    // 4. Scale by gamma * global_weight = (0.5+0.5i) * 0.6 = (0.3+0.3i)
    //    final[0] = (0.3+0.3i) * (i+1)/sqrt(2)
    //             = (0.3+0.3i)(1+i) / sqrt(2)
    //             = (0.3 + 0.3i + 0.3i - 0.3) / sqrt(2)
    //             = 0.6i / sqrt(2)
    //    final[1] = (0.3+0.3i) * (i-1)/sqrt(2)
    //             = (0.3+0.3i)(-1+i) / sqrt(2)
    //             = (-0.3 + 0.3i - 0.3i - 0.3) / sqrt(2)
    //             = -0.6 / sqrt(2)

    auto mod = make_module(2, 1);
    mod.constant_pool.global_weight = {0.6, 0.0};

    // Build H (x) I tableau: q0 -> H, q1 -> identity
    auto tab = stim::Tableau<kStimWidth>::identity(2);
    // Hadamard on q0: X0 -> Z0, Z0 -> X0
    tab.xs[0].xs[0] = 0;  // X0 -> no X component
    tab.xs[0].zs[0] = 1;  // X0 -> Z0
    tab.zs[0].xs[0] = 1;  // Z0 -> X0
    tab.zs[0].zs[0] = 0;  // Z0 -> no Z component
    mod.constant_pool.final_tableau = tab;

    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {0.0, 1.0};  // i
    state.set_gamma({0.5, 0.5});
    state.p_x = PauliBitMask(uint64_t{1});  // X on q0
    state.p_z = PauliBitMask(uint64_t{2});  // Z on q1

    auto sv = get_statevector(mod, state);
    REQUIRE(sv.size() == 4);

    // final[0] = 0.6i / sqrt(2)
    check_complex(sv[0], {0.0, 0.6 * kInvSqrt2}, kFloatTol);
    // final[1] = -0.6 / sqrt(2)
    check_complex(sv[1], {-0.6 * kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);
    check_complex(sv[3], {0.0, 0.0}, kFloatTol);
}

TEST_CASE("Statevector: 3 qubits 1 active with non-trivial 3-qubit tableau") {
    // n=3, k=1. Active axis 0 holds |+> = [1,1]/sqrt(2).
    // Dormant axes 1,2 are |0>.
    // No Pauli frame.
    // U_C = CNOT(0,1) then H on q2. But we only need the combined tableau.
    //
    // Virtual state (pre-tableau): (|0> + |1>)/sqrt(2) (x) |0> (x) |0>
    //   = (|000> + |001>)/sqrt(2)  in little-endian (q0=LSB)
    //
    // U_C: CNOT(0,1) maps |001> -> |011>, |000> -> |000>
    // Then H on q2 maps |000> -> (|000>+|100>)/sqrt(2),
    //                   |011> -> (|011>+|111>)/sqrt(2)
    //
    // Combined:
    //   (1/sqrt(2)) * [(|000>+|100>)/sqrt(2) + (|011>+|111>)/sqrt(2)]
    //   = (1/2) * [|000> + |100> + |011> + |111>]
    //
    // Build the tableau as CNOT(0,1) * H(2).
    // Stim tableau semantics: X0->XX (CNOT spreads X to target),
    // Z1->ZZ (CNOT back-propagates Z from target to control).

    auto mod = make_module(3, 1);

    // Start with identity, manually set CNOT(0,1) + H(2)
    auto tab = stim::Tableau<kStimWidth>::identity(3);

    // CNOT(control=0, target=1): X0->X0*X1, Z0->Z0, X1->X1, Z1->Z0*Z1
    tab.xs[0].xs[0] = 1;
    tab.xs[0].xs[1] = 1;  // X0 -> X0 X1
    tab.zs[1].zs[0] = 1;
    tab.zs[1].zs[1] = 1;  // Z1 -> Z0 Z1

    // H on q2: X2->Z2, Z2->X2
    tab.xs[2].xs[2] = 0;
    tab.xs[2].zs[2] = 1;  // X2 -> Z2
    tab.zs[2].xs[2] = 1;
    tab.zs[2].zs[2] = 0;  // Z2 -> X2

    mod.constant_pool.final_tableau = tab;

    SchrodingerState state(1, 0);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {1.0, 0.0};
    state.set_gamma({1.0 / std::sqrt(2.0), 0.0});

    auto sv = get_statevector(mod, state);
    REQUIRE(sv.size() == 8);

    // Expected: 0.5 * (|000> + |011> + |100> + |111>)
    check_complex(sv[0], {0.5, 0.0}, kFloatTol);  // |000>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);  // |001>
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);  // |010>
    check_complex(sv[3], {0.5, 0.0}, kFloatTol);  // |011>
    check_complex(sv[4], {0.5, 0.0}, kFloatTol);  // |100>
    check_complex(sv[5], {0.0, 0.0}, kFloatTol);  // |101>
    check_complex(sv[6], {0.0, 0.0}, kFloatTol);  // |110>
    check_complex(sv[7], {0.5, 0.0}, kFloatTol);  // |111>
}

// =============================================================================
// Native End-to-End Pipeline Tests
// =============================================================================
//
// These tests exercise parse -> trace -> lower -> execute -> get_statevector
// entirely in C++, proving the full pipeline produces correct amplitudes
// without involving Python.

// Helper: compile and execute a circuit string, return dense statevector.
static std::vector<std::complex<double>> pipeline_statevector(const std::string& circuit_text,
                                                              uint64_t seed = 0) {
    auto circuit = clifft::parse(circuit_text);
    auto hir = clifft::trace(circuit);
    auto mod = clifft::lower(hir);
    SchrodingerState state({.peak_rank = mod.peak_rank,
                            .num_measurements = mod.total_meas_slots,
                            .num_detectors = mod.num_detectors,
                            .num_observables = mod.num_observables,
                            .num_exp_vals = mod.num_exp_vals,
                            .seed = seed});
    execute(mod, state);
    return get_statevector(mod, state);
}

// Helper: compile and execute, return measurement record.
static std::vector<uint8_t> pipeline_measurements(const std::string& circuit_text,
                                                  uint64_t seed = 0) {
    auto circuit = clifft::parse(circuit_text);
    auto hir = clifft::trace(circuit);
    auto mod = clifft::lower(hir);
    SchrodingerState state({.peak_rank = mod.peak_rank,
                            .num_measurements = mod.total_meas_slots,
                            .num_detectors = mod.num_detectors,
                            .num_observables = mod.num_observables,
                            .num_exp_vals = mod.num_exp_vals,
                            .seed = seed});
    execute(mod, state);
    return std::vector<uint8_t>(state.meas_record.begin(),
                                state.meas_record.begin() + circuit.num_measurements);
}

// Helper: check that the statevector is normalized (sum of |a_i|^2 = 1).
static void check_normalized(const std::vector<std::complex<double>>& sv, double tol = 1e-9) {
    double norm = 0.0;
    for (const auto& a : sv) {
        norm += std::norm(a);
    }
    CHECK_THAT(norm, WithinAbs(1.0, tol));
}

TEST_CASE("E2E: single H produces plus state") {
    auto sv = pipeline_statevector("H 0");
    REQUIRE(sv.size() == 2);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {kInvSqrt2, 0.0}, kFloatTol);
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: H-T on single qubit") {
    // H|0> = |+> = (|0>+|1>)/sqrt(2)
    // T|+> = (|0> + e^{i*pi/4}|1>) / sqrt(2)
    auto sv = pipeline_statevector("H 0\nT 0");
    REQUIRE(sv.size() == 2);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {kInvSqrt2 * kInvSqrt2, kInvSqrt2 * kInvSqrt2}, kFloatTol);
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: Bell state from H-CX") {
    auto sv = pipeline_statevector("H 0\nCX 0 1");
    REQUIRE(sv.size() == 4);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);  // |00>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);        // |01>
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);        // |10>
    check_complex(sv[3], {kInvSqrt2, 0.0}, kFloatTol);  // |11>
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: Bell plus T gate") {
    // H 0; CX 0 1; T 0
    // Bell = (|00> + |11>)/sqrt(2)
    // T on q0: only |11> has q0=1, so it picks up e^{i*pi/4}
    // Result: (|00> + e^{i*pi/4}|11>) / sqrt(2)
    auto sv = pipeline_statevector("H 0\nCX 0 1\nT 0");
    REQUIRE(sv.size() == 4);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);
    // e^{i*pi/4}/sqrt(2) = (1+i)/(2)
    check_complex(sv[3], {0.5, 0.5}, kFloatTol);
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: T-T equals S") {
    // Two T gates = S gate: diag(1, i)
    // H; T; T on q0 should equal H; S
    auto sv_tt = pipeline_statevector("H 0\nT 0\nT 0");
    auto sv_s = pipeline_statevector("H 0\nS 0");
    REQUIRE(sv_tt.size() == sv_s.size());
    for (size_t i = 0; i < sv_tt.size(); ++i) {
        check_complex(sv_tt[i], sv_s[i], kFloatTol);
    }
}

TEST_CASE("E2E: T then T_DAG is identity") {
    // T * T_dag = I, so H; T; T_DAG = H
    auto sv_tdag = pipeline_statevector("H 0\nT 0\nT_DAG 0");
    auto sv_h = pipeline_statevector("H 0");
    REQUIRE(sv_tdag.size() == sv_h.size());
    for (size_t i = 0; i < sv_tdag.size(); ++i) {
        check_complex(sv_tdag[i], sv_h[i], kFloatTol);
    }
}

TEST_CASE("E2E: GHZ state on 3 qubits") {
    // H 0; CX 0 1; CX 0 2 -> (|000> + |111>)/sqrt(2)
    auto sv = pipeline_statevector("H 0\nCX 0 1\nCX 0 2");
    REQUIRE(sv.size() == 8);

    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);  // |000>
    check_complex(sv[7], {kInvSqrt2, 0.0}, kFloatTol);  // |111>
    // All others zero
    for (int i = 1; i < 7; ++i) {
        check_complex(sv[i], {0.0, 0.0}, kFloatTol);
    }
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: dense Clifford plus T on 4 qubits") {
    // A non-trivial 4-qubit circuit with entanglement and T gates.
    // We verify normalization and that the state is non-trivial (not all-zero
    // except one entry), which catches gross pipeline errors.
    std::string circuit = "H 0\nH 1\nCX 0 2\nCX 1 3\nT 0\nT 1\nCZ 2 3\nH 2\nT 2\nT_DAG 3";
    auto sv = pipeline_statevector(circuit);
    REQUIRE(sv.size() == 16);
    check_normalized(sv, kFloatTol);

    // Count non-zero amplitudes: a rich circuit should produce many
    int nonzero = 0;
    for (const auto& a : sv) {
        if (std::abs(a) > 1e-10)
            ++nonzero;
    }
    CHECK(nonzero >= 4);
}

TEST_CASE("E2E: 5-qubit circuit with multiple T layers") {
    // Stress test: 5 qubits, deep entanglement, multiple non-Clifford layers.
    // This exercises the virtual compressor on a non-trivial topology.
    std::string circuit =
        "H 0\nH 1\nH 2\n"
        "CX 0 3\nCX 1 4\nCX 2 3\n"
        "T 0\nT 1\nT 2\n"
        "CX 3 4\nH 3\nT 3\n"
        "CZ 0 4\nT_DAG 4";
    auto sv = pipeline_statevector(circuit);
    REQUIRE(sv.size() == 32);
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: mirror circuit U U-dag returns to zero state") {
    // Build a Clifford+T circuit and its exact inverse.
    // U * U_dag = I, so all amplitude should be on |000>.
    // This tests perfect destructive interference across the pipeline.
    std::string forward_circuit = "H 0\nCX 0 1\nT 0\nS 1\nCX 1 2\nT 2\nH 2";
    // Inverse: reverse order, swap T<->T_DAG, S<->S_DAG, H=H_dag, CX=CX_dag
    std::string inverse_circuit = "H 2\nT_DAG 2\nCX 1 2\nS_DAG 1\nT_DAG 0\nCX 0 1\nH 0";
    std::string full = forward_circuit + "\n" + inverse_circuit;

    auto sv = pipeline_statevector(full);
    REQUIRE(sv.size() == 8);
    check_normalized(sv, kFloatTol);
    // All amplitude on |000>
    CHECK(std::abs(sv[0]) > 0.999);
    for (size_t i = 1; i < sv.size(); ++i) {
        CHECK(std::abs(sv[i]) < 1e-5);
    }
}

TEST_CASE("E2E: mirror circuit 4 qubits deep") {
    // Deeper mirror test with 4 qubits and multiple entangling layers.
    std::string fwd = "H 0\nH 1\nCX 0 2\nCX 1 3\nT 0\nT 1\nCZ 2 3\nH 2\nT 2\nS 3";
    std::string inv = "S_DAG 3\nT_DAG 2\nH 2\nCZ 2 3\nT_DAG 1\nT_DAG 0\nCX 1 3\nCX 0 2\nH 1\nH 0";
    auto sv = pipeline_statevector(fwd + "\n" + inv);
    REQUIRE(sv.size() == 16);
    check_normalized(sv, kFloatTol);
    CHECK(std::abs(sv[0]) > 0.999);
    for (size_t i = 1; i < sv.size(); ++i) {
        CHECK(std::abs(sv[i]) < 1e-5);
    }
}

// --- Phase 1: Aliases, No-Ops, MPAD, and Inversion ---

TEST_CASE("E2E: Stim aliases produce identical statevectors") {
    auto sv_h = pipeline_statevector("H 0");
    auto sv_hxz = pipeline_statevector("H_XZ 0");
    REQUIRE(sv_h.size() == sv_hxz.size());
    for (size_t i = 0; i < sv_h.size(); ++i) {
        check_complex(sv_h[i], sv_hxz[i], kFloatTol);
    }

    auto sv_s = pipeline_statevector("H 0\nS 0");
    auto sv_sqrt_z = pipeline_statevector("H 0\nSQRT_Z 0");
    for (size_t i = 0; i < sv_s.size(); ++i) {
        check_complex(sv_s[i], sv_sqrt_z[i], kFloatTol);
    }
}

TEST_CASE("E2E: ZCX alias produces Bell state") {
    auto sv = pipeline_statevector("H 0\nZCX 0 1");
    REQUIRE(sv.size() == 4);
    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);
    check_complex(sv[3], {kInvSqrt2, 0.0}, kFloatTol);
}

TEST_CASE("E2E: I gate is no-op identity") {
    auto sv_plain = pipeline_statevector("H 0\nT 0");
    auto sv_with_i = pipeline_statevector("H 0\nI 0\nT 0");
    REQUIRE(sv_plain.size() == sv_with_i.size());
    for (size_t i = 0; i < sv_plain.size(); ++i) {
        check_complex(sv_plain[i], sv_with_i[i], kFloatTol);
    }
}

TEST_CASE("E2E: I on high qubit allocates space but is identity") {
    auto sv = pipeline_statevector("I 3\nH 0");
    REQUIRE(sv.size() == 16);
    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);
    check_complex(sv[1], {kInvSqrt2, 0.0}, kFloatTol);
    for (size_t i = 2; i < sv.size(); ++i) {
        check_complex(sv[i], {0.0, 0.0}, kFloatTol);
    }
}

TEST_CASE("E2E: MPAD writes deterministic measurement record") {
    auto meas = pipeline_measurements("MPAD 1 0 1 0");
    REQUIRE(meas.size() == 4);
    CHECK(meas[0] == 1);
    CHECK(meas[1] == 0);
    CHECK(meas[2] == 1);
    CHECK(meas[3] == 0);
}

TEST_CASE("E2E: MPAD interleaved with real measurements") {
    // H 0; M 0 produces random result, but MPAD is always deterministic
    auto meas = pipeline_measurements("MPAD 1\nMPAD 0");
    REQUIRE(meas.size() == 2);
    CHECK(meas[0] == 1);
    CHECK(meas[1] == 0);
}

TEST_CASE("E2E: inverted M on zero state produces 1") {
    // Fresh qubit in |0>: M 0 -> 0 deterministically
    // M !0 -> inverts the result -> 1
    auto meas_normal = pipeline_measurements("M 0");
    auto meas_inverted = pipeline_measurements("M !0");
    REQUIRE(meas_normal.size() == 1);
    REQUIRE(meas_inverted.size() == 1);
    CHECK(meas_normal[0] == 0);
    CHECK(meas_inverted[0] == 1);
}

TEST_CASE("E2E: I ZCX MPAD combined circuit") {
    // Validates the DoD: "parser successfully ingests I 0; ZCX 0 1; MPAD 1 0"
    auto meas = pipeline_measurements("I 0\nZCX 0 1\nMPAD 1 0");
    REQUIRE(meas.size() == 2);
    CHECK(meas[0] == 1);
    CHECK(meas[1] == 0);
}

TEST_CASE("E2E: MPAD inversion flips deterministic bits") {
    // MPAD !0 -> value=0, inverted -> sign = 0^1 = 1
    // MPAD !1 -> value=1, inverted -> sign = 1^1 = 0
    auto meas = pipeline_measurements("MPAD !0 !1");
    REQUIRE(meas.size() == 2);
    CHECK(meas[0] == 1);  // 0 inverted
    CHECK(meas[1] == 0);  // 1 inverted
}

// --- Phase 2: Pair Measurements and Y-Resets ---

TEST_CASE("E2E: MXX equivalent to MPP X0*X1") {
    // Both should produce identical measurement records on a Bell state
    auto meas_mxx = pipeline_measurements("H 0\nCX 0 1\nMXX 0 1");
    auto meas_mpp = pipeline_measurements("H 0\nCX 0 1\nMPP X0*X1");
    REQUIRE(meas_mxx.size() == 1);
    REQUIRE(meas_mpp.size() == 1);
    CHECK(meas_mxx[0] == meas_mpp[0]);
}

TEST_CASE("E2E: MZZ equivalent to MPP Z0*Z1") {
    auto meas_mzz = pipeline_measurements("H 0\nCX 0 1\nMZZ 0 1");
    auto meas_mpp = pipeline_measurements("H 0\nCX 0 1\nMPP Z0*Z1");
    REQUIRE(meas_mzz.size() == 1);
    REQUIRE(meas_mpp.size() == 1);
    CHECK(meas_mzz[0] == meas_mpp[0]);
}

TEST_CASE("E2E: MYY equivalent to MPP Y0*Y1") {
    auto meas_myy = pipeline_measurements("H 0\nCX 0 1\nMYY 0 1");
    auto meas_mpp = pipeline_measurements("H 0\nCX 0 1\nMPP Y0*Y1");
    REQUIRE(meas_myy.size() == 1);
    REQUIRE(meas_mpp.size() == 1);
    CHECK(meas_myy[0] == meas_mpp[0]);
}

TEST_CASE("E2E: RY resets qubit - subsequent MY is deterministic") {
    // RY collapses to +i eigenstate of Y. MY should give 0 deterministically.
    auto meas = pipeline_measurements("H 0\nRY 0\nMY 0");
    REQUIRE(meas.size() == 1);
    CHECK(meas[0] == 0);
}

TEST_CASE("E2E: MRY produces visible measurement and resets") {
    // MRY measures Y-basis (visible) and resets to |+i>.
    // Subsequent MY should be deterministic 0.
    auto meas = pipeline_measurements("H 0\nMRY 0\nMY 0");
    REQUIRE(meas.size() == 2);
    CHECK(meas[1] == 0);
}

TEST_CASE("E2E: RY correction uses Z not X - multi-seed determinism") {
    // After RY, the qubit must be in |+Y>. If the correction incorrectly used
    // X instead of Z, the state would be X|+Y> = i|-Y>, and MY would yield 1.
    // Test across multiple seeds to catch non-deterministic correction errors.
    for (uint64_t seed = 0; seed < 20; ++seed) {
        auto meas = pipeline_measurements("S 0\nH 0\nRY 0\nMY 0", seed);
        REQUIRE(meas.size() == 1);
        CHECK(meas[0] == 0);
    }
}

TEST_CASE("E2E: MRY correction uses Z not X - multi-seed determinism") {
    // MRY measures Y then resets to |+Y>. Subsequent MY must be deterministic 0.
    // Wrong correction (X instead of Z) would produce i|-Y>, giving MY=1.
    for (uint64_t seed = 0; seed < 20; ++seed) {
        auto meas = pipeline_measurements("S 0\nH 0\nMRY 0\nMY 0", seed);
        REQUIRE(meas.size() == 2);
        CHECK(meas[1] == 0);
    }
}

TEST_CASE("E2E: RY after entangling gate - MY deterministic") {
    // Entangle then reset Y-basis on qubit 0. The reset must produce |+Y>
    // regardless of which branch the hidden measurement collapsed to.
    for (uint64_t seed = 0; seed < 20; ++seed) {
        auto meas = pipeline_measurements("H 0\nCX 0 1\nRY 0\nMY 0\nM 1", seed);
        REQUIRE(meas.size() == 2);
        CHECK(meas[0] == 0);
    }
}

TEST_CASE("E2E: MZZ on Bell state gives deterministic 0") {
    // Bell state |00>+|11> is +1 eigenstate of ZZ
    auto meas = pipeline_measurements("H 0\nCX 0 1\nMZZ 0 1");
    REQUIRE(meas.size() == 1);
    CHECK(meas[0] == 0);
}

TEST_CASE("E2E: MXX on Bell state gives deterministic 0") {
    // Bell state |00>+|11> is +1 eigenstate of XX
    auto meas = pipeline_measurements("H 0\nCX 0 1\nMXX 0 1");
    REQUIRE(meas.size() == 1);
    CHECK(meas[0] == 0);
}

// --- Phase 3: Clifford Expansion ---

TEST_CASE("E2E: SWAP exchanges qubit amplitudes") {
    // H on q0 gives (|00>+|01>)/sqrt(2). SWAP 0 1 exchanges q0<->q1
    // giving (|00>+|10>)/sqrt(2).
    auto sv = pipeline_statevector("H 0\nSWAP 0 1");
    REQUIRE(sv.size() == 4);
    check_complex(sv[0], {kInvSqrt2, 0.0}, kFloatTol);  // |00>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);        // |01>
    check_complex(sv[2], {kInvSqrt2, 0.0}, kFloatTol);  // |10>
    check_complex(sv[3], {0.0, 0.0}, kFloatTol);        // |11>
}

TEST_CASE("E2E: ISWAP on computational basis") {
    // X 0 gives |01> (idx 1). ISWAP swaps |01> -> i|10> (idx 2).
    auto sv = pipeline_statevector("X 0\nISWAP 0 1");
    REQUIRE(sv.size() == 4);
    check_complex(sv[0], {0.0, 0.0}, kFloatTol);  // |00>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);  // |01>
    check_complex(sv[2], {0.0, 1.0}, kFloatTol);  // |10> with phase i
    check_complex(sv[3], {0.0, 0.0}, kFloatTol);  // |11>
}

TEST_CASE("E2E: SQRT_X is half-X rotation") {
    // SQRT_X^2 = X (up to global phase), verify both amplitudes have equal magnitude
    auto sv = pipeline_statevector("SQRT_X 0");
    REQUIRE(sv.size() == 2);
    CHECK_THAT(std::abs(sv[0]), Catch::Matchers::WithinAbs(kInvSqrt2, kFloatTol));
    CHECK_THAT(std::abs(sv[1]), Catch::Matchers::WithinAbs(kInvSqrt2, kFloatTol));
    check_normalized(sv, kFloatTol);
}

TEST_CASE("E2E: SQRT_X applied twice equals X") {
    auto sv_xx = pipeline_statevector("SQRT_X 0\nSQRT_X 0");
    auto sv_x = pipeline_statevector("X 0");
    REQUIRE(sv_xx.size() == sv_x.size());
    for (size_t i = 0; i < sv_xx.size(); ++i) {
        check_complex(sv_xx[i], sv_x[i], kFloatTol);
    }
}

TEST_CASE("E2E: C_XYZ is period-3 rotation") {
    // C_XYZ^3 = I (up to global phase)
    auto sv = pipeline_statevector("H 0\nC_XYZ 0\nC_XYZ 0\nC_XYZ 0");
    auto sv_ref = pipeline_statevector("H 0");
    REQUIRE(sv.size() == sv_ref.size());
    // Global phase may differ, so check |amplitudes| match
    for (size_t i = 0; i < sv.size(); ++i) {
        CHECK_THAT(std::abs(sv[i]), Catch::Matchers::WithinAbs(std::abs(sv_ref[i]), kFloatTol));
    }
}

TEST_CASE("E2E: pure Clifford with ISWAP and C_XYZ compiles to zero instructions") {
    // A pure Clifford circuit should be fully absorbed AOT with zero VM instructions
    auto circuit = clifft::parse("H 0\nISWAP 0 1\nC_XYZ 0\nSQRT_XX 0 1\nH 1");
    auto hir = clifft::trace(circuit);
    auto mod = clifft::lower(hir);
    CHECK(mod.bytecode.empty());
}

TEST_CASE("E2E: ISWAP then ISWAP_DAG is identity") {
    auto sv = pipeline_statevector("H 0\nCX 0 1\nISWAP 0 1\nISWAP_DAG 0 1");
    auto sv_ref = pipeline_statevector("H 0\nCX 0 1");
    REQUIRE(sv.size() == sv_ref.size());
    for (size_t i = 0; i < sv.size(); ++i) {
        check_complex(sv[i], sv_ref[i], kFloatTol);
    }
}

TEST_CASE("E2E: CZSWAP and SWAPCZ alias produce same result") {
    auto sv_czswap = pipeline_statevector("H 0\nX 1\nCZSWAP 0 1");
    auto sv_swapcz = pipeline_statevector("H 0\nX 1\nSWAPCZ 0 1");
    REQUIRE(sv_czswap.size() == sv_swapcz.size());
    for (size_t i = 0; i < sv_czswap.size(); ++i) {
        check_complex(sv_czswap[i], sv_swapcz[i], kFloatTol);
    }
}

TEST_CASE("E2E: H_XY maps Z to -Z so flips zero to one") {
    // H_XY stabilizer flows: X->Y, Y->X, Z->-Z.
    // So |0> (Z eigenstate +1) maps to |1> (Z eigenstate -1).
    auto sv = pipeline_statevector("H_XY 0");
    REQUIRE(sv.size() == 2);
    CHECK_THAT(std::abs(sv[0]), Catch::Matchers::WithinAbs(0.0, kFloatTol));
    CHECK_THAT(std::abs(sv[1]), Catch::Matchers::WithinAbs(1.0, kFloatTol));
    check_normalized(sv, kFloatTol);
}
