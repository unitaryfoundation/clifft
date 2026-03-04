#include "ucc/backend/backend.h"
#include "ucc/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <vector>

using namespace ucc;
using Catch::Matchers::WithinAbs;

constexpr double kTol = 1e-9;

// Tolerance for float-precision tableau multiplication.
// Stim's to_flat_unitary_matrix returns complex<float> (~7 decimal digits).
constexpr double kFloatTol = 1e-6;

static void check_complex(std::complex<double> actual, std::complex<double> expected,
                          double tol = kTol) {
    CHECK_THAT(actual.real(), WithinAbs(expected.real(), tol));
    CHECK_THAT(actual.imag(), WithinAbs(expected.imag(), tol));
}

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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    constexpr double s = 0.70710678118654752440;  // 1/sqrt(2)
    check_complex(sv[0], {s, 0.0});
    check_complex(sv[1], {s, 0.0});
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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 4);
    constexpr double s = 0.70710678118654752440;
    check_complex(sv[0], {s, 0.0});
    check_complex(sv[1], {0.0, 0.0});
    check_complex(sv[2], {0.0, 0.0});
    check_complex(sv[3], {s, 0.0});
}

// =============================================================================
// Pauli Frame Tests
// =============================================================================

TEST_CASE("Statevector: X frame flips the zero state") {
    // |0> with p_x[0]=1 -> P|0> = X|0> = |1>
    auto mod = make_module(1, 1);
    SchrodingerState state(1, 0);
    state.p_x = stim::bitword<kStimWidth>(uint64_t{1});  // X on qubit 0

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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};
    state.p_z = stim::bitword<kStimWidth>(uint64_t{1});  // Z on qubit 0

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    constexpr double s = 0.70710678118654752440;
    // Z|+> = (Z|0> + Z|1>)/sqrt(2) = (|0> - |1>)/sqrt(2)
    check_complex(sv[0], {s, 0.0});
    check_complex(sv[1], {-s, 0.0});
}

TEST_CASE("Statevector: XZ frame on 2-qubit state") {
    // 2 qubits, |00> state, p_x = 0b01 (X on q0), p_z = 0b10 (Z on q1)
    // P = X0 Z1, applied to |00>:
    // X0 Z1 |00> = X0 |00> = |01> (bit 0 flipped)
    // Z1 on |00> is trivial (eigenvalue +1 since bit 1 = 0)
    auto mod = make_module(2, 2);
    SchrodingerState state(2, 0);
    state.p_x = stim::bitword<kStimWidth>(uint64_t{1});  // X on qubit 0
    state.p_z = stim::bitword<kStimWidth>(uint64_t{2});  // Z on qubit 1

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
    state.gamma = {0.0, 1.0};  // gamma = i

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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 8);
    constexpr double s = 0.70710678118654752440;
    check_complex(sv[0], {s, 0.0});    // |000>
    check_complex(sv[1], {s, 0.0});    // |001>
    check_complex(sv[2], {0.0, 0.0});  // |010>
    check_complex(sv[3], {0.0, 0.0});  // |011>
    check_complex(sv[4], {0.0, 0.0});  // |100>
    check_complex(sv[5], {0.0, 0.0});  // |101>
    check_complex(sv[6], {0.0, 0.0});  // |110>
    check_complex(sv[7], {0.0, 0.0});  // |111>
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
    constexpr double s = 0.70710678118654752440;
    check_complex(sv[0], {s, 0.0}, kFloatTol);
    check_complex(sv[1], {s, 0.0}, kFloatTol);
}

TEST_CASE("Statevector: Hadamard tableau on one-state produces minus") {
    // U_C = H. Virtual state = |1> (via X frame on |0>). Physical = H*X|0> = H|1> = |->.
    auto mod = make_module(1, 1);
    mod.constant_pool.final_tableau = stim::Tableau<kStimWidth>::gate1("+Z", "+X");
    SchrodingerState state(1, 0);
    state.p_x = stim::bitword<kStimWidth>(uint64_t{1});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    constexpr double s = 0.70710678118654752440;
    // H|1> = (|0> - |1>)/sqrt(2)
    check_complex(sv[0], {s, 0.0}, kFloatTol);
    check_complex(sv[1], {-s, 0.0}, kFloatTol);
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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 4);
    constexpr double s = 0.70710678118654752440;
    check_complex(sv[0], {s, 0.0}, kFloatTol);    // |00>
    check_complex(sv[1], {0.0, 0.0}, kFloatTol);  // |01>
    check_complex(sv[2], {0.0, 0.0}, kFloatTol);  // |10>
    check_complex(sv[3], {s, 0.0}, kFloatTol);    // |11>
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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    constexpr double s = 0.70710678118654752440;
    check_complex(sv[0], {s, 0.0}, kFloatTol);
    check_complex(sv[1], {0.0, s}, kFloatTol);  // i/sqrt(2)
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
    state.p_z = stim::bitword<kStimWidth>(uint64_t{1});

    auto sv = get_statevector(mod, state);

    REQUIRE(sv.size() == 2);
    constexpr double s = 0.70710678118654752440;
    // Z|0> = |0>, then H|0> = |+>
    check_complex(sv[0], {s, 0.0}, kFloatTol);
    check_complex(sv[1], {s, 0.0}, kFloatTol);
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
    state.gamma = {1.0 / std::sqrt(2.0), 0.0};
    state.p_z = stim::bitword<kStimWidth>(uint64_t{1});

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
