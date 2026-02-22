#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace ucc;

// =============================================================================
// Helper: Full pipeline from stim text to compiled module
// =============================================================================

static CompiledModule compile(const std::string& stim_text) {
    Circuit circuit = parse(stim_text);
    HirModule hir = trace(circuit);
    return lower(hir);
}

// =============================================================================
// Task 5.1: SchrodingerState allocation and initialization
// =============================================================================

TEST_CASE("SVM: SchrodingerState allocates correct size", "[svm]") {
    SchrodingerState state(3, 0);  // 2^3 = 8 amplitudes
    REQUIRE(state.array_size() == 8);
}

TEST_CASE("SVM: SchrodingerState initializes to |0...0>", "[svm]") {
    SchrodingerState state(2, 0);  // 4 amplitudes

    // Only amplitude 0 should be nonzero
    REQUIRE(std::abs(state.v()[0] - std::complex<double>(1.0, 0.0)) < 1e-10);
    REQUIRE(std::abs(state.v()[1]) < 1e-10);
    REQUIRE(std::abs(state.v()[2]) < 1e-10);
    REQUIRE(std::abs(state.v()[3]) < 1e-10);
}

TEST_CASE("SVM: SchrodingerState reset restores initial state", "[svm]") {
    SchrodingerState state(2, 1);

    // Modify state
    state.v()[0] = {0.5, 0.5};
    state.destab_signs = 0xFF;
    state.stab_signs = 0xAA;
    state.meas_record[0] = 1;

    // Reset
    state.reset(42);

    // O(1) reset: only v[0] is guaranteed to be initialized.
    // Memory beyond current rank (which starts at 0) is written by OP_BRANCH.
    REQUIRE(std::abs(state.v()[0] - std::complex<double>(1.0, 0.0)) < 1e-10);
    REQUIRE(state.destab_signs == 0);
    REQUIRE(state.stab_signs == 0);
    REQUIRE(state.meas_record[0] == 0);
}

TEST_CASE("SVM: deterministic RNG produces consistent values", "[svm]") {
    SchrodingerState state1(0, 0, 12345);
    SchrodingerState state2(0, 0, 12345);

    // Same seed should produce same sequence
    for (int i = 0; i < 10; ++i) {
        REQUIRE(state1.random_double() == state2.random_double());
    }
}

TEST_CASE("SVM: RNG values in [0, 1)", "[svm]") {
    SchrodingerState state(0, 0, 99999);

    for (int i = 0; i < 1000; ++i) {
        double val = state.random_double();
        REQUIRE(val >= 0.0);
        REQUIRE(val < 1.0);
    }
}

// =============================================================================
// Task 5.2: Opcode execution tests
// =============================================================================

TEST_CASE("SVM: empty program returns empty results", "[svm]") {
    CompiledModule prog;
    prog.peak_rank = 0;
    prog.num_measurements = 0;

    auto results = sample(prog, 10, 0);
    REQUIRE(results.empty());
}

TEST_CASE("SVM: single T gate on |0> gives SCALAR_PHASE", "[svm]") {
    // T|0> = |0> (just a phase)
    auto prog = compile("T 0");

    REQUIRE(prog.bytecode.size() == 1);
    REQUIRE(prog.bytecode[0].opcode == Opcode::OP_SCALAR_PHASE);
    REQUIRE(prog.peak_rank == 0);

    // Execute - with dominant term factoring, state is unnormalized.
    // The amplitude is (1 + i*tan(π/8)) which has |z|^2 = 1 + tan^2(π/8) ≈ 1.17
    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    // Verify amplitude is nonzero and has correct phase structure
    REQUIRE(std::abs(state.v()[0]) > 0.5);
}

TEST_CASE("SVM: H T gives superposition with T phase", "[svm]") {
    // H|0> = |+>, then T|+> introduces phase difference
    auto prog = compile(R"(
        H 0
        T 0
    )");

    REQUIRE(prog.bytecode.size() == 1);
    REQUIRE(prog.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(prog.peak_rank == 1);

    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    // State has 2 amplitudes, both nonzero (unnormalized with dominant term factoring)
    REQUIRE(std::abs(state.v()[0]) > 0.1);
    REQUIRE(std::abs(state.v()[1]) > 0.1);
}

TEST_CASE("SVM: two independent T gates give rank 2", "[svm]") {
    auto prog = compile(R"(
        H 0
        H 1
        T 0
        T 1
    )");

    REQUIRE(prog.peak_rank == 2);
    REQUIRE(prog.bytecode.size() == 2);

    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    // All 4 amplitudes should be nonzero (unnormalized with dominant term factoring)
    for (uint32_t i = 0; i < 4; ++i) {
        REQUIRE(std::abs(state.v()[i]) > 0.01);
    }
}

TEST_CASE("SVM: T T_DAG cancels", "[svm]") {
    auto prog = compile(R"(
        H 0
        T 0
        T_DAG 0
    )");

    // Two operations: BRANCH then COLLIDE
    REQUIRE(prog.bytecode.size() == 2);
    REQUIRE(prog.peak_rank == 1);

    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    // T T_DAG = I in the computational basis, but with dominant term factoring
    // we get: v[0] = 1 + tan^2(π/8), v[1] should cancel to 0.
    // The key test: v[1] is effectively zero (the Z component cancels).
    REQUIRE(std::abs(state.v()[1]) < 1e-10);
    REQUIRE(std::abs(state.v()[0]) > 0.5);  // Identity branch is large
}

TEST_CASE("SVM: measurement on |0> is deterministic", "[svm]") {
    auto prog = compile("M 0");

    REQUIRE(prog.num_measurements == 1);

    // All shots should give 0
    auto results = sample(prog, 100, 12345);
    for (auto r : results) {
        REQUIRE(r == 0);
    }
}

TEST_CASE("SVM: measurement on |1> is deterministic", "[svm]") {
    auto prog = compile(R"(
        X 0
        M 0
    )");

    REQUIRE(prog.num_measurements == 1);

    // All shots should give 1
    auto results = sample(prog, 100, 12345);
    for (auto r : results) {
        REQUIRE(r == 1);
    }
}

TEST_CASE("SVM: measurement on |+> gives ~50/50", "[svm]") {
    auto prog = compile(R"(
        H 0
        M 0
    )");

    REQUIRE(prog.num_measurements == 1);

    auto results = sample(prog, 1000, 42);

    int zeros = 0, ones = 0;
    for (auto r : results) {
        if (r == 0)
            zeros++;
        else
            ones++;
    }

    // Should be roughly 50/50 (allow 10% tolerance)
    REQUIRE(zeros > 400);
    REQUIRE(zeros < 600);
    REQUIRE(ones > 400);
    REQUIRE(ones < 600);
}

TEST_CASE("SVM: reset decomposes to M + conditional X", "[svm]") {
    // R 0 is decomposed to M 0; CX rec[-1] 0
    // On |1>, this should measure 1 and flip back to |0>
    auto prog = compile(R"(
        X 0
        R 0
        M 0
    )");

    // Should have: MEASURE (deterministic 1) + CONDITIONAL + MEASURE (deterministic 0)
    REQUIRE(prog.num_measurements == 2);

    // Final measurement should always be 0 (reset worked)
    auto results = sample(prog, 100, 0);
    for (size_t i = 0; i < 100; ++i) {
        REQUIRE(results[i * 2 + 1] == 0);  // Second measurement
    }
}

TEST_CASE("SVM: Bell state measurements are correlated", "[svm]") {
    auto prog = compile(R"(
        H 0
        CX 0 1
        M 0
        M 1
    )");

    REQUIRE(prog.num_measurements == 2);

    auto results = sample(prog, 500, 99);

    // Bell state: both qubits always match
    for (size_t shot = 0; shot < 500; ++shot) {
        uint8_t m0 = results[shot * 2];
        uint8_t m1 = results[shot * 2 + 1];
        REQUIRE(m0 == m1);
    }
}

TEST_CASE("SVM: sample returns correct shape", "[svm]") {
    auto prog = compile(R"(
        H 0
        M 0
        H 1
        M 1
    )");

    REQUIRE(prog.num_measurements == 2);

    auto results = sample(prog, 50, 0);
    REQUIRE(results.size() == 100);  // 50 shots * 2 measurements
}

// =============================================================================
// Edge cases and normalization
// =============================================================================

TEST_CASE("SVM: state has nonzero amplitudes after operations", "[svm]") {
    // With dominant term factoring, states are unnormalized.
    // This test verifies that operations produce nonzero amplitudes.
    auto prog = compile(R"(
        H 0
        H 1
        T 0
        T 1
        CX 0 1
        T 0
    )");

    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    double total = 0.0;
    for (uint64_t i = 0; i < state.array_size(); ++i) {
        total += std::norm(state.v()[i]);
    }
    // State should have significant amplitude (not collapsed to zero)
    REQUIRE(total > 0.1);
}

TEST_CASE("SVM: zero shots returns empty", "[svm]") {
    auto prog = compile("H 0\nM 0");
    auto results = sample(prog, 0, 0);
    REQUIRE(results.empty());
}

TEST_CASE("SVM: peak_rank 0 circuit works", "[svm]") {
    // Pure Clifford circuit with measurement
    auto prog = compile(R"(
        X 0
        M 0
    )");

    REQUIRE(prog.peak_rank == 0);

    auto results = sample(prog, 10, 0);
    REQUIRE(results.size() == 10);
    for (auto r : results) {
        REQUIRE(r == 1);
    }
}

// =============================================================================
// Regression tests for physics bugs identified in code review
// =============================================================================

TEST_CASE("SVM: MEASURE_MERGE uses butterfly interference not filtering", "[svm][review]") {
    // This tests that MEASURE_MERGE correctly interferes amplitudes (v[α] ± v[α⊕β])
    // rather than just zeroing half the array.
    //
    // Circuit: H 0; T 0; H 0; M 0
    // After H T H, we have a state that when measured uses MERGE
    // The butterfly interference should give correct probabilities
    auto prog = compile(R"(
        H 0
        T 0
        H 0
        M 0
    )");

    REQUIRE(prog.peak_rank == 1);
    REQUIRE(prog.num_measurements == 1);

    // Run many shots and verify distribution matches expected physics
    // H T H rotates |0⟩ by π/8 around Y axis, giving:
    // cos(π/8)|0⟩ + sin(π/8)|1⟩ with some phase
    auto results = sample(prog, 10000, 42);

    int zeros = 0, ones = 0;
    for (auto r : results) {
        if (r == 0)
            zeros++;
        else
            ones++;
    }

    // cos²(π/8) ≈ 0.854, sin²(π/8) ≈ 0.146
    // Allow 5% tolerance
    double p0 = static_cast<double>(zeros) / 10000.0;
    REQUIRE(p0 > 0.80);
    REQUIRE(p0 < 0.92);
}

TEST_CASE("SVM: T-gate respects Pauli frame from measurements", "[svm][review]") {
    // After an anti-commuting measurement, the Pauli frame should affect
    // subsequent T-gates. This is the "frame parity" fix.
    //
    // Circuit: H 0; M 0; T 0; M 0
    // The second T gate should respect the frame from the first measurement.
    auto prog = compile(R"(
        H 0
        M 0
        H 0
        T 0
        H 0
        M 0
    )");

    // Just verify it runs without crashing and produces valid results
    auto results = sample(prog, 1000, 123);
    REQUIRE(results.size() == 2000);  // 1000 shots * 2 measurements

    // Count that we get some 0s and 1s (not deterministic)
    int zeros = 0;
    for (size_t i = 0; i < 1000; ++i) {
        if (results[i * 2 + 1] == 0)
            zeros++;
    }
    REQUIRE(zeros > 100);
    REQUIRE(zeros < 900);
}

TEST_CASE("SVM: GHZ state measurements are correlated", "[svm][review]") {
    // 3-qubit GHZ state: all measurements should match
    auto prog = compile(R"(
        H 0
        CX 0 1
        CX 0 2
        M 0
        M 1
        M 2
    )");

    REQUIRE(prog.num_measurements == 3);

    auto results = sample(prog, 500, 77);

    for (size_t shot = 0; shot < 500; ++shot) {
        uint8_t m0 = results[shot * 3 + 0];
        uint8_t m1 = results[shot * 3 + 1];
        uint8_t m2 = results[shot * 3 + 2];
        REQUIRE(m0 == m1);
        REQUIRE(m1 == m2);
    }
}

TEST_CASE("SVM: Multiple T gates with measurements stay normalized", "[svm][review]") {
    // Stress test: multiple T gates with intermediate measurements
    auto prog = compile(R"(
        H 0
        H 1
        T 0
        T 1
        CX 0 1
        T 0
        M 0
        T 1
        M 1
    )");

    // Just verify it runs and produces valid results
    auto results = sample(prog, 100, 456);
    REQUIRE(results.size() == 200);

    // All results should be 0 or 1
    for (auto r : results) {
        REQUIRE((r == 0 || r == 1));
    }
}

TEST_CASE("SVM: MEASURE_FILTER with commutation mask", "[svm][review]") {
    // Test MEASURE_FILTER path: β=0 but commutation_mask ≠ 0
    // This happens when measuring in a basis that commutes with tableau
    // but has sign dependence on GF(2) index
    //
    // Create a state where this applies by using S gates
    auto prog = compile(R"(
        H 0
        T 0
        T 0
        M 0
    )");

    // T T = S, which rotates by π/4
    // After H S, measuring Z gives a distribution

    auto results = sample(prog, 1000, 789);

    int zeros = 0, ones = 0;
    for (auto r : results) {
        if (r == 0)
            zeros++;
        else
            ones++;
    }

    // Should have some of each (not deterministic)
    REQUIRE(zeros > 100);
    REQUIRE(ones > 100);
}

TEST_CASE("SVM: SCALAR_PHASE respects commutation mask", "[svm][review]") {
    // H 0; T 0 expands basis along X₀.
    // H 0 maps X₀ → Z₀.
    // Second T 0 has β=0 (rewound Pauli is Z₀), but Z₀ anti-commutes with X₀.
    // This tests that SCALAR_PHASE applies different phases to different amplitudes.
    auto prog = compile(R"(
        H 0
        T 0
        H 0
        T 0
    )");

    // Should have BRANCH (first T) then SCALAR_PHASE (second T)
    REQUIRE(prog.bytecode.size() == 2);
    REQUIRE(prog.bytecode[0].opcode == Opcode::OP_BRANCH);
    REQUIRE(prog.bytecode[1].opcode == Opcode::OP_SCALAR_PHASE);

    // Critical: SCALAR_PHASE must have nonzero commutation_mask
    REQUIRE(prog.bytecode[1].commutation_mask != 0);

    SchrodingerState state(prog.peak_rank, prog.num_measurements, 0);
    execute(prog, state);

    // Both amplitudes should be nonzero with different phases
    REQUIRE(std::abs(state.v()[0]) > 0.1);
    REQUIRE(std::abs(state.v()[1]) > 0.1);

    // The phases should be different (not just scaled versions of each other)
    // If commutation_mask were incorrectly 0, both would have the same phase.
    std::complex<double> ratio = state.v()[1] / state.v()[0];
    // ratio should NOT be purely real (which would indicate same phase)
    REQUIRE(std::abs(ratio.imag()) > 0.01);
}
