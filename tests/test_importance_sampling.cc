// Tests for importance sampling: DP table, subset sampling, and sample_k API.

#include "clifft/api/reference_syndrome.h"
#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Compile a stim circuit string with normalized syndromes.
clifft::CompiledModule compile_circuit(const std::string& stim_text) {
    auto circuit = clifft::parse(stim_text);
    auto hir = clifft::trace(circuit);
    auto ref = clifft::compute_reference_syndrome(hir);
    return clifft::lower(hir, {}, ref.detectors, ref.observables);
}

}  // namespace

// =============================================================================
// noise_site_probabilities
// =============================================================================

TEST_CASE("noise_site_probabilities - basic extraction") {
    // Circuit with one DEPOLARIZE1(0.03) on qubit 0 and X_ERROR(0.01) on qubit 1,
    // plus readout noise on qubit 0.
    std::string circuit_text = R"(
        R 0 1
        DEPOLARIZE1(0.03) 0
        X_ERROR(0.01) 1
        M(0.005) 0
        M 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);

    // Should have 2 quantum noise sites + 1 readout noise entry = 3 total.
    REQUIRE(probs.size() == 3);

    // DEPOLARIZE1(0.03): total prob = 0.03
    CHECK_THAT(probs[0], WithinAbs(0.03, 1e-12));
    // X_ERROR(0.01): total prob = 0.01
    CHECK_THAT(probs[1], WithinAbs(0.01, 1e-12));
    // Readout noise: 0.005
    CHECK_THAT(probs[2], WithinAbs(0.005, 1e-12));
}

TEST_CASE("noise_site_probabilities - no noise") {
    std::string circuit_text = R"(
        R 0
        H 0
        M 0
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    CHECK(probs.empty());
}

// =============================================================================
// sample_k with k=0 - zero faults means zero errors
// =============================================================================

TEST_CASE("sample_k - k=0 produces no errors") {
    // Repetition code distance 3: needs at least 2 faults to flip observable.
    std::string circuit_text = R"(
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        M 0 1 2
        DETECTOR rec[-1] rec[-2]
        DETECTOR rec[-2] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto result = clifft::sample_k(prog, 1000, 0, 42);

    // With k=0 forced faults, no noise fires at all.
    // All observables should be 0, all detectors should be 0.
    uint32_t num_obs = prog.num_observables;
    uint32_t num_det = prog.num_detectors;
    for (uint32_t shot = 0; shot < 1000; ++shot) {
        for (uint32_t i = 0; i < num_obs; ++i) {
            CHECK(result.observables[shot * num_obs + i] == 0);
        }
        for (uint32_t i = 0; i < num_det; ++i) {
            CHECK(result.detectors[shot * num_det + i] == 0);
        }
    }
}

// =============================================================================
// sample_k_survivors - k=0 yields zero logical errors
// =============================================================================

TEST_CASE("sample_k_survivors - k=0 no errors") {
    std::string circuit_text = R"(
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        M 0 1 2
        DETECTOR rec[-1] rec[-2]
        DETECTOR rec[-2] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto result = clifft::sample_k_survivors(prog, 5000, 0, 42);

    CHECK(result.total_shots == 5000);
    CHECK(result.passed_shots == 5000);
    CHECK(result.logical_errors == 0);
}

// =============================================================================
// sample_k - all faults forced means every site fires
// =============================================================================

TEST_CASE("sample_k - k=N forces all sites") {
    // Single qubit with X_ERROR(0.5). k=1 forces the single noise site to fire.
    // Every shot should flip the qubit.
    std::string circuit_text = R"(
        R 0
        X_ERROR(0.5) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    uint32_t n_total = static_cast<uint32_t>(probs.size());
    REQUIRE(n_total == 1);

    auto result = clifft::sample_k(prog, 500, 1, 42);

    // Every shot should have the observable flipped.
    uint32_t flips = 0;
    for (uint32_t shot = 0; shot < 500; ++shot) {
        if (result.observables[shot] != 0)
            flips++;
    }
    CHECK(flips == 500);
}

// =============================================================================
// sample_k - invalid k throws
// =============================================================================

TEST_CASE("sample_k - k exceeds total sites throws") {
    std::string circuit_text = R"(
        R 0
        X_ERROR(0.1) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    uint32_t n_total = static_cast<uint32_t>(probs.size());

    CHECK_THROWS_AS(clifft::sample_k(prog, 10, n_total + 1, 42), std::invalid_argument);
    CHECK_THROWS_AS(clifft::sample_k_survivors(prog, 10, n_total + 1, 42), std::invalid_argument);
}

TEST_CASE("sample_k - zero probability site rejects impossible strata") {
    // X_ERROR(0.0) creates a noise site with p=0 (empty channel list).
    // k=1 is impossible because the only site can never fire.
    std::string circuit_text = R"(
        R 0
        X_ERROR(0.0) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    REQUIRE(probs.size() == 1);
    CHECK(probs[0] == 0.0);

    // k=0 is fine: zero faults on a zero-probability site is valid.
    CHECK_NOTHROW(clifft::sample_k(prog, 10, 0, 42));
    // k=1 is impossible: zero-mass stratum (the only site has p=0).
    CHECK_THROWS_AS(clifft::sample_k(prog, 10, 1, 42), std::invalid_argument);
    CHECK_THROWS_AS(clifft::sample_k_survivors(prog, 10, 1, 42), std::invalid_argument);
}

TEST_CASE("sample_k - noiseless circuit rejects k greater than 0") {
    // Truly noiseless circuit: no noise instructions at all.
    std::string circuit_text = R"(
        R 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    CHECK(probs.empty());

    CHECK_NOTHROW(clifft::sample_k(prog, 10, 0, 42));
    CHECK_THROWS_AS(clifft::sample_k(prog, 10, 1, 42), std::invalid_argument);
}

// =============================================================================
// Forced faults with readout noise
// =============================================================================

TEST_CASE("sample_k - readout noise forcing") {
    // No quantum noise, only readout noise on qubit 0.
    // k=1 should force the readout flip every time.
    std::string circuit_text = R"(
        R 0
        M(0.1) 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    REQUIRE(probs.size() == 1);  // One readout noise entry, no quantum noise

    auto result = clifft::sample_k(prog, 500, 1, 42);

    // Every shot should have the observable flipped (readout noise forced).
    uint32_t flips = 0;
    for (uint32_t shot = 0; shot < 500; ++shot) {
        if (result.observables[shot] != 0)
            flips++;
    }
    CHECK(flips == 500);
}

// =============================================================================
// Uniform mode detection and correctness
// =============================================================================

TEST_CASE("sample_k - uniform probability uses Fisher-Yates path") {
    // All X_ERROR(0.01) on 5 qubits: uniform probabilities.
    std::string circuit_text = R"(
        R 0 1 2 3 4
        X_ERROR(0.01) 0 1 2 3 4
        M 0 1 2 3 4
        DETECTOR rec[-1] rec[-2]
        DETECTOR rec[-2] rec[-3]
        DETECTOR rec[-3] rec[-4]
        DETECTOR rec[-4] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    REQUIRE(probs.size() == 5);

    // With k=2 forced faults on 5 uniform sites, exactly 2 detectors
    // should fire per shot (give or take adjacency effects).
    // Main check: it runs without crashing and produces reasonable results.
    auto result = clifft::sample_k(prog, 1000, 2, 42);

    // Every shot should have exactly 2 faults. Since X_ERROR flips qubits,
    // we can check that exactly 2 measurements are flipped per shot.
    for (uint32_t shot = 0; shot < 1000; ++shot) {
        uint32_t flips = 0;
        for (uint32_t i = 0; i < 5; ++i) {
            if (result.measurements[shot * 5 + i] != 0)
                flips++;
        }
        CHECK(flips == 2);
    }
}

// =============================================================================
// Non-uniform mode: verify exactly k faults fire
// =============================================================================

TEST_CASE("sample_k - non-uniform probabilities fire exactly k sites") {
    // Mix of different error rates to trigger non-uniform DP path.
    std::string circuit_text = R"(
        R 0 1 2
        X_ERROR(0.01) 0
        X_ERROR(0.05) 1
        X_ERROR(0.1) 2
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);
    auto probs = clifft::noise_site_probabilities(prog);
    REQUIRE(probs.size() == 3);

    for (uint32_t k = 0; k <= 3; ++k) {
        auto result = clifft::sample_k(prog, 500, k, 42 + k);
        for (uint32_t shot = 0; shot < 500; ++shot) {
            uint32_t flips = 0;
            for (uint32_t i = 0; i < 3; ++i) {
                if (result.measurements[shot * 3 + i] != 0)
                    flips++;
            }
            CHECK(flips == k);
        }
    }
}

// =============================================================================
// Non-uniform subset distribution: sites with higher probability should be
// selected more often
// =============================================================================

TEST_CASE("sample_k - non-uniform favors higher probability sites") {
    // 3 sites with very different error rates. k=1 should strongly prefer site 2.
    std::string circuit_text = R"(
        R 0 1 2
        X_ERROR(0.001) 0
        X_ERROR(0.01) 1
        X_ERROR(0.5) 2
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);

    auto result = clifft::sample_k(prog, 10000, 1, 42);

    // Count which site was selected in each shot.
    uint32_t count[3] = {0, 0, 0};
    for (uint32_t shot = 0; shot < 10000; ++shot) {
        for (uint32_t i = 0; i < 3; ++i) {
            if (result.measurements[shot * 3 + i] != 0)
                count[i]++;
        }
    }

    // Odds ratios: w0=0.001/0.999, w1=0.01/0.99, w2=0.5/0.5=1.0
    // Site 2 should dominate. Check count[2] >> count[1] >> count[0].
    CHECK(count[2] > count[1]);
    CHECK(count[1] > count[0]);
    CHECK(count[2] > 5000);  // Site 2 should appear in most shots
}

// =============================================================================
// sample_k_survivors with postselection
// =============================================================================

TEST_CASE("sample_k_survivors - postselection discards some shots") {
    // Circuit with postselection: some k-fault patterns will be discarded.
    std::string circuit_text = R"(
        R 0 1
        X_ERROR(0.1) 0 1
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto circuit = clifft::parse(circuit_text);
    auto hir = clifft::trace(circuit);
    auto ref = clifft::compute_reference_syndrome(hir);

    // Postselect on first detector: shots where qubit 0 flips get discarded.
    std::vector<uint8_t> ps_mask = {1, 0};
    auto prog = clifft::lower(hir, ps_mask, ref.detectors, ref.observables);

    // k=1: one of two sites fires. When site 0 fires, detector 0 triggers
    // and the shot is postselected out. When site 1 fires, it passes.
    auto result = clifft::sample_k_survivors(prog, 1000, 1, 42);

    CHECK(result.total_shots == 1000);
    // Roughly half should be discarded (site 0 vs site 1 equal probability).
    CHECK(result.passed_shots > 200);
    CHECK(result.passed_shots < 800);
}

// =============================================================================
// Deterministic reproducibility with seed
// =============================================================================

TEST_CASE("sample_k - deterministic with same seed") {
    std::string circuit_text = R"(
        R 0 1 2
        DEPOLARIZE1(0.05) 0 1 2
        M 0 1 2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )";
    auto prog = compile_circuit(circuit_text);

    auto r1 = clifft::sample_k(prog, 100, 2, 12345);
    auto r2 = clifft::sample_k(prog, 100, 2, 12345);

    CHECK(r1.measurements == r2.measurements);
    CHECK(r1.detectors == r2.detectors);
    CHECK(r1.observables == r2.observables);
}
