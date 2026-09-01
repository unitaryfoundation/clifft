#pragma once

// Shared helpers for tests that reorder HIR operations and must confirm the
// reordered program samples the same distribution as the original: a random
// noisy-circuit generator plus a statistical equivalence check. Split out of
// test_logical_noise_prefix.cc so test_schedule_dependence.cc can reuse both
// without duplicating them.

#include "clifft/frontend/hir.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace clifft::test {

// ---------------------------------------------------------------------------
// Random noisy circuit generation
// ---------------------------------------------------------------------------

// A generated circuit plus, for each NOISE line, enough information to
// realize a fixed firing outcome for it later (used by legality-oracle style
// tests that replay a fixed noise realization).
struct GeneratedCircuit {
    struct NoiseLine {
        size_t line_index = 0;
        uint32_t qubit = 0;
        bool is_depolarize1 = false;
        double prob = 0.0;
    };

    std::vector<std::string> lines;
    std::vector<NoiseLine> noise_lines;
};

inline double next_unit(std::mt19937& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

// Deterministic generator over a small gate set: T, T_DAG, R_Z at a few
// angles, M, MX, MR, R, X_ERROR, DEPOLARIZE1, and DETECTORs referencing
// earlier records. Every generated line is individually valid, so the
// circuit as a whole always parses and traces.
inline GeneratedCircuit generate_noisy_circuit(std::mt19937& rng, uint32_t num_qubits,
                                               uint32_t num_ops) {
    static const double kAngles[] = {0.125, 0.25, 0.375, 0.625, 0.75, 0.875};
    static const double kNoiseProbs[] = {0.05, 0.1, 0.2, 0.3};

    GeneratedCircuit circuit;
    uint32_t measurement_count = 0;
    for (uint32_t op = 0; op < num_ops; ++op) {
        const uint32_t q = rng() % num_qubits;
        switch (rng() % 9) {
            case 0:
                circuit.lines.push_back("T " + std::to_string(q));
                break;
            case 1:
                circuit.lines.push_back("T_DAG " + std::to_string(q));
                break;
            case 2:
                circuit.lines.push_back("R_Z(" +
                                        std::to_string(kAngles[rng() % std::size(kAngles)]) + ") " +
                                        std::to_string(q));
                break;
            case 3:
                circuit.lines.push_back("M " + std::to_string(q));
                ++measurement_count;
                break;
            case 4:
                circuit.lines.push_back("MX " + std::to_string(q));
                ++measurement_count;
                break;
            case 5:
                circuit.lines.push_back("MR " + std::to_string(q));
                ++measurement_count;
                break;
            case 6:
                circuit.lines.push_back("R " + std::to_string(q));
                break;
            case 7: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("X_ERROR(" + std::to_string(prob) + ") " +
                                        std::to_string(q));
                circuit.noise_lines.push_back(
                    {circuit.lines.size() - 1, q, /*is_depolarize1=*/false, prob});
                break;
            }
            case 8: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("DEPOLARIZE1(" + std::to_string(prob) + ") " +
                                        std::to_string(q));
                circuit.noise_lines.push_back(
                    {circuit.lines.size() - 1, q, /*is_depolarize1=*/true, prob});
                break;
            }
            default:
                break;
        }
        if (measurement_count > 0 && (rng() % 5) == 0) {
            const uint32_t back = 1 + (rng() % measurement_count);
            circuit.lines.push_back("DETECTOR rec[-" + std::to_string(back) + "]");
        }
    }
    return circuit;
}

inline std::string join_lines(const std::vector<std::string>& lines) {
    std::string text;
    for (const std::string& line : lines) {
        text += line;
        text += '\n';
    }
    return text;
}

inline std::string generate_noisy_source(std::mt19937& rng, uint32_t num_qubits, uint32_t num_ops) {
    return join_lines(generate_noisy_circuit(rng, num_qubits, num_ops).lines);
}

// ---------------------------------------------------------------------------
// Sampling equivalence
// ---------------------------------------------------------------------------

inline double tolerance_at_6_sigma(double mean_a, double mean_b, uint32_t shots) {
    // A floor keeps the tolerance from collapsing to zero for a column that
    // is (near-)deterministic in both samples.
    constexpr double kMinP = 1e-3;
    const double pooled = std::clamp(0.5 * (mean_a + mean_b), kMinP, 1.0 - kMinP);
    return 6.0 * std::sqrt(2.0 * pooled * (1.0 - pooled) / shots);
}

inline double column_mean(std::span<const uint8_t> values, uint32_t num_columns, uint32_t column,
                          uint32_t shots) {
    uint64_t ones = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        ones += values[static_cast<size_t>(shot) * num_columns + column];
    }
    return static_cast<double>(ones) / shots;
}

inline double parity_mean(std::span<const uint8_t> values, uint32_t num_columns, uint32_t col_a,
                          uint32_t col_b, uint32_t shots) {
    uint64_t ones = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        const uint8_t a = values[static_cast<size_t>(shot) * num_columns + col_a];
        const uint8_t b = values[static_cast<size_t>(shot) * num_columns + col_b];
        ones += static_cast<uint8_t>(a ^ b);
    }
    return static_cast<double>(ones) / shots;
}

inline void check_columns_agree(std::span<const uint8_t> a, std::span<const uint8_t> b,
                                uint32_t num_columns, uint32_t shots, const char* label) {
    for (uint32_t col = 0; col < num_columns; ++col) {
        const double mean_a = column_mean(a, num_columns, col, shots);
        const double mean_b = column_mean(b, num_columns, col, shots);
        const double tol = tolerance_at_6_sigma(mean_a, mean_b, shots);
        INFO(label << " column " << col << ": " << mean_a << " vs " << mean_b << ", tol " << tol);
        CHECK_THAT(mean_a, Catch::Matchers::WithinAbs(mean_b, tol));
    }
}

inline void check_parities_agree(std::span<const uint8_t> a, std::span<const uint8_t> b,
                                 uint32_t num_columns, uint32_t shots, const char* label) {
    for (uint32_t col = 0; col + 1 < num_columns; ++col) {
        const double mean_a = parity_mean(a, num_columns, col, col + 1, shots);
        const double mean_b = parity_mean(b, num_columns, col, col + 1, shots);
        const double tol = tolerance_at_6_sigma(mean_a, mean_b, shots);
        INFO(label << " parity (" << col << "," << col + 1 << "): " << mean_a << " vs " << mean_b
                   << ", tol " << tol);
        CHECK_THAT(mean_a, Catch::Matchers::WithinAbs(mean_b, tol));
    }
}

// Samples `original` and `reordered` with different seeds and requires every
// record column mean, every detector column mean, and every consecutive-pair
// record parity to agree within a cross-binomial tolerance. The point is
// distributional equality, not bit-exact records.
inline void check_sampling_equivalent(const HirModule& original, const HirModule& reordered,
                                      uint32_t shots, uint64_t seed_a, uint64_t seed_b) {
    const clifft::sampling::ExecutablePlan plan_a(clifft::sampling::plan_sampling(original));
    const clifft::sampling::ExecutablePlan plan_b(clifft::sampling::plan_sampling(reordered));
    REQUIRE(plan_a.num_visible_records() == plan_b.num_visible_records());
    REQUIRE(plan_a.num_detectors() == plan_b.num_detectors());

    const clifft::sampling::SamplingResult result_a =
        clifft::sampling::sample(plan_a, shots, seed_a);
    const clifft::sampling::SamplingResult result_b =
        clifft::sampling::sample(plan_b, shots, seed_b);

    check_columns_agree(result_a.measurements, result_b.measurements, plan_a.num_visible_records(),
                        shots, "record");
    check_columns_agree(result_a.detectors, result_b.detectors, plan_a.num_detectors(), shots,
                        "detector");
    check_parities_agree(result_a.measurements, result_b.measurements, plan_a.num_visible_records(),
                         shots, "record");
}

}  // namespace clifft::test
