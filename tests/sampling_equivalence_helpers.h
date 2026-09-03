#pragma once

// Shared helpers for tests that reorder HIR operations and must confirm the
// reordered program samples the same distribution as the original: a random
// noisy-circuit generator plus a statistical equivalence check. Split out of
// test_logical_noise_prefix.cc so test_schedule_dependence.cc and
// test_active_width_schedule_pass.cc can reuse both without duplicating them.

#include "clifft/frontend/hir.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <cassert>
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
        enum class Kind : uint8_t { XError, ZError, Depolarize1, Depolarize2, PauliChannel1 };

        size_t line_index = 0;
        uint32_t qubit = 0;
        uint32_t qubit2 = 0;  // Depolarize2 only.
        Kind kind = Kind::XError;
        double prob = 0.0;    // XError, ZError, Depolarize1, Depolarize2.
        double prob_x = 0.0;  // PauliChannel1 only.
        double prob_z = 0.0;  // PauliChannel1 only.
    };

    std::vector<std::string> lines;
    std::vector<NoiseLine> noise_lines;
};

inline double next_unit(std::mt19937& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

// Deterministic generator over a mixed gate set: absorbed Cliffords (H, S,
// CX, CZ) that the frontend folds into its frame without emitting an HIR op,
// movable non-Clifford ops (T, T_DAG, R_Z, M, MX, MR), a reset, a noisy
// measurement, classical feedback on an earlier record, every Pauli noise
// channel this suite covers, and DETECTOR/OBSERVABLE_INCLUDE targets into
// the growing record. Every generated line is individually valid, so the
// circuit as a whole always parses and traces.
//
// Noise probabilities are drawn from a set that includes 0.5 and 1.0, not
// just small values: a wrong noise-crossing sign shows up in a fixed
// fraction of shots landing on the wrong outcome, and that fraction has to
// clear check_sampling_equivalent's statistical tolerance to be caught. At a
// small probability the wrong fraction can hide inside the tolerance band;
// at 0.5 or 1.0 it cannot.
inline GeneratedCircuit generate_noisy_circuit(std::mt19937& rng, uint32_t num_qubits,
                                               uint32_t num_ops) {
    assert(num_qubits >= 2 && "two-qubit gates and channels need a second qubit");
    static const double kAngles[] = {0.125, 0.25, 0.375, 0.625, 0.75, 0.875};
    static const double kNoiseProbs[] = {0.05, 0.1, 0.2, 0.3, 0.5, 1.0};

    GeneratedCircuit circuit;
    uint32_t measurement_count = 0;
    for (uint32_t op = 0; op < num_ops; ++op) {
        const uint32_t q = rng() % num_qubits;
        const uint32_t q2 = (q + 1 + rng() % (num_qubits - 1)) % num_qubits;
        switch (rng() % 20) {
            case 0:
            case 18:
                circuit.lines.push_back("T " + std::to_string(q));
                break;
            case 1:
                circuit.lines.push_back("T_DAG " + std::to_string(q));
                break;
            case 2:
            case 19:
                circuit.lines.push_back("R_Z(" +
                                        std::to_string(kAngles[rng() % std::size(kAngles)]) + ") " +
                                        std::to_string(q));
                break;
            case 3:
                circuit.lines.push_back("H " + std::to_string(q));
                break;
            case 4:
                circuit.lines.push_back("S " + std::to_string(q));
                break;
            case 5:
                circuit.lines.push_back("CX " + std::to_string(q) + " " + std::to_string(q2));
                break;
            case 6:
                circuit.lines.push_back("CZ " + std::to_string(q) + " " + std::to_string(q2));
                break;
            case 7:
                circuit.lines.push_back("M " + std::to_string(q));
                ++measurement_count;
                break;
            case 8:
                circuit.lines.push_back("MX " + std::to_string(q));
                ++measurement_count;
                break;
            case 9: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("M(" + std::to_string(prob) + ") " + std::to_string(q));
                ++measurement_count;
                break;
            }
            case 10:
                circuit.lines.push_back("MR " + std::to_string(q));
                ++measurement_count;
                break;
            case 11:
                circuit.lines.push_back("R " + std::to_string(q));
                break;
            case 12: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("X_ERROR(" + std::to_string(prob) + ") " +
                                        std::to_string(q));
                circuit.noise_lines.push_back({.line_index = circuit.lines.size() - 1,
                                               .qubit = q,
                                               .kind = GeneratedCircuit::NoiseLine::Kind::XError,
                                               .prob = prob});
                break;
            }
            case 13: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("Z_ERROR(" + std::to_string(prob) + ") " +
                                        std::to_string(q));
                circuit.noise_lines.push_back({.line_index = circuit.lines.size() - 1,
                                               .qubit = q,
                                               .kind = GeneratedCircuit::NoiseLine::Kind::ZError,
                                               .prob = prob});
                break;
            }
            case 14: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("DEPOLARIZE1(" + std::to_string(prob) + ") " +
                                        std::to_string(q));
                circuit.noise_lines.push_back(
                    {.line_index = circuit.lines.size() - 1,
                     .qubit = q,
                     .kind = GeneratedCircuit::NoiseLine::Kind::Depolarize1,
                     .prob = prob});
                break;
            }
            case 15: {
                const double prob = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                circuit.lines.push_back("DEPOLARIZE2(" + std::to_string(prob) + ") " +
                                        std::to_string(q) + " " + std::to_string(q2));
                circuit.noise_lines.push_back(
                    {.line_index = circuit.lines.size() - 1,
                     .qubit = q,
                     .qubit2 = q2,
                     .kind = GeneratedCircuit::NoiseLine::Kind::Depolarize2,
                     .prob = prob});
                break;
            }
            case 16: {
                const double prob_x = kNoiseProbs[rng() % std::size(kNoiseProbs)];
                // PAULI_CHANNEL_1 requires P(X) + P(Y) + P(Z) <= 1. P(Y) is
                // always exactly zero (so that channel is structurally
                // absent), so P(Z) is drawn only from values that keep the
                // pair within budget, falling back to zero when P(X) alone
                // already uses it up.
                double allowed_z[std::size(kNoiseProbs)];
                size_t allowed_count = 0;
                for (double candidate : kNoiseProbs) {
                    if (prob_x + candidate <= 1.0) {
                        allowed_z[allowed_count++] = candidate;
                    }
                }
                const double prob_z = allowed_count > 0 ? allowed_z[rng() % allowed_count] : 0.0;
                circuit.lines.push_back("PAULI_CHANNEL_1(" + std::to_string(prob_x) + ", 0, " +
                                        std::to_string(prob_z) + ") " + std::to_string(q));
                circuit.noise_lines.push_back(
                    {.line_index = circuit.lines.size() - 1,
                     .qubit = q,
                     .kind = GeneratedCircuit::NoiseLine::Kind::PauliChannel1,
                     .prob_x = prob_x,
                     .prob_z = prob_z});
                break;
            }
            case 17: {
                if (measurement_count > 0) {
                    const uint32_t max_back = std::min(measurement_count, 4u);
                    const uint32_t back = 1 + (rng() % max_back);
                    circuit.lines.push_back("CX rec[-" + std::to_string(back) + "] " +
                                            std::to_string(q));
                } else {
                    circuit.lines.push_back("T " + std::to_string(q));
                }
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
    if (measurement_count > 0) {
        const uint32_t back = 1 + (rng() % measurement_count);
        circuit.lines.push_back("OBSERVABLE_INCLUDE(0) rec[-" + std::to_string(back) + "]");
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

// Replaces every noise line with the explicit Pauli(s) its fixed realization
// fired, or drops the line if it did not fire. Uses its own RNG stream so
// the realization is reproducible but independent of any reordering draws.
// Every replacement is itself a Clifford gate (X, Y, or Z), so the frontend
// absorbs it into its Clifford frame instead of emitting an HIR op -- the
// same as a dropped line -- which is what lets a legality-oracle test
// compare the realized circuit's ops one-for-one against the noise-free
// positions of the original.
inline std::vector<std::string> realize_noise(const GeneratedCircuit& circuit, std::mt19937& rng) {
    using Kind = GeneratedCircuit::NoiseLine::Kind;
    std::vector<uint8_t> drop(circuit.lines.size(), 0);
    std::vector<std::string> replacement(circuit.lines.size());
    for (const GeneratedCircuit::NoiseLine& noise : circuit.noise_lines) {
        const double roll = next_unit(rng);
        switch (noise.kind) {
            case Kind::XError:
                if (roll >= noise.prob) {
                    drop[noise.line_index] = 1;
                } else {
                    replacement[noise.line_index] = "X " + std::to_string(noise.qubit);
                }
                break;
            case Kind::ZError:
                if (roll >= noise.prob) {
                    drop[noise.line_index] = 1;
                } else {
                    replacement[noise.line_index] = "Z " + std::to_string(noise.qubit);
                }
                break;
            case Kind::Depolarize1: {
                if (roll >= noise.prob) {
                    drop[noise.line_index] = 1;
                    break;
                }
                static const char* const paulis[] = {"X", "Y", "Z"};
                replacement[noise.line_index] =
                    std::string(paulis[rng() % 3]) + " " + std::to_string(noise.qubit);
                break;
            }
            case Kind::Depolarize2: {
                if (roll >= noise.prob) {
                    drop[noise.line_index] = 1;
                    break;
                }
                // One of the 15 non-identity two-qubit Paulis: each qubit
                // independently I, X, Y, or Z, excluding the all-identity
                // combination. A replacement with one nonidentity factor
                // becomes a single Pauli gate line; two nonidentity factors
                // become two lines, one per qubit.
                static const char* const single[] = {nullptr, "X", "Y", "Z"};
                const uint32_t combo = 1 + (rng() % 15);
                const uint32_t first = combo / 4;
                const uint32_t second = combo % 4;
                std::string lines;
                if (first != 0) {
                    lines = std::string(single[first]) + " " + std::to_string(noise.qubit);
                }
                if (second != 0) {
                    if (!lines.empty()) {
                        lines += "\n";
                    }
                    lines += std::string(single[second]) + " " + std::to_string(noise.qubit2);
                }
                replacement[noise.line_index] = lines;
                break;
            }
            case Kind::PauliChannel1: {
                if (roll < noise.prob_x) {
                    replacement[noise.line_index] = "X " + std::to_string(noise.qubit);
                } else if (roll < noise.prob_x + noise.prob_z) {
                    replacement[noise.line_index] = "Z " + std::to_string(noise.qubit);
                } else {
                    drop[noise.line_index] = 1;
                }
                break;
            }
        }
    }

    std::vector<std::string> realized;
    realized.reserve(circuit.lines.size());
    for (size_t i = 0; i < circuit.lines.size(); ++i) {
        if (drop[i]) {
            continue;
        }
        realized.push_back(replacement[i].empty() ? circuit.lines[i] : replacement[i]);
    }
    return realized;
}

// ---------------------------------------------------------------------------
// Logical noise prefix
// ---------------------------------------------------------------------------

// True when some operation's logical_noise_prefix entry disagrees with its
// schedule position (the number of NOISE ops that actually precede it in
// `ops`). That disagreement is exactly what a noise-crossing reorder leaves
// behind, so this doubles as a witness that a pass built on apply_schedule
// under noise transparency actually moved something across a NOISE op,
// rather than merely leaving the option open.
inline bool crossed_noise(const HirModule& hir) {
    if (!hir.has_logical_noise_prefix()) {
        return false;
    }
    uint32_t schedule_count = 0;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.logical_noise_prefix[i] != schedule_count) {
            return true;
        }
        if (hir.ops[i].op_type() == OpType::NOISE) {
            ++schedule_count;
        }
    }
    return false;
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
