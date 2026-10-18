#pragma once

// Differential checks for HIR reorderings, shared by noise-position,
// dependence, and scheduler tests. These compare Clifft plans; independent
// simulator references are covered by the Python Aer and Stim tests.

#include "clifft/frontend/hir.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/xoshiro.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <optional>
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

// Mix non-Clifford gates, measurements, feedback, and noise with absorbed
// Cliffords that vary their Pauli bases. Include frequent errors so incorrect
// noise-crossing signs cannot hide within sampling tolerances.
inline GeneratedCircuit generate_noisy_circuit(clifft::Xoshiro256PlusPlus& rng, uint32_t num_qubits,
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

inline std::string generate_noisy_source(clifft::Xoshiro256PlusPlus& rng, uint32_t num_qubits,
                                         uint32_t num_ops) {
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

// realize_noise's realized lines, plus how many of circuit.noise_lines
// fired versus dropped -- a witness a caller can require to be positive on
// both sides, proving the realization stream actually exercises both
// outcomes rather than only ever firing or only ever dropping.
struct RealizedCircuit {
    std::vector<std::string> lines;
    size_t fired = 0;
    size_t dropped = 0;
};

inline RealizedCircuit realize_noise(const GeneratedCircuit& circuit,
                                     clifft::Xoshiro256PlusPlus& rng) {
    using Kind = GeneratedCircuit::NoiseLine::Kind;
    std::vector<uint8_t> drop(circuit.lines.size(), 0);
    std::vector<std::string> replacement(circuit.lines.size());
    for (const GeneratedCircuit::NoiseLine& noise : circuit.noise_lines) {
        const double roll = rng.next_double();
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

    RealizedCircuit result;
    result.lines.reserve(circuit.lines.size());
    for (size_t i = 0; i < circuit.lines.size(); ++i) {
        if (drop[i]) {
            continue;
        }
        result.lines.push_back(replacement[i].empty() ? circuit.lines[i] : replacement[i]);
    }
    for (const GeneratedCircuit::NoiseLine& noise : circuit.noise_lines) {
        if (drop[noise.line_index]) {
            ++result.dropped;
        } else {
            ++result.fired;
        }
    }
    return result;
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

// ---------------------------------------------------------------------------
// Exact distribution evaluation and per-realization / unconditional checks
// ---------------------------------------------------------------------------

struct ExactShotOutcome {
    uint64_t visible_records = 0;
    uint64_t detectors = 0;
    uint64_t observables = 0;

    auto operator<=>(const ExactShotOutcome&) const = default;
};

struct ExactOutcomeEntry {
    ExactShotOutcome outcome{};
    double probability = 0.0;
};

struct ExactPlanDistribution {
    uint32_t num_visible_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    double discarded_probability = 0.0;
    std::vector<ExactOutcomeEntry> kept_outcomes;
    std::vector<double> unconditional_exp_vals;
    std::vector<double> kept_exp_vals;
    size_t reachable_trajectories = 0;
};

inline void canonicalize_outcome_entries(std::vector<ExactOutcomeEntry>& entries) {
    std::ranges::sort(entries, [](const ExactOutcomeEntry& a, const ExactOutcomeEntry& b) {
        return a.outcome < b.outcome;
    });
    size_t write = 0;
    for (size_t read = 0; read < entries.size();) {
        ExactShotOutcome key = entries[read].outcome;
        double sum = 0.0;
        while (read < entries.size() && entries[read].outcome == key) {
            sum += entries[read].probability;
            ++read;
        }
        if (sum > 1e-15) {
            entries[write++] = ExactOutcomeEntry{key, sum};
        }
    }
    entries.resize(write);
}

inline bool distributions_equal(const ExactPlanDistribution& a, const ExactPlanDistribution& b,
                                double tolerance = 1e-9) {
    if (a.num_visible_records != b.num_visible_records || a.num_detectors != b.num_detectors ||
        a.num_observables != b.num_observables || a.num_exp_vals != b.num_exp_vals) {
        return false;
    }
    if (std::abs(a.discarded_probability - b.discarded_probability) > tolerance) {
        return false;
    }
    for (uint32_t e = 0; e < a.num_exp_vals; ++e) {
        if (std::abs(a.kept_exp_vals[e] - b.kept_exp_vals[e]) > tolerance) {
            return false;
        }
    }
    size_t i = 0;
    size_t j = 0;
    while (i < a.kept_outcomes.size() || j < b.kept_outcomes.size()) {
        if (j == b.kept_outcomes.size() ||
            (i < a.kept_outcomes.size() &&
             a.kept_outcomes[i].outcome < b.kept_outcomes[j].outcome)) {
            if (a.kept_outcomes[i].probability > tolerance) {
                return false;
            }
            ++i;
        } else if (i == a.kept_outcomes.size() ||
                   b.kept_outcomes[j].outcome < a.kept_outcomes[i].outcome) {
            if (b.kept_outcomes[j].probability > tolerance) {
                return false;
            }
            ++j;
        } else {
            if (std::abs(a.kept_outcomes[i].probability - b.kept_outcomes[j].probability) >
                tolerance) {
                return false;
            }
            ++i;
            ++j;
        }
    }
    return true;
}

inline void require_distributions_equal(const ExactPlanDistribution& a,
                                        const ExactPlanDistribution& b, double tolerance = 1e-9) {
    REQUIRE(a.num_visible_records == b.num_visible_records);
    REQUIRE(a.num_detectors == b.num_detectors);
    REQUIRE(a.num_observables == b.num_observables);
    REQUIRE(a.num_exp_vals == b.num_exp_vals);
    REQUIRE_THAT(a.discarded_probability,
                 Catch::Matchers::WithinAbs(b.discarded_probability, tolerance));
    for (uint32_t e = 0; e < a.num_exp_vals; ++e) {
        CAPTURE(e);
        REQUIRE_THAT(a.kept_exp_vals[e], Catch::Matchers::WithinAbs(b.kept_exp_vals[e], tolerance));
    }
    size_t i = 0;
    size_t j = 0;
    while (i < a.kept_outcomes.size() || j < b.kept_outcomes.size()) {
        if (j == b.kept_outcomes.size() ||
            (i < a.kept_outcomes.size() &&
             a.kept_outcomes[i].outcome < b.kept_outcomes[j].outcome)) {
            CAPTURE(a.kept_outcomes[i].outcome.visible_records,
                    a.kept_outcomes[i].outcome.detectors, a.kept_outcomes[i].outcome.observables);
            REQUIRE_THAT(a.kept_outcomes[i].probability,
                         Catch::Matchers::WithinAbs(0.0, tolerance));
            ++i;
        } else if (i == a.kept_outcomes.size() ||
                   b.kept_outcomes[j].outcome < a.kept_outcomes[i].outcome) {
            CAPTURE(b.kept_outcomes[j].outcome.visible_records,
                    b.kept_outcomes[j].outcome.detectors, b.kept_outcomes[j].outcome.observables);
            REQUIRE_THAT(0.0,
                         Catch::Matchers::WithinAbs(b.kept_outcomes[j].probability, tolerance));
            ++j;
        } else {
            CAPTURE(a.kept_outcomes[i].outcome.visible_records,
                    a.kept_outcomes[i].outcome.detectors, a.kept_outcomes[i].outcome.observables);
            REQUIRE_THAT(a.kept_outcomes[i].probability,
                         Catch::Matchers::WithinAbs(b.kept_outcomes[j].probability, tolerance));
            ++i;
            ++j;
        }
    }
}

// Rewrites a SamplingPlan into a replayable form where:
//   1. ApplyReadoutNoise flip symbols become unbound Presampled symbols so
//      Executor::replay_shot can evaluate downstream feedback and affine
//      observables under explicit readout-flip assignments.
//   2. WriteDetector postselection flags are cleared inside the executor and
//      evaluated after the shot finishes so mid-circuit rejection never
//      truncates suffix measurement probabilities during exhaustive replay.
struct PreparedReplayPlan {
    struct ReadoutSite {
        clifft::sampling::SymbolId flip{};
        uint32_t record = 0;
        double prob_zero_to_one = 0.0;
        double prob_one_to_zero = 0.0;
    };

    clifft::sampling::SamplingPlan source_plan;
    std::vector<ReadoutSite> readout_sites{};
    std::vector<uint64_t> visible_record_flip_mask{};
    std::vector<uint64_t> detector_flip_mask{};
    std::vector<uint64_t> observable_flip_mask{};
    uint64_t postselected_detector_mask = 0;
    clifft::sampling::SamplingPlan replay_plan;
    clifft::sampling::ExecutablePlan executable;
    std::vector<uint32_t> ordinal_of_symbol;
    clifft::sampling::Executor executor;

    explicit PreparedReplayPlan(const clifft::sampling::SamplingPlan& plan)
        : source_plan(plan),
          replay_plan(build_replay_plan(plan, readout_sites, visible_record_flip_mask,
                                        detector_flip_mask, observable_flip_mask,
                                        postselected_detector_mask)),
          executable(replay_plan),
          ordinal_of_symbol(
              build_presampled_ordinals(replay_plan, executable.num_presampled_symbols())),
          executor(executable) {}

    PreparedReplayPlan(const PreparedReplayPlan&) = delete;
    PreparedReplayPlan& operator=(const PreparedReplayPlan&) = delete;

  private:
    static clifft::sampling::SamplingPlan build_replay_plan(
        const clifft::sampling::SamplingPlan& plan, std::vector<ReadoutSite>& readout_sites,
        std::vector<uint64_t>& visible_record_flip_mask, std::vector<uint64_t>& detector_flip_mask,
        std::vector<uint64_t>& observable_flip_mask, uint64_t& postselected_detector_mask) {
        const uint32_t total_records = plan.num_visible_records + plan.num_hidden_records;
        REQUIRE(plan.num_visible_records <= 64);
        REQUIRE(plan.num_detectors <= 64);
        REQUIRE(plan.num_observables <= 64);
        REQUIRE(total_records <= 12);

        clifft::sampling::SamplingPlan out = plan;
        out.source_map.reset();
        out.actions.clear();
        out.actions.reserve(plan.actions.size());

        std::vector<uint64_t> record_flip_mask(total_records, 0);
        detector_flip_mask.assign(plan.num_detectors, 0);
        observable_flip_mask.assign(plan.num_observables, 0);
        postselected_detector_mask = 0;

        for (const clifft::sampling::PlannedAction& planned : plan.actions) {
            if (const auto* rn =
                    std::get_if<clifft::sampling::ApplyReadoutNoise>(&planned.action)) {
                const size_t site_idx = readout_sites.size();
                REQUIRE(site_idx < 64);
                const uint32_t rec = clifft::sampling::index(rn->record);
                readout_sites.push_back(ReadoutSite{
                    .flip = rn->flip,
                    .record = rec,
                    .prob_zero_to_one = rn->prob_zero_to_one,
                    .prob_one_to_zero = rn->prob_one_to_zero,
                });
                out.symbols[clifft::sampling::index(rn->flip)] =
                    clifft::sampling::SymbolKind::Presampled;
                record_flip_mask[rec] ^= (uint64_t{1} << site_idx);
                continue;
            }
            if (const auto* wd = std::get_if<clifft::sampling::WriteDetector>(&planned.action)) {
                const uint32_t det = clifft::sampling::index(wd->detector);
                uint64_t mask = 0;
                for (clifft::sampling::RecordSlot r : wd->outcome.records()) {
                    mask ^= record_flip_mask[clifft::sampling::index(r)];
                }
                detector_flip_mask[det] = mask;
                if (wd->postselected) {
                    postselected_detector_mask |= (uint64_t{1} << det);
                }
                clifft::sampling::WriteDetector modified = *wd;
                modified.postselected = false;
                out.actions.push_back(clifft::sampling::PlannedAction{
                    planned.active_before, planned.active_after, std::move(modified)});
                continue;
            }
            if (const auto* wo = std::get_if<clifft::sampling::WriteObservable>(&planned.action)) {
                const uint32_t obs = clifft::sampling::index(wo->observable);
                if (const auto* parity =
                        std::get_if<clifft::sampling::RecordParity>(&wo->outcome)) {
                    uint64_t mask = 0;
                    for (clifft::sampling::RecordSlot r : parity->records()) {
                        mask ^= record_flip_mask[clifft::sampling::index(r)];
                    }
                    observable_flip_mask[obs] = mask;
                }
                out.actions.push_back(planned);
                continue;
            }
            out.actions.push_back(planned);
        }

        visible_record_flip_mask.assign(record_flip_mask.begin(),
                                        record_flip_mask.begin() + plan.num_visible_records);
        return out;
    }

    static std::vector<uint32_t> build_presampled_ordinals(
        const clifft::sampling::SamplingPlan& plan, uint32_t num_presampled_symbols) {
        std::vector<uint32_t> ordinals(plan.symbols.size(), 0);
        uint32_t count = 0;
        for (size_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
            if (plan.symbols[symbol] == clifft::sampling::SymbolKind::Presampled) {
                ordinals[symbol] = count++;
            }
        }
        REQUIRE(count == num_presampled_symbols);
        return ordinals;
    }
};

// Evaluates the exact distribution over visible records, detectors,
// observables, postselection, and expectation values conditioned on a fixed
// quantum noise realization (by summing over all hidden reset records and all
// readout-noise flip realizations).
inline ExactPlanDistribution evaluate_conditional_distribution(
    PreparedReplayPlan& prepared, std::span<const std::optional<size_t>> noise_realization) {
    const auto& plan = prepared.source_plan;
    const uint32_t num_visible = plan.num_visible_records;
    const uint32_t num_hidden = plan.num_hidden_records;
    const uint32_t total_records = num_visible + num_hidden;
    const size_t num_readout = prepared.readout_sites.size();
    REQUIRE(total_records + num_readout <= 14);

    std::vector<uint8_t> presampled(prepared.executable.num_presampled_symbols(), 0);
    for (size_t site = 0; site < noise_realization.size(); ++site) {
        if (!noise_realization[site].has_value()) {
            continue;
        }
        const clifft::sampling::SymbolId symbol =
            plan.presampled_noise_sites[site].outcomes[*noise_realization[site]].symbol;
        presampled[prepared.ordinal_of_symbol[clifft::sampling::index(symbol)]] = 1;
    }

    ExactPlanDistribution dist;
    dist.num_visible_records = num_visible;
    dist.num_detectors = plan.num_detectors;
    dist.num_observables = plan.num_observables;
    dist.num_exp_vals = plan.num_exp_vals;
    dist.unconditional_exp_vals.assign(plan.num_exp_vals, 0.0);
    dist.kept_exp_vals.assign(plan.num_exp_vals, 0.0);

    const size_t num_raw_patterns = size_t{1} << total_records;
    const size_t num_readout_patterns = size_t{1} << num_readout;
    std::vector<uint8_t> forced(total_records, 0);
    std::vector<uint8_t> current_records(total_records, 0);
    double total_probability = 0.0;

    for (size_t rho_bits = 0; rho_bits < num_readout_patterns; ++rho_bits) {
        for (size_t j = 0; j < num_readout; ++j) {
            const uint32_t ord =
                prepared.ordinal_of_symbol[clifft::sampling::index(prepared.readout_sites[j].flip)];
            presampled[ord] = static_cast<uint8_t>((rho_bits >> j) & 1U);
        }

        for (size_t pattern = 0; pattern < num_raw_patterns; ++pattern) {
            for (uint32_t bit = 0; bit < total_records; ++bit) {
                forced[bit] = static_cast<uint8_t>((pattern >> bit) & 1U);
                current_records[bit] = forced[bit];
            }

            double readout_prob = 1.0;
            for (size_t j = 0; j < num_readout; ++j) {
                const auto& rs = prepared.readout_sites[j];
                const bool src_one = current_records[rs.record] != 0;
                const double p_flip = src_one ? rs.prob_one_to_zero : rs.prob_zero_to_one;
                const bool flipped = ((rho_bits >> j) & 1U) != 0;
                const double p_step = flipped ? p_flip : (1.0 - p_flip);
                readout_prob *= p_step;
                if (flipped) {
                    current_records[rs.record] ^= 1U;
                }
            }
            if (readout_prob <= 0.0) {
                continue;
            }

            const clifft::sampling::ReplayResult replay =
                prepared.executor.replay_shot(forced, presampled);
            if (!replay.reachable) {
                continue;
            }

            const double weight = readout_prob * std::exp(replay.log_probability);
            if (weight <= 0.0) {
                continue;
            }

            ++dist.reachable_trajectories;
            total_probability += weight;

            uint64_t vis_bits = 0;
            for (uint32_t r = 0; r < num_visible; ++r) {
                const uint8_t bit =
                    forced[r] ^
                    static_cast<uint8_t>(
                        std::popcount(prepared.visible_record_flip_mask[r] & rho_bits) & 1);
                vis_bits |= static_cast<uint64_t>(bit) << r;
            }

            uint64_t det_bits = 0;
            for (uint32_t d = 0; d < plan.num_detectors; ++d) {
                const uint8_t bit =
                    prepared.executor.detectors()[d] ^
                    static_cast<uint8_t>(std::popcount(prepared.detector_flip_mask[d] & rho_bits) &
                                         1);
                det_bits |= static_cast<uint64_t>(bit) << d;
            }

            uint64_t obs_bits = 0;
            for (uint32_t o = 0; o < plan.num_observables; ++o) {
                const uint8_t bit =
                    prepared.executor.observables()[o] ^
                    static_cast<uint8_t>(
                        std::popcount(prepared.observable_flip_mask[o] & rho_bits) & 1);
                obs_bits |= static_cast<uint64_t>(bit) << o;
            }

            for (uint32_t e = 0; e < plan.num_exp_vals; ++e) {
                dist.unconditional_exp_vals[e] += weight * prepared.executor.exp_vals()[e];
            }

            if ((det_bits & prepared.postselected_detector_mask) != 0) {
                dist.discarded_probability += weight;
            } else {
                dist.kept_outcomes.push_back(
                    ExactOutcomeEntry{ExactShotOutcome{vis_bits, det_bits, obs_bits}, weight});
                for (uint32_t e = 0; e < plan.num_exp_vals; ++e) {
                    dist.kept_exp_vals[e] += weight * prepared.executor.exp_vals()[e];
                }
            }
        }
    }

    REQUIRE_THAT(total_probability, Catch::Matchers::WithinAbs(1.0, 1e-9));
    canonicalize_outcome_entries(dist.kept_outcomes);
    return dist;
}

// Evaluates the full unconditional exact distribution of `plan` by integrating
// over all quantum noise realizations, readout flips, and hidden reset records.
inline ExactPlanDistribution compute_exact_plan_distribution(
    const clifft::sampling::SamplingPlan& plan) {
    PreparedReplayPlan prepared(plan);
    const size_t num_sites = plan.presampled_noise_sites.size();
    std::vector<size_t> outcome_counts(num_sites);
    size_t total_realizations = 1;
    for (size_t site = 0; site < num_sites; ++site) {
        outcome_counts[site] = plan.presampled_noise_sites[site].outcomes.size();
        total_realizations *= outcome_counts[site] + 1;
        REQUIRE(total_realizations <= 256);
    }

    ExactPlanDistribution total;
    total.num_visible_records = plan.num_visible_records;
    total.num_detectors = plan.num_detectors;
    total.num_observables = plan.num_observables;
    total.num_exp_vals = plan.num_exp_vals;
    total.unconditional_exp_vals.assign(plan.num_exp_vals, 0.0);
    total.kept_exp_vals.assign(plan.num_exp_vals, 0.0);

    std::vector<size_t> digit(num_sites, 0);
    std::vector<std::optional<size_t>> realization(num_sites);
    double sum_noise_weights = 0.0;

    for (size_t iter = 0; iter < total_realizations; ++iter) {
        double noise_weight = 1.0;
        for (size_t site = 0; site < num_sites; ++site) {
            const auto& site_info = plan.presampled_noise_sites[site];
            if (digit[site] == 0) {
                realization[site] = std::nullopt;
                noise_weight *= std::max(0.0, 1.0 - site_info.total_probability);
            } else {
                realization[site] = digit[site] - 1;
                noise_weight *= site_info.outcomes[digit[site] - 1].probability;
            }
        }
        sum_noise_weights += noise_weight;

        if (noise_weight > 0.0) {
            const ExactPlanDistribution cond =
                evaluate_conditional_distribution(prepared, realization);
            total.reachable_trajectories += cond.reachable_trajectories;
            total.discarded_probability += noise_weight * cond.discarded_probability;
            for (uint32_t e = 0; e < plan.num_exp_vals; ++e) {
                total.unconditional_exp_vals[e] += noise_weight * cond.unconditional_exp_vals[e];
                total.kept_exp_vals[e] += noise_weight * cond.kept_exp_vals[e];
            }
            for (const ExactOutcomeEntry& entry : cond.kept_outcomes) {
                total.kept_outcomes.push_back(
                    ExactOutcomeEntry{entry.outcome, noise_weight * entry.probability});
            }
        }

        for (size_t site = 0; site < num_sites; ++site) {
            if (++digit[site] <= outcome_counts[site]) {
                break;
            }
            digit[site] = 0;
        }
    }

    REQUIRE_THAT(sum_noise_weights, Catch::Matchers::WithinAbs(1.0, 1e-9));
    canonicalize_outcome_entries(total.kept_outcomes);
    return total;
}

inline ExactPlanDistribution compute_exact_hir_distribution(
    const HirModule& hir, const clifft::sampling::SamplingPlanOptions& options = {}) {
    return compute_exact_plan_distribution(clifft::sampling::plan_sampling(hir, options));
}

inline void check_unconditional_exact_equivalent(
    const HirModule& original, const HirModule& optimized,
    const clifft::sampling::SamplingPlanOptions& options = {}, double tolerance = 1e-9) {
    const ExactPlanDistribution dist_a = compute_exact_hir_distribution(original, options);
    const ExactPlanDistribution dist_b = compute_exact_hir_distribution(optimized, options);
    require_distributions_equal(dist_a, dist_b, tolerance);
}

struct ExactEquivalenceStats {
    size_t realizations = 0;
    size_t records = 0;
    size_t reachable = 0;
};

// Compare full visible record probabilities, reachability, detectors,
// observables, postselection, and expectation values for the same fixed
// quantum noise outcomes. Supports hidden reset records and readout noise by
// marginalizing over hidden records and readout flips within each realization.
inline ExactEquivalenceStats check_exact_equivalent(
    const HirModule& original, const HirModule& reordered, clifft::Xoshiro256PlusPlus& rng,
    size_t max_realizations = 16, const clifft::sampling::SamplingPlanOptions& options = {}) {
    const clifft::sampling::SamplingPlan plan_a =
        clifft::sampling::plan_sampling(original, options);
    const clifft::sampling::SamplingPlan plan_b =
        clifft::sampling::plan_sampling(reordered, options);

    REQUIRE(plan_a.num_visible_records == plan_b.num_visible_records);
    REQUIRE(plan_a.num_detectors == plan_b.num_detectors);
    REQUIRE(plan_a.num_observables == plan_b.num_observables);
    REQUIRE(plan_a.num_exp_vals == plan_b.num_exp_vals);
    REQUIRE(plan_a.num_visible_records <= 10);

    REQUIRE(plan_a.presampled_noise_sites.size() == plan_b.presampled_noise_sites.size());
    const size_t num_sites = plan_a.presampled_noise_sites.size();
    std::vector<size_t> outcome_counts(num_sites);
    for (size_t site = 0; site < num_sites; ++site) {
        const auto& outcomes_a = plan_a.presampled_noise_sites[site].outcomes;
        const auto& outcomes_b = plan_b.presampled_noise_sites[site].outcomes;
        REQUIRE(outcomes_a.size() == outcomes_b.size());
        for (size_t k = 0; k < outcomes_a.size(); ++k) {
            REQUIRE(outcomes_a[k].probability == outcomes_b[k].probability);
        }
        outcome_counts[site] = outcomes_a.size();
    }

    PreparedReplayPlan prepared_a(plan_a);
    PreparedReplayPlan prepared_b(plan_b);

    const uint32_t num_visible_records = plan_a.num_visible_records;
    const size_t num_record_patterns = size_t{1} << num_visible_records;
    std::vector<std::optional<size_t>> realization(num_sites);
    ExactEquivalenceStats stats;

    auto run_realization = [&]() {
        ++stats.realizations;
        stats.records += num_record_patterns;
        const ExactPlanDistribution dist_a =
            evaluate_conditional_distribution(prepared_a, realization);
        const ExactPlanDistribution dist_b =
            evaluate_conditional_distribution(prepared_b, realization);
        stats.reachable += dist_a.kept_outcomes.size();
        require_distributions_equal(dist_a, dist_b, 1e-9);
    };

    size_t total_realizations = 1;
    for (size_t site = 0; site < num_sites; ++site) {
        total_realizations *= outcome_counts[site] + 1;
        if (total_realizations > max_realizations) {
            total_realizations = max_realizations + 1;
            break;
        }
    }

    if (total_realizations <= max_realizations) {
        std::vector<size_t> digit(num_sites, 0);
        for (size_t iter = 0; iter < total_realizations; ++iter) {
            for (size_t site = 0; site < num_sites; ++site) {
                realization[site] =
                    digit[site] == 0 ? std::nullopt : std::optional<size_t>(digit[site] - 1);
            }
            run_realization();
            for (size_t site = 0; site < num_sites; ++site) {
                if (++digit[site] <= outcome_counts[site]) {
                    break;
                }
                digit[site] = 0;
            }
        }
    } else {
        std::ranges::fill(realization, std::nullopt);
        run_realization();
        for (size_t draw = 1; draw < max_realizations; ++draw) {
            for (size_t site = 0; site < num_sites; ++site) {
                const size_t radix = outcome_counts[site] + 1;
                const size_t choice = rng() % radix;
                realization[site] = choice == 0 ? std::nullopt : std::optional<size_t>(choice - 1);
            }
            run_realization();
        }
    }

    return stats;
}

}  // namespace clifft::test
