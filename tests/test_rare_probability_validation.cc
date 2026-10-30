#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"

#include "dense_circuit_oracle.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <numbers>

using namespace clifft;
using namespace clifft::test;

namespace {

ExactPlanDistribution rare_distribution(double probability) {
    ExactPlanDistribution result;
    result.num_visible_records = 1;
    result.num_exp_vals = 1;
    result.kept_outcomes = {{{0, 0, 0, {1.0}}, 1.0 - probability},
                            {{1, 0, 0, {-1.0}}, probability}};
    result.kept_exp_vals = {1.0 - 2.0 * probability};
    return result;
}

}  // namespace

TEST_CASE("Exact comparison rejects corrupt rare outcomes", "[rare_probability]") {
    const double probability = GENERATE(1e-12, 1e-16, 1e-24);
    const auto reference = rare_distribution(probability);
    auto actual = reference;
    SECTION("missing outcome") {
        actual.kept_outcomes.pop_back();
    }
    SECTION("misweighted outcome") {
        actual.kept_outcomes.back().probability *= 1.5;
    }
    SECTION("wrong conditional expectation") {
        actual.kept_outcomes.back().outcome.exp_vals[0] = 1.0;
    }
    std::string detail;
    REQUIRE_FALSE(distributions_equal(reference, actual, 1e-9, &detail));
    REQUIRE(detail.find("outcome(records=1") != std::string::npos);
    REQUIRE(detail.find("reference_probability=") != std::string::npos);
    REQUIRE(detail.find("actual_probability=") != std::string::npos);
    REQUIRE(detail.find("contract=branch-relative") != std::string::npos);
    REQUIRE(detail.find("omitted_bound=") != std::string::npos);
}

TEST_CASE("Outcome aggregation retains tiny joint probabilities", "[rare_probability]") {
    auto distribution = rare_distribution(1e-20);
    canonicalize_outcome_entries(distribution.kept_outcomes);
    REQUIRE(distribution.kept_outcomes.size() == 2);
    REQUIRE(distribution.kept_outcomes.back().probability == 1e-20);
    distribution.kept_outcomes.push_back(distribution.kept_outcomes.back());
    canonicalize_outcome_entries(distribution.kept_outcomes);
    REQUIRE(distribution.kept_outcomes.size() == 2);
    REQUIRE(distribution.kept_outcomes.back().probability == 2e-20);
}

TEST_CASE("Unrepresentable reference probabilities fail instead of losing support",
          "[rare_probability]") {
    REQUIRE(checked_probability_product(0.0, 1e-200) == 0.0);
    REQUIRE_THROWS_WITH(checked_probability_product(1e-200, 1e-200),
                        "inconclusive reference: invalid or underflowed probability product");
}

TEST_CASE("Omitted mass bounds round upward under optimized arithmetic", "[rare_probability]") {
    const double probability = GENERATE(1e-20, 0.125);
    double bound = 0.0;
    add_omitted_probability(bound, probability);
    REQUIRE(bound > probability);
    const double previous = bound;
    add_omitted_probability(bound, 0.0);
    REQUIRE(bound == previous);
    add_omitted_probability(bound, probability);
    REQUIRE(bound > 2.0 * probability);
}

TEST_CASE("Rare probe clusters retain conditional expectation checks", "[rare_probability]") {
    auto reference = rare_distribution(1e-20);
    auto actual = reference;
    // A bridge joins the two endpoints in the tolerant probe cluster. Its
    // negligible weight must not hide the different conditional means.
    reference.kept_outcomes.back().probability *= 1.0 - 1e-6;
    reference.kept_outcomes.push_back({{1, 0, 0, {-1.0 + 0.75e-9}}, 1e-26});
    actual.kept_outcomes.back().outcome.exp_vals[0] += 1.5e-9;
    std::string detail;
    REQUIRE_FALSE(distributions_equal(reference, actual, 1e-9, &detail));
    REQUIRE(detail.find("exp_val[0] mismatch") != std::string::npos);
}

TEST_CASE("Exact comparison rejects invalid distributions even when both agree",
          "[rare_probability]") {
    auto invalid = rare_distribution(1e-20);
    SECTION("nonfinite probability") {
        invalid.kept_outcomes.back().probability = opaque_nan();
    }
    SECTION("infinite conditional expectation") {
        invalid.kept_outcomes.back().outcome.exp_vals[0] = opaque_infinity();
    }
    SECTION("nonfinite weighted expectation") {
        invalid.kept_exp_vals[0] = opaque_nan();
    }
    SECTION("negative probability") {
        invalid.kept_outcomes.back().probability = -1e-20;
    }
    SECTION("normalization") {
        invalid.kept_outcomes.front().probability = 0.9;
    }
    SECTION("unsupported record bit") {
        invalid.kept_outcomes.back().outcome.visible_records = 2;
    }
    SECTION("invalid probe dimension") {
        invalid.kept_outcomes.back().outcome.exp_vals.clear();
    }
    SECTION("impossible conditional expectation") {
        invalid.kept_outcomes.back().outcome.exp_vals[0] = 1.5;
    }
    REQUIRE_FALSE(distributions_equal(invalid, invalid));
}

TEST_CASE("Dense oracle and replay retain rare conditional and joint Born probabilities",
          "[rare_probability]") {
    const auto circuit = parse("R_Y(1e-8) 0 1\nM 0 1\nEXP_VAL Z0\nEXP_VAL Z1\n");
    // Stable closed form: no subtraction of near-equal numbers to obtain p.
    const long double amplitude = std::sin(1e-8L * std::numbers::pi_v<long double> / 2);
    const double p = static_cast<double>(amplitude * amplitude);
    REQUIRE(p > kMeasurementDustEpsilon);
    REQUIRE(p < 1e-15);
    ExactPlanDistribution analytic;
    analytic.num_visible_records = 2;
    analytic.num_exp_vals = 2;
    analytic.kept_exp_vals = {1.0 - 2 * p, 1.0 - 2 * p};
    for (uint64_t bits = 0; bits < 4; ++bits) {
        const bool first = (bits & 1) != 0;
        const bool second = (bits & 2) != 0;
        analytic.kept_outcomes.push_back({{bits, 0, 0, {first ? -1.0 : 1.0, second ? -1.0 : 1.0}},
                                          (first ? p : 1 - p) * (second ? p : 1 - p)});
    }
    const ExactComparisonBudget budget{
        .probability_relative = 1e-9,
        .expectation_absolute = 1e-12,
        .normalization_absolute = 1e-12,
        .max_omitted_probability = p * p * 1e-10,
    };
    require_distributions_equal(analytic, compute_dense_circuit_distribution(circuit), budget);
    require_distributions_equal(analytic, compute_exact_hir_distribution(trace(circuit)), budget);
}

TEST_CASE("Dense noise and readout references preserve tiny supported outcomes",
          "[rare_probability]") {
    const std::string prefix = GENERATE(
        "X_ERROR(1e-20) 0\nM 0\n", "PAULI_CHANNEL_1(4e-21, 6e-21, 0) 0\nM 0\n",
        "CORRELATED_ERROR(4e-21) X0\nELSE_CORRELATED_ERROR(6e-21) X0\nM 0\n", "M(1e-20) 0\n");
    auto analytic = rare_distribution(1e-20);
    analytic.num_exp_vals = 0;
    analytic.kept_exp_vals.clear();
    for (auto& entry : analytic.kept_outcomes)
        entry.outcome.exp_vals.clear();
    const ExactComparisonBudget budget{.max_omitted_probability = 1e-30};
    const auto circuit = parse(prefix);
    require_distributions_equal(analytic, compute_dense_circuit_distribution(circuit), budget);
    require_distributions_equal(analytic, compute_exact_hir_distribution(trace(circuit)), budget);
}

TEST_CASE("Rare discarded probabilities use the branch relative contract", "[rare_probability]") {
    const auto circuit = parse("X_ERROR(1e-20) 0\nM 0\nDETECTOR rec[-1]\n");
    const uint8_t postselection[] = {1};
    const sampling::SamplingPlanOptions options{.postselection_mask = postselection};
    auto reference = compute_dense_circuit_distribution(circuit, options);
    REQUIRE(reference.discarded_probability == 1e-20);
    const ExactComparisonBudget budget{.max_omitted_probability = 1e-30};
    require_distributions_equal(reference, compute_exact_hir_distribution(trace(circuit), options),
                                budget);
    auto actual = reference;
    actual.discarded_probability = 0;
    REQUIRE_FALSE(distributions_equal(reference, actual, budget));
}

TEST_CASE("Dense reference reports conditional pruning as omitted joint mass",
          "[rare_probability]") {
    const auto circuit = parse("H 0\nM 0\nR_Y(1e-8) 1\nM 1\n");
    const double amplitude = std::sin(1e-8 * std::numbers::pi / 2);
    const double p = amplitude * amplitude;
    const auto retained = compute_dense_circuit_distribution(circuit, {}, 0.0);
    const auto pruned = compute_dense_circuit_distribution(circuit, {}, 1e-15);
    REQUIRE(retained.kept_outcomes.size() == 4);
    REQUIRE(pruned.kept_outcomes.size() == 2);
    // Two disjoint histories of weight one half each lose conditional mass p.
    REQUIRE_THAT(pruned.omitted_probability_bound, Catch::Matchers::WithinRel(p, 1e-12));
    const ExactComparisonBudget budget{.max_omitted_probability = p * 1e-9};
    std::string detail;
    REQUIRE_FALSE(distributions_equal(pruned, pruned, budget, &detail));
    REQUIRE(detail.find("inconclusive reference") != std::string::npos);
    REQUIRE_FALSE(distributions_equal(retained, pruned, budget));
}

TEST_CASE("Reference precision policies distinguish zero dust and supported probabilities",
          "[rare_probability]") {
    const double cutoff =
        GENERATE(kMeasurementDustEpsilonFor<double>, kMeasurementDustEpsilonFor<float>);
    for (double probability : {0.0, cutoff / 4, cutoff * 4}) {
        CAPTURE(cutoff, probability);
        // Calibrate the reference cutoff independently of GPU availability.
        detail_dense_oracle::TrajectoryBranch branch;
        branch.psi = {std::sqrt(1.0 - probability), std::sqrt(probability)};
        branch.probability = 0.25;
        std::vector<detail_dense_oracle::TrajectoryBranch> children;
        double omitted = 0;
        detail_dense_oracle::branch_projective_measurement(branch, 1, {0, 1, false}, false, false,
                                                           nullptr, children, cutoff, omitted);
        REQUIRE(children.size() == (probability > cutoff ? 2 : 1));
        if (probability == 0 || probability > cutoff) {
            REQUIRE(omitted == 0.0);
        } else {
            REQUIRE_THAT(omitted, Catch::Matchers::WithinRel(0.25 * probability, 1e-12));
        }
    }
}
