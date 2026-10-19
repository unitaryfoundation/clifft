#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/phase_polynomial_pass.h"
#include "clifft/optimizer/rotation_simplification_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/xoshiro.h"

#include "dense_circuit_oracle.h"
#include "sampling_equivalence_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::test;

namespace {

HirModule parse_and_trace(const std::string& source) {
    return clifft::trace(clifft::parse(source));
}

struct OwnedPlanOptions {
    ReferenceSyndrome reference;
    std::vector<uint8_t> postselection_mask;

    [[nodiscard]] clifft::sampling::SamplingPlanOptions as_span_options() const {
        return clifft::sampling::SamplingPlanOptions{
            .postselection_mask = postselection_mask,
            .expected_detectors = reference.detectors,
            .expected_observables = reference.observables,
        };
    }
};

OwnedPlanOptions build_plan_options(const HirModule& hir, const PlantedCircuit& planted,
                                    bool postselect_checks) {
    OwnedPlanOptions owned;
    if (hir.num_detectors == 0 && hir.num_observables == 0) {
        return owned;
    }
    owned.reference = compute_reference_syndrome(hir);
    if (postselect_checks && !planted.check_detectors.empty()) {
        owned.postselection_mask.assign(hir.num_detectors, 0);
        for (uint32_t det : planted.check_detectors) {
            if (det < hir.num_detectors) {
                owned.postselection_mask[det] = 1;
            }
        }
    }
    return owned;
}

double max_visible_record_probability(const ExactPlanDistribution& dist) {
    std::vector<double> by_record(size_t{1} << dist.num_visible_records, 0.0);
    double max_p = 0.0;
    for (const ExactOutcomeEntry& entry : dist.kept_outcomes) {
        const auto idx = static_cast<size_t>(entry.outcome.visible_records);
        by_record[idx] += entry.probability;
        max_p = std::max(max_p, by_record[idx]);
    }
    return max_p;
}

void assert_noiseless_scaffold_polarization(const ExactPlanDistribution& dist,
                                            const PlantedCircuit& planted) {
    // A uniform distribution over m >= 3 visible records has max probability
    // 2^-m <= 0.125. Our interferometer scaffolds keep the modal visible record
    // probability at or above 2 * 2^-m so passes are not tested on flat,
    // maximally mixed output distributions.
    REQUIRE(dist.num_visible_records >= 3);
    const double uniform_prob = 1.0 / static_cast<double>(1ULL << dist.num_visible_records);
    REQUIRE(max_visible_record_probability(dist) >= 2.0 * uniform_prob - 1e-9);

    // Every designated stabilizer-check detector must be deterministically 0
    // in the absence of injected faults.
    for (uint32_t det : planted.check_detectors) {
        for (const ExactOutcomeEntry& entry : dist.kept_outcomes) {
            if (entry.probability > 1e-12) {
                REQUIRE(((entry.outcome.detectors >> det) & 1ULL) == 0ULL);
            }
        }
    }
}

}  // namespace

// =============================================================================
// Analytic calibration of the dense Schrodinger oracle and HIR evaluator
// =============================================================================

TEST_CASE("Dense Schrodinger oracle and HIR evaluator match closed form analytic probabilities",
          "[pass_equivalence][optimizer]") {
    SECTION("entangled Bell-pair reset marginalizes hidden record and resets only target qubit") {
        const std::string source =
            "H 0\n"
            "CX 0 1\n"
            "R 0\n"
            "CX 1 0\n"
            "M 0 1\n";
        const Circuit circuit = clifft::parse(source);
        const HirModule hir = clifft::trace(circuit);

        const ExactPlanDistribution dense = compute_dense_circuit_distribution(circuit);
        const ExactPlanDistribution exact = compute_exact_hir_distribution(hir);
        require_distributions_equal(dense, exact);

        REQUIRE(exact.kept_outcomes.size() == 2);
        REQUIRE(exact.kept_outcomes[0].outcome.visible_records == 0b00);
        REQUIRE_THAT(exact.kept_outcomes[0].probability, Catch::Matchers::WithinAbs(0.5, 1e-12));
        REQUIRE(exact.kept_outcomes[1].outcome.visible_records == 0b11);
        REQUIRE_THAT(exact.kept_outcomes[1].probability, Catch::Matchers::WithinAbs(0.5, 1e-12));
    }

    SECTION("repeated asymmetric readout noise composes before classical feedback") {
        // Qubit 0 is prepared in |1>. M(0.1) flips 1->0 with prob 0.1 (leaving
        // 0 w.p. 0.1, 1 w.p. 0.9). READOUT_NOISE(0.25, 0.4) then flips 0->1 w.p.
        // 0.25 and 1->0 w.p. 0.4, so final rec[0] is:
        //   P(rec[0] = 0) = 0.1 * 0.75 + 0.9 * 0.40 = 0.435
        //   P(rec[0] = 1) = 0.1 * 0.25 + 0.9 * 0.60 = 0.565
        // CX rec[-1] 1 copies rec[0] onto qubit 1, so only (00) and (11) occur.
        const std::string source =
            "X 0\n"
            "M(0.1) 0\n"
            "READOUT_NOISE(0.25, 0.4) rec[-1]\n"
            "CX rec[-1] 1\n"
            "M 1\n"
            "DETECTOR rec[-1] rec[-2]\n";
        const Circuit circuit = clifft::parse(source);
        const HirModule hir = clifft::trace(circuit);

        const ExactPlanDistribution dense = compute_dense_circuit_distribution(circuit);
        const ExactPlanDistribution exact = compute_exact_hir_distribution(hir);
        require_distributions_equal(dense, exact);

        REQUIRE(exact.kept_outcomes.size() == 2);
        REQUIRE(exact.kept_outcomes[0].outcome.visible_records == 0b00);
        REQUIRE(exact.kept_outcomes[0].outcome.detectors == 0);
        REQUIRE_THAT(exact.kept_outcomes[0].probability, Catch::Matchers::WithinAbs(0.435, 1e-12));
        REQUIRE(exact.kept_outcomes[1].outcome.visible_records == 0b11);
        REQUIRE(exact.kept_outcomes[1].outcome.detectors == 0);
        REQUIRE_THAT(exact.kept_outcomes[1].probability, Catch::Matchers::WithinAbs(0.565, 1e-12));
    }

    SECTION("mid-circuit postselection preserves suffix interferometer and EXP_VAL weights") {
        const std::string source =
            "H 0\n"
            "X_ERROR(0.2) 1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "T 0\n"
            "T 0\n"
            "S_DAG 0\n"
            "H 0\n"
            "EXP_VAL Z0\n"
            "M 0\n";
        const Circuit circuit = clifft::parse(source);
        const HirModule hir = clifft::trace(circuit);
        const uint8_t postselect[] = {1};
        const clifft::sampling::SamplingPlanOptions options{
            .postselection_mask = postselect,
        };

        const ExactPlanDistribution dense = compute_dense_circuit_distribution(circuit, options);
        const ExactPlanDistribution exact = compute_exact_hir_distribution(hir, options);
        require_distributions_equal(dense, exact);

        REQUIRE_THAT(exact.discarded_probability, Catch::Matchers::WithinAbs(0.2, 1e-12));
        REQUIRE_THAT(exact.kept_exp_vals[0], Catch::Matchers::WithinAbs(0.8, 1e-12));
        REQUIRE(exact.kept_outcomes.size() == 1);
        REQUIRE(exact.kept_outcomes[0].outcome.visible_records == 0b00);
        REQUIRE_THAT(exact.kept_outcomes[0].probability, Catch::Matchers::WithinAbs(0.8, 1e-12));
        REQUIRE_THAT(exact.kept_outcomes[0].outcome.exp_vals[0],
                     Catch::Matchers::WithinAbs(1.0, 1e-12));
    }
}

// =============================================================================
// Negative and mutation controls
// =============================================================================

TEST_CASE("Exact distribution checker detects correlation bugs with identical column marginals",
          "[pass_equivalence][optimizer]") {
    // Both circuits produce two 50/50 measurement columns with mean 0.5 each.
    // The first copies measurement 0 onto qubit 1 via classical feedback; the
    // second prepares qubit 1 independently in |+>. A 1-bit marginal check
    // cannot distinguish them, whereas the exact joint distribution must.
    const HirModule correlated = parse_and_trace(
        "H 0\n"
        "M 0\n"
        "CX rec[-1] 1\n"
        "M 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n");
    const HirModule independent = parse_and_trace(
        "H 0\n"
        "H 1\n"
        "M 0\n"
        "M 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n");

    const ExactPlanDistribution dist_corr = compute_exact_hir_distribution(correlated);
    const ExactPlanDistribution dist_indep = compute_exact_hir_distribution(independent);

    REQUIRE(dist_corr.kept_outcomes.size() == 2);
    REQUIRE(dist_indep.kept_outcomes.size() == 4);
    REQUIRE_FALSE(distributions_equal(dist_corr, dist_indep));
}

TEST_CASE("Exact distribution checker detects illegal feedback absorption across readout noise",
          "[pass_equivalence][optimizer]") {
    // When measurement 0 has readout noise, classical feedback CX rec[-1] 1
    // propagates the readout flip onto qubit 1 so m0 and m1 still agree in the
    // detector. Replacing the classical feedback with a pre-measurement quantum
    // CX 0 1 leaves qubit 1 unflipped when measurement 0 suffers a readout error.
    // Both circuits have identical noiseless behavior and identical 0.5 column
    // marginals under noise, so only joint/detector distribution checking catches it.
    const HirModule faithful_feedback = parse_and_trace(
        "H 0\n"
        "M(0.2) 0\n"
        "CX rec[-1] 1\n"
        "M 1\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");
    const HirModule buggy_pre_measurement_cx = parse_and_trace(
        "H 0\n"
        "CX 0 1\n"
        "M(0.2) 0\n"
        "M 1\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n");

    const ExactPlanDistribution dist_faithful = compute_exact_hir_distribution(faithful_feedback);
    const ExactPlanDistribution dist_buggy =
        compute_exact_hir_distribution(buggy_pre_measurement_cx);

    REQUIRE_FALSE(distributions_equal(dist_faithful, dist_buggy));
}

TEST_CASE("Exact distribution checker detects a single rotation sign flip inside an interferometer",
          "[pass_equivalence][optimizer]") {
    const HirModule original = parse_and_trace(
        "H 0\n"
        "T 0\n"
        "T 0\n"
        "S_DAG 0\n"
        "H 0\n"
        "EXP_VAL Z0\n"
        "M 0\n"
        "DETECTOR rec[-1]\n");
    HirModule mutated = original;
    REQUIRE(mutated.ops.size() >= 2);
    REQUIRE(mutated.ops[0].op_type() == OpType::T_GATE);
    mutated.set_sign(mutated.ops[0], !mutated.sign(mutated.ops[0]));

    const ExactPlanDistribution dist_orig = compute_exact_hir_distribution(original);
    const ExactPlanDistribution dist_mut = compute_exact_hir_distribution(mutated);

    REQUIRE_FALSE(distributions_equal(dist_orig, dist_mut));
}

TEST_CASE("Exact distribution checker detects a dropped logical noise prefix across Pauli noise",
          "[pass_equivalence][optimizer]") {
    // Moving T 0 before X_ERROR(0.25) 0 without recording logical_noise_prefix = 0
    // drops the conditional sign flip when X_ERROR fires. The noiseless branch
    // is unchanged, but the noisy distribution diverges.
    const HirModule original = parse_and_trace(
        "H 0\n"
        "X_ERROR(0.25) 0\n"
        "T 0\n"
        "T 0\n"
        "S_DAG 0\n"
        "H 0\n"
        "M 0\n");

    HirModule buggy_reorder = original;
    REQUIRE(buggy_reorder.ops.size() >= 3);
    REQUIRE(buggy_reorder.ops[0].op_type() == OpType::NOISE);
    REQUIRE(buggy_reorder.ops[1].op_type() == OpType::T_GATE);
    // Swap T 0 across X_ERROR(0.25) 0 without initializing logical_noise_prefix.
    std::swap(buggy_reorder.ops[0], buggy_reorder.ops[1]);

    const ExactPlanDistribution dist_orig = compute_exact_hir_distribution(original);
    const ExactPlanDistribution dist_buggy = compute_exact_hir_distribution(buggy_reorder);

    REQUIRE_FALSE(distributions_equal(dist_orig, dist_buggy));
}

TEST_CASE("Exact distribution checker detects illegal measurement movement across EXP_VAL",
          "[pass_equivalence][optimizer]") {
    // Before M 0, qubit 0 is in |+>, so EXP_VAL Z0 is deterministically 0 on
    // both measurement branches. Moving M 0 ahead of EXP_VAL Z0 collapses the
    // state first so EXP_VAL Z0 becomes +1 on m0=0 and -1 on m0=1. The kept
    // average is 0 in both circuits, so only per-outcome EXP_VAL tracking
    // catches the illegal reorder.
    const HirModule probe_before_measure = parse_and_trace(
        "H 0\n"
        "EXP_VAL Z0\n"
        "M 0\n");
    const HirModule measure_before_probe = parse_and_trace(
        "H 0\n"
        "M 0\n"
        "EXP_VAL Z0\n");

    const ExactPlanDistribution dist_before = compute_exact_hir_distribution(probe_before_measure);
    const ExactPlanDistribution dist_after = compute_exact_hir_distribution(measure_before_probe);

    REQUIRE_THAT(dist_before.kept_exp_vals[0], Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(dist_after.kept_exp_vals[0], Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_FALSE(distributions_equal(dist_before, dist_after));

    // Even when the collapsing operation is a hidden reset R 0 on an entangled
    // partner (so no visible record bit distinguishes the two branches), the
    // distinct per-trajectory EXP_VAL values (+1 and -1 vs 0) remain separated.
    const HirModule probe_before_reset = parse_and_trace(
        "H 0\n"
        "CX 0 1\n"
        "EXP_VAL Z1\n"
        "R 0\n");
    const HirModule reset_before_probe = parse_and_trace(
        "H 0\n"
        "CX 0 1\n"
        "R 0\n"
        "EXP_VAL Z1\n");

    REQUIRE_FALSE(distributions_equal(compute_exact_hir_distribution(probe_before_reset),
                                      compute_exact_hir_distribution(reset_before_probe)));
}

TEST_CASE("Exact distribution checker detects record reset readout and postselection bugs",
          "[pass_equivalence][optimizer]") {
    SECTION("wrong detector record reference across hidden reset") {
        const HirModule correct_ref = parse_and_trace(
            "H 0 1\n"
            "R 0\n"
            "M 0 1\n"
            "DETECTOR rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n");
        const HirModule wrong_ref = parse_and_trace(
            "H 0 1\n"
            "R 0\n"
            "M 0 1\n"
            "DETECTOR rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n");
        REQUIRE_FALSE(distributions_equal(compute_exact_hir_distribution(correct_ref),
                                          compute_exact_hir_distribution(wrong_ref)));
    }

    SECTION("dropped conditional reset correction after hidden reset") {
        const HirModule faithful_reset = parse_and_trace(
            "H 0\n"
            "CX 0 1\n"
            "R 0\n"
            "M 0 1\n");
        HirModule dropped_reset_correction = faithful_reset;
        // Erase the CONDITIONAL_PAULI operation that flips qubit 0 back to |0>
        // when the hidden reset measurement yields 1.
        std::erase_if(dropped_reset_correction.ops, [](const HeisenbergOp& op) {
            return op.op_type() == OpType::CONDITIONAL_PAULI;
        });
        REQUIRE(dropped_reset_correction.ops.size() + 1 == faithful_reset.ops.size());
        REQUIRE_FALSE(
            distributions_equal(compute_exact_hir_distribution(faithful_reset),
                                compute_exact_hir_distribution(dropped_reset_correction)));
    }

    SECTION("swapped asymmetric readout noise probabilities") {
        const HirModule asymmetric_a = parse_and_trace(
            "H 0\n"
            "T 0\n"
            "H 0\n"
            "M 0\n"
            "READOUT_NOISE(0.3, 0.05) rec[-1]\n");
        const HirModule asymmetric_b = parse_and_trace(
            "H 0\n"
            "T 0\n"
            "H 0\n"
            "M 0\n"
            "READOUT_NOISE(0.05, 0.3) rec[-1]\n");
        REQUIRE_FALSE(distributions_equal(compute_exact_hir_distribution(asymmetric_a),
                                          compute_exact_hir_distribution(asymmetric_b)));
    }

    SECTION("missing postselection flag on check detector") {
        const HirModule hir = parse_and_trace(
            "H 0\n"
            "Z_ERROR(0.25) 0\n"
            "H 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n");
        const uint8_t postselected_mask[] = {1};
        const clifft::sampling::SamplingPlanOptions with_postselection{
            .postselection_mask = postselected_mask,
        };
        const clifft::sampling::SamplingPlanOptions without_postselection{};
        REQUIRE_FALSE(
            distributions_equal(compute_exact_hir_distribution(hir, with_postselection),
                                compute_exact_hir_distribution(hir, without_postselection)));
    }
}

TEST_CASE("Exact distribution checker handles near-tie probe vectors and postselected probes",
          "[pass_equivalence][optimizer]") {
    SECTION("order-independent probe matching across tolerance boundary") {
        // Coordinate 0 differs by 0.75e-9 (<= 1e-9) in dist_a and by 1.25e-9
        // (> 1e-9) in dist_b, while corresponding entries between dist_a and
        // dist_b differ by at most 0.5e-9. Comparison within the shared
        // classical-outcome group must not depend on tolerant sort order.
        ExactPlanDistribution dist_a;
        dist_a.num_visible_records = 1;
        dist_a.num_exp_vals = 2;
        dist_a.kept_exp_vals = {0.375e-9, 0.0};
        dist_a.kept_outcomes = {
            ExactOutcomeEntry{ExactShotOutcome{0, 0, 0, {0.0, 1.0}}, 0.5},
            ExactOutcomeEntry{ExactShotOutcome{0, 0, 0, {0.75e-9, -1.0}}, 0.5},
        };
        canonicalize_outcome_entries(dist_a.kept_outcomes);

        ExactPlanDistribution dist_b;
        dist_b.num_visible_records = 1;
        dist_b.num_exp_vals = 2;
        dist_b.kept_exp_vals = {0.625e-9, 0.0};
        dist_b.kept_outcomes = {
            ExactOutcomeEntry{ExactShotOutcome{0, 0, 0, {0.0, 1.0}}, 0.5},
            ExactOutcomeEntry{ExactShotOutcome{0, 0, 0, {1.25e-9, -1.0}}, 0.5},
        };
        canonicalize_outcome_entries(dist_b.kept_outcomes);

        require_distributions_equal(dist_a, dist_b);
    }

    SECTION("postselection ignores unobservable EXP_VAL differences on rejected shots") {
        // When DETECTOR rec[-1] is postselected to 0, only the m0=0 survivor
        // (where qubit 0 is already in |0>) is observable. Inserting R 0 before
        // EXP_VAL Z0 changes the rejected m0=1 branch from -1 to +1 without
        // affecting any surviving trajectory.
        const Circuit without_reset = clifft::parse(
            "H 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "EXP_VAL Z0\n");
        const Circuit with_reset = clifft::parse(
            "H 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "R 0\n"
            "EXP_VAL Z0\n");
        const HirModule hir_without_reset = clifft::trace(without_reset);
        const HirModule hir_with_reset = clifft::trace(with_reset);

        const uint8_t postselected_mask[] = {1};
        const clifft::sampling::SamplingPlanOptions postselected{
            .postselection_mask = postselected_mask,
        };
        require_distributions_equal(compute_exact_hir_distribution(hir_without_reset, postselected),
                                    compute_exact_hir_distribution(hir_with_reset, postselected));
        require_distributions_equal(compute_dense_circuit_distribution(without_reset, postselected),
                                    compute_dense_circuit_distribution(with_reset, postselected));

        // Without postselection, the m0=1 branch is kept and the two circuits
        // must be distinguished.
        REQUIRE_FALSE(distributions_equal(compute_exact_hir_distribution(hir_without_reset),
                                          compute_exact_hir_distribution(hir_with_reset)));
    }
}

// =============================================================================
// Paired positive + boundary-hazard barrier verification
// =============================================================================

TEST_CASE("Paired scaffolds verify rewrites fire when legal and stop at boundary hazards",
          "[pass_equivalence][optimizer][schedule_pass]") {
    SECTION(
        "PeepholeFusionPass fuses across commuting ops but stops at anticommuting EXP_VAL or "
        "reset") {
        const std::string legal_source =
            "H 0\n"
            "T 0\n"
            "Z 0\n"
            "T 0\n"
            "H 0\n"
            "M 0\n";
        const Circuit legal_circuit = clifft::parse(legal_source);
        HirModule legal_hir = clifft::trace(legal_circuit);
        PeepholeFusionPass legal_pass;
        legal_pass.run(legal_hir);
        require_distributions_equal(compute_dense_circuit_distribution(legal_circuit),
                                    compute_exact_hir_distribution(legal_hir));
        REQUIRE(legal_pass.fusions() == 1);

        const std::string probe_barrier_source =
            "H 0\n"
            "T 0\n"
            "EXP_VAL Y0\n"
            "T 0\n"
            "H 0\n"
            "M 0\n";
        const Circuit probe_circuit = clifft::parse(probe_barrier_source);
        HirModule probe_hir = clifft::trace(probe_circuit);
        PeepholeFusionPass probe_pass;
        probe_pass.run(probe_hir);
        require_distributions_equal(compute_dense_circuit_distribution(probe_circuit),
                                    compute_exact_hir_distribution(probe_hir));
        REQUIRE(probe_pass.fusions() == 0);

        const std::string reset_barrier_source =
            "H 0\n"
            "T 0\n"
            "R 0\n"
            "H 0\n"
            "T 0\n"
            "H 0\n"
            "M 0\n";
        const Circuit reset_circuit = clifft::parse(reset_barrier_source);
        HirModule reset_hir = clifft::trace(reset_circuit);
        PeepholeFusionPass reset_pass;
        reset_pass.run(reset_hir);
        require_distributions_equal(compute_dense_circuit_distribution(reset_circuit),
                                    compute_exact_hir_distribution(reset_hir));
        REQUIRE(reset_pass.fusions() == 0);
    }

    SECTION(
        "RotationSimplificationPass collapses ancilla phase gadget on |0> but not after RX or "
        "X_ERROR") {
        // When qubit 1 is in |0>, CX 0 1; T 1; CX 0 1 applies T on qubit 0 and
        // cancels against T_DAG 0.
        const std::string legal_source =
            "H 0\n"
            "CX 0 1\n"
            "T 1\n"
            "CX 0 1\n"
            "T_DAG 0\n"
            "H 0\n"
            "M 0 1\n";
        const Circuit legal_circuit = clifft::parse(legal_source);
        HirModule legal_hir = clifft::trace(legal_circuit);
        RotationSimplificationPass legal_pass;
        legal_pass.run(legal_hir);
        require_distributions_equal(compute_dense_circuit_distribution(legal_circuit),
                                    compute_exact_hir_distribution(legal_hir));
        REQUIRE(legal_pass.rotations_removed() == 2);

        // Preparing qubit 1 in |+> via RX 1 invalidates the Z1 = +1 stabilizer
        // assumption, so the two rotations must not cancel.
        const std::string rx_barrier_source =
            "RX 1\n"
            "H 0\n"
            "CX 0 1\n"
            "T 1\n"
            "CX 0 1\n"
            "T_DAG 0\n"
            "H 0\n"
            "M 0 1\n";
        const Circuit rx_circuit = clifft::parse(rx_barrier_source);
        HirModule rx_hir = clifft::trace(rx_circuit);
        RotationSimplificationPass rx_pass;
        rx_pass.run(rx_hir);
        require_distributions_equal(compute_dense_circuit_distribution(rx_circuit),
                                    compute_exact_hir_distribution(rx_hir));
        REQUIRE(rx_pass.rotations_removed() == 0);

        // Injecting X_ERROR(0.2) 1 before the gadget also invalidates the static
        // Z1 = +1 stabilizer assumption across trajectories.
        const std::string noise_barrier_source =
            "X_ERROR(0.2) 1\n"
            "H 0\n"
            "CX 0 1\n"
            "T 1\n"
            "CX 0 1\n"
            "T_DAG 0\n"
            "H 0\n"
            "M 0 1\n";
        const Circuit noise_circuit = clifft::parse(noise_barrier_source);
        HirModule noise_hir = clifft::trace(noise_circuit);
        RotationSimplificationPass noise_pass;
        noise_pass.run(noise_hir);
        require_distributions_equal(compute_dense_circuit_distribution(noise_circuit),
                                    compute_exact_hir_distribution(noise_hir));
        REQUIRE(noise_pass.rotations_removed() == 0);
    }

    SECTION("StatevectorSqueezePass bubbles independent measurement but stops at EXP_VAL barrier") {
        const std::string legal_source =
            "H 0\n"
            "T 0\n"
            "H 1\n"
            "T 1\n"
            "H 0\n"
            "M 0\n"
            "H 1\n"
            "M 1\n";
        const Circuit legal_circuit = clifft::parse(legal_source);
        const HirModule legal_orig = clifft::trace(legal_circuit);
        HirModule legal_squeezed = legal_orig;
        StatevectorSqueezePass squeeze;
        squeeze.run(legal_squeezed);
        require_distributions_equal(compute_dense_circuit_distribution(legal_circuit),
                                    compute_exact_hir_distribution(legal_squeezed));
        REQUIRE(clifft::sampling::plan_sampling(legal_orig).peak_active_width == 2);
        REQUIRE(clifft::sampling::plan_sampling(legal_squeezed).peak_active_width == 1);

        const std::string barrier_source =
            "H 0\n"
            "T 0\n"
            "H 1\n"
            "T 1\n"
            "EXP_VAL Z0*Z1\n"
            "H 0\n"
            "M 0\n"
            "H 1\n"
            "M 1\n";
        const Circuit barrier_circuit = clifft::parse(barrier_source);
        HirModule barrier_squeezed = clifft::trace(barrier_circuit);
        squeeze.run(barrier_squeezed);
        require_distributions_equal(compute_dense_circuit_distribution(barrier_circuit),
                                    compute_exact_hir_distribution(barrier_squeezed));
        REQUIRE(clifft::sampling::plan_sampling(barrier_squeezed).peak_active_width == 2);
    }
}

// =============================================================================
// Planted-structure and boundary-hazard exact equivalence for individual passes
// =============================================================================

namespace {

void assert_requested_hazards_present(const PlantedCircuit& planted, const HazardConfig& hazards,
                                      const HirModule& hir) {
    REQUIRE(!planted.hazard_slots.empty());
    if (hazards.pauli_noise || hazards.correlated_noise) {
        REQUIRE(!hir.noise_sites.empty());
    }
    if (hazards.readout_noise) {
        REQUIRE(!hir.readout_noise.empty());
    }
    if (hazards.resets) {
        REQUIRE(hir.num_hidden_measurements > 0);
    }
    if (hazards.exp_val) {
        REQUIRE(hir.num_exp_vals > 0);
    }
}

}  // namespace

TEST_CASE("PeepholeFusionPass preserves exact distributions across planted scaffolds and hazards",
          "[pass_equivalence][optimizer]") {
    clifft::Xoshiro256PlusPlus rng(0x578001);
    clifft::Xoshiro256PlusPlus control_rng(0x578002);

    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true},
        {.correlated_noise = true},
        {.readout_noise = true},
        {.resets = true, .exp_val = true},
        {.pauli_noise = true, .readout_noise = true, .resets = true, .exp_val = true},
    };
    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::PhaseSandwich,
        PlantedScaffoldKind::FeedbackAndReset,
    };

    for (PlantedScaffoldKind scaffold : scaffolds) {
        size_t scaffold_rewrites = 0;
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const HazardConfig& hazards = hazard_configs[h_idx];
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), h_idx, trial, source);

                const Circuit circuit = clifft::parse(source);
                const HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                HirModule optimized = original;
                PeepholeFusionPass pass;
                pass.run(optimized);

                scaffold_rewrites += pass.fusions() + pass.cancellations();

                const OwnedPlanOptions owned = build_plan_options(original, planted, false);
                const auto options = owned.as_span_options();
                const ExactPlanDistribution dense_dist =
                    compute_dense_circuit_distribution(circuit, options);
                const ExactPlanDistribution orig_dist =
                    compute_exact_hir_distribution(original, options);
                const ExactPlanDistribution opt_dist =
                    compute_exact_hir_distribution(optimized, options);

                if (h_idx == 0) {
                    assert_noiseless_scaffold_polarization(dense_dist, planted);
                }
                require_distributions_equal(dense_dist, orig_dist);
                require_distributions_equal(dense_dist, opt_dist);
                check_exact_equivalent(original, optimized, control_rng, 16, options);
            }
        }
        REQUIRE(scaffold_rewrites >= 8);
    }
}

TEST_CASE("PhasePolynomialPass preserves exact distributions across planted scaffolds and hazards",
          "[pass_equivalence][optimizer]") {
    clifft::Xoshiro256PlusPlus rng(0x578011);
    clifft::Xoshiro256PlusPlus control_rng(0x578012);

    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true},
        {.correlated_noise = true},
        {.readout_noise = true},
        {.resets = true, .exp_val = true},
        {.pauli_noise = true, .readout_noise = true, .exp_val = true},
    };
    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::StabilizerReducedPhase,
        PlantedScaffoldKind::PhaseSandwich,
    };

    for (PlantedScaffoldKind scaffold : scaffolds) {
        size_t scaffold_reduced_blocks = 0;
        size_t scaffold_t_saved = 0;
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const HazardConfig& hazards = hazard_configs[h_idx];
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), h_idx, trial, source);

                const Circuit circuit = clifft::parse(source);
                const HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                HirModule optimized = original;
                PhasePolynomialPass pass;
                pass.run(optimized);

                scaffold_reduced_blocks += pass.blocks_reduced();
                if (pass.input_t_count() > pass.output_t_count()) {
                    scaffold_t_saved += pass.input_t_count() - pass.output_t_count();
                }

                const OwnedPlanOptions owned = build_plan_options(original, planted, false);
                const auto options = owned.as_span_options();
                const ExactPlanDistribution dense_dist =
                    compute_dense_circuit_distribution(circuit, options);
                const ExactPlanDistribution orig_dist =
                    compute_exact_hir_distribution(original, options);
                const ExactPlanDistribution opt_dist =
                    compute_exact_hir_distribution(optimized, options);

                if (h_idx == 0) {
                    assert_noiseless_scaffold_polarization(dense_dist, planted);
                }
                require_distributions_equal(dense_dist, orig_dist);
                require_distributions_equal(dense_dist, opt_dist);
                check_exact_equivalent(original, optimized, control_rng, 16, options);
            }
        }
        REQUIRE(scaffold_reduced_blocks >= 6);
        REQUIRE(scaffold_t_saved >= 8);
    }
}

TEST_CASE(
    "RotationSimplificationPass preserves exact distributions across planted scaffolds and hazards",
    "[pass_equivalence][optimizer]") {
    clifft::Xoshiro256PlusPlus rng(0x578021);
    clifft::Xoshiro256PlusPlus control_rng(0x578022);

    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true},
        {.correlated_noise = true},
        {.readout_noise = true},
        {.resets = true, .exp_val = true},
        {.pauli_noise = true, .readout_noise = true, .postselect_check_detectors = true},
    };
    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::StabilizerReducedPhase,
        PlantedScaffoldKind::FeedbackAndReset,
    };

    for (PlantedScaffoldKind scaffold : scaffolds) {
        size_t scaffold_regions = 0;
        size_t scaffold_removed = 0;
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const HazardConfig& hazards = hazard_configs[h_idx];
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), h_idx, trial, source);

                const Circuit circuit = clifft::parse(source);
                const HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                HirModule optimized = original;
                RotationSimplificationPass pass;
                pass.run(optimized);

                scaffold_regions += pass.regions_reduced();
                scaffold_removed += pass.rotations_removed();

                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                const ExactPlanDistribution dense_dist =
                    compute_dense_circuit_distribution(circuit, options);
                const ExactPlanDistribution orig_dist =
                    compute_exact_hir_distribution(original, options);
                const ExactPlanDistribution opt_dist =
                    compute_exact_hir_distribution(optimized, options);

                if (h_idx == 0) {
                    assert_noiseless_scaffold_polarization(dense_dist, planted);
                }
                require_distributions_equal(dense_dist, orig_dist);
                require_distributions_equal(dense_dist, opt_dist);
                check_exact_equivalent(original, optimized, control_rng, 16, options);
            }
        }
        REQUIRE(scaffold_regions >= 4);
        REQUIRE(scaffold_removed >= 4);
    }
}

TEST_CASE("Squeeze and schedule passes preserve exact distributions across hazards",
          "[pass_equivalence][schedule_pass]") {
    clifft::Xoshiro256PlusPlus rng(0x578031);
    clifft::Xoshiro256PlusPlus control_rng(0x578032);

    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true},
        {.correlated_noise = true},
        {.readout_noise = true},
        {.resets = true, .exp_val = true},
        {.pauli_noise = true, .readout_noise = true, .postselect_check_detectors = true},
    };
    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::MultiRoundSchedule,
        PlantedScaffoldKind::FeedbackAndReset,
    };

    size_t schedule_applied = 0;
    size_t noise_crossed = 0;
    size_t width_reductions = 0;

    for (PlantedScaffoldKind scaffold : scaffolds) {
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const HazardConfig& hazards = hazard_configs[h_idx];
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), h_idx, trial, source);

                const Circuit circuit = clifft::parse(source);
                const HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                const ExactPlanDistribution dense_dist =
                    compute_dense_circuit_distribution(circuit, options);
                if (h_idx == 0) {
                    assert_noiseless_scaffold_polarization(dense_dist, planted);
                }
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(original, options));

                const uint32_t unscheduled_peak =
                    clifft::sampling::plan_sampling(original, options).peak_active_width;

                HirModule squeezed = original;
                StatevectorSqueezePass squeeze;
                squeeze.run(squeezed);
                check_exact_equivalent(original, squeezed, control_rng, 16, options);
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(squeezed, options));

                HirModule scheduled = squeezed;
                ActiveWidthSchedulePass scheduler;
                scheduler.run(scheduled);
                const uint32_t scheduled_peak =
                    clifft::sampling::plan_sampling(scheduled, options).peak_active_width;

                schedule_applied += scheduler.applied() ? 1 : 0;
                noise_crossed += crossed_noise(scheduled) ? 1 : 0;
                width_reductions += (scheduled_peak < unscheduled_peak) ? 1 : 0;

                check_exact_equivalent(original, scheduled, control_rng, 16, options);
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(scheduled, options));
            }
        }
    }

    REQUIRE(schedule_applied >= 10);
    REQUIRE(noise_crossed >= 6);
    REQUIRE(width_reductions >= 10);
}

TEST_CASE(
    "Full optimizer and scheduler pipeline preserves exact distributions across planted corpus",
    "[pass_equivalence][optimizer][schedule_pass]") {
    clifft::Xoshiro256PlusPlus rng(0x578041);
    clifft::Xoshiro256PlusPlus control_rng(0x578042);

    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::PhaseSandwich,
        PlantedScaffoldKind::StabilizerReducedPhase,
        PlantedScaffoldKind::FeedbackAndReset,
        PlantedScaffoldKind::MultiRoundSchedule,
    };
    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true, .readout_noise = true},
        {.correlated_noise = true, .resets = true},
        {.pauli_noise = true,
         .readout_noise = true,
         .exp_val = true,
         .postselect_check_detectors = true},
    };

    size_t width_improvements = 0;

    for (PlantedScaffoldKind scaffold : scaffolds) {
        size_t scaffold_width_improvements = 0;
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const HazardConfig& hazards = hazard_configs[h_idx];
            for (int trial = 0; trial < 3; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), h_idx, trial, source);

                const Circuit circuit = clifft::parse(source);
                const HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                const ExactPlanDistribution dense_dist =
                    compute_dense_circuit_distribution(circuit, options);
                if (h_idx == 0) {
                    assert_noiseless_scaffold_polarization(dense_dist, planted);
                }
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(original, options));

                const uint32_t base_peak =
                    clifft::sampling::plan_sampling(original, options).peak_active_width;

                // Step-by-step pass bisection across every equivalence-preserving
                // pass in kRegisteredPasses (both default-enabled and optional):
                // verify against the independent dense Schrodinger oracle after
                // each individual pass so any divergence is immediately localized.
                HirModule staged = original;
                for (const auto& info : kRegisteredPasses) {
                    if (info.name == "RemoveNoisePass" || info.name == "DropNonUnitaryPass") {
                        continue;
                    }
                    info.make()->run(staged);
                    CAPTURE(info.name);
                    require_distributions_equal(dense_dist,
                                                compute_exact_hir_distribution(staged, options));
                }

                HirModule optimized = original;
                HirPassManager pm = default_hir_pass_manager();
                pm.add_pass(std::make_unique<ActiveWidthSchedulePass>());
                pm.run(optimized);

                const uint32_t opt_peak =
                    clifft::sampling::plan_sampling(optimized, options).peak_active_width;
                if (opt_peak < base_peak) {
                    ++width_improvements;
                    ++scaffold_width_improvements;
                }

                check_exact_equivalent(original, optimized, control_rng, 16, options);
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(optimized, options));
            }
        }
        REQUIRE(scaffold_width_improvements >= 3);
    }

    REQUIRE(width_improvements >= 20);
}

namespace {

struct RegisteredPassEquivalenceSpec {
    std::string_view name;
    PlantedScaffoldKind primary_scaffold;
    bool (*run_and_did_rewrite)(HirModule&);
};

bool run_peephole_for_registry(HirModule& hir) {
    PeepholeFusionPass pass;
    pass.run(hir);
    return (pass.fusions() + pass.cancellations()) > 0;
}

bool run_phase_poly_for_registry(HirModule& hir) {
    PhasePolynomialPass pass;
    pass.run(hir);
    return pass.blocks_reduced() > 0;
}

bool run_rotation_simplification_for_registry(HirModule& hir) {
    RotationSimplificationPass pass;
    pass.run(hir);
    return (pass.regions_reduced() + pass.rotations_removed()) > 0;
}

bool run_squeeze_for_registry(HirModule& hir) {
    const OwnedPlanOptions before_owned = build_plan_options(hir, {}, false);
    const uint32_t before_peak =
        clifft::sampling::plan_sampling(hir, before_owned.as_span_options()).peak_active_width;
    StatevectorSqueezePass pass;
    pass.run(hir);
    const OwnedPlanOptions after_owned = build_plan_options(hir, {}, false);
    const uint32_t after_peak =
        clifft::sampling::plan_sampling(hir, after_owned.as_span_options()).peak_active_width;
    return after_peak < before_peak;
}

bool run_active_width_schedule_for_registry(HirModule& hir) {
    StatevectorSqueezePass squeeze;
    squeeze.run(hir);
    ActiveWidthSchedulePass scheduler;
    scheduler.run(hir);
    return scheduler.applied();
}

constexpr RegisteredPassEquivalenceSpec kEquivalenceVerifiedPasses[] = {
    {.name = "PeepholeFusionPass",
     .primary_scaffold = PlantedScaffoldKind::PhaseSandwich,
     .run_and_did_rewrite = run_peephole_for_registry},
    {.name = "PhasePolynomialPass",
     .primary_scaffold = PlantedScaffoldKind::PhaseSandwich,
     .run_and_did_rewrite = run_phase_poly_for_registry},
    {.name = "RotationSimplificationPass",
     .primary_scaffold = PlantedScaffoldKind::StabilizerReducedPhase,
     .run_and_did_rewrite = run_rotation_simplification_for_registry},
    {.name = "StatevectorSqueezePass",
     .primary_scaffold = PlantedScaffoldKind::MultiRoundSchedule,
     .run_and_did_rewrite = run_squeeze_for_registry},
    {.name = "ActiveWidthSchedulePass",
     .primary_scaffold = PlantedScaffoldKind::MultiRoundSchedule,
     .run_and_did_rewrite = run_active_width_schedule_for_registry},
};

// Passes whose specification intentionally alters circuit semantics (stripping
// noise or non-unitary operations) and therefore cannot preserve sampling
// distributions.
constexpr std::string_view kDestructiveRegisteredPasses[] = {
    "RemoveNoisePass",
    "DropNonUnitaryPass",
};

}  // namespace

TEST_CASE("Every registered pass is covered by planted equivalence or explicitly exempted",
          "[pass_equivalence][optimizer]") {
    for (const auto& info : kRegisteredPasses) {
        const auto* verified_it = std::find_if(
            std::begin(kEquivalenceVerifiedPasses), std::end(kEquivalenceVerifiedPasses),
            [&](const RegisteredPassEquivalenceSpec& spec) { return spec.name == info.name; });
        const bool is_verified = verified_it != std::end(kEquivalenceVerifiedPasses);
        const bool is_destructive = std::find(std::begin(kDestructiveRegisteredPasses),
                                              std::end(kDestructiveRegisteredPasses),
                                              info.name) != std::end(kDestructiveRegisteredPasses);

        INFO("Pass '" << info.name
                      << "' in kRegisteredPasses must be added to kEquivalenceVerifiedPasses in "
                         "tests/test_pass_equivalence.cc with a planted scaffold that triggers "
                         "its rewrite metric and a paired hazard-barrier SECTION (or to "
                         "kDestructiveRegisteredPasses if it intentionally alters semantics).");
        REQUIRE(is_verified != is_destructive);
    }

    clifft::Xoshiro256PlusPlus rng(0x578099);
    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true, .resets = true, .exp_val = true},
        {.readout_noise = true, .postselect_check_detectors = true},
    };

    for (const RegisteredPassEquivalenceSpec& spec : kEquivalenceVerifiedPasses) {
        CAPTURE(spec.name);
        size_t rewrites = 0;
        for (size_t h_idx = 0; h_idx < std::size(hazard_configs); ++h_idx) {
            const PlantedCircuit planted = generate_planted_scaffold(spec.primary_scaffold, rng, 4);
            const std::string source =
                join_lines(inject_boundary_hazards(planted, hazard_configs[h_idx], rng, 4));
            CAPTURE(h_idx, source);

            const Circuit circuit = clifft::parse(source);
            const HirModule original = clifft::trace(circuit);
            const OwnedPlanOptions owned = build_plan_options(
                original, planted, hazard_configs[h_idx].postselect_check_detectors);
            const auto options = owned.as_span_options();
            const ExactPlanDistribution dense_dist =
                compute_dense_circuit_distribution(circuit, options);

            HirModule optimized = original;
            if (spec.run_and_did_rewrite(optimized)) {
                ++rewrites;
            }
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(optimized, options));
        }
        INFO("Registered pass '" << spec.name << "' did not fire on its primary planted scaffold.");
        REQUIRE(rewrites > 0);
    }
}
