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
#include <span>
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

OwnedPlanOptions build_plan_options(HirModule& hir, const PlantedCircuit& planted,
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
    hir.postselection.mask = owned.postselection_mask;
    hir.postselection.expected_detectors = owned.reference.detectors;
    hir.postselection.expected_observables = owned.reference.observables;
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

    SECTION("illegal postselected lookahead across prefix-anticommuting EXP_VAL") {
        // Postselected DETECTOR rec[-1] forces the downstream M 1 outcome to 0
        // (Z1 = +1), under which T 0 and CX 1 0; T_DAG 0; CX 1 0 cancel on the
        // surviving branch. However, EXP_VAL X0 is evaluated before M 1 projects
        // Z1, when the state is still entangled across Z1 = +1 (where <X0> = 1)
        // and Z1 = -1 (where <X0> = 0), giving <X0> = 0.5. Folding the rotations
        // across EXP_VAL X0 preserves all classical measurement/detector outcomes
        // while corrupting the surviving probe expectation from 0.5 to 1.0.
        const Circuit circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "EXP_VAL X0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        const Circuit illegally_folded = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "EXP_VAL X0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        const uint8_t postselected_mask[] = {1};
        const clifft::sampling::SamplingPlanOptions postselected{
            .postselection_mask = postselected_mask,
        };
        REQUIRE_FALSE(distributions_equal(
            compute_dense_circuit_distribution(circuit, postselected),
            compute_exact_hir_distribution(clifft::trace(illegally_folded), postselected)));
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

    SECTION(
        "PhasePolynomialPass recovers postselected constraints when legal and stops at boundary "
        "hazards") {
        const uint8_t postselected_mask[] = {1};
        const clifft::sampling::SamplingPlanOptions postselected{
            .postselection_mask = postselected_mask,
        };

        // Positive 1: postselected lookahead crosses a prefix-commuting EXP_VAL Z0
        // probe and folds T 0 against CX 1 0; T_DAG 0; CX 1 0 under Z1 = +1.
        const Circuit commuting_probe_circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "EXP_VAL Z0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        HirModule commuting_probe_hir = clifft::trace(commuting_probe_circuit);
        commuting_probe_hir.postselection.mask = {1};
        PhasePolynomialPass commuting_probe_pass;
        commuting_probe_pass.run(commuting_probe_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(commuting_probe_circuit, postselected),
            compute_exact_hir_distribution(commuting_probe_hir, postselected));
        REQUIRE(commuting_probe_pass.blocks_reduced() == 1);
        REQUIRE(commuting_probe_pass.output_t_count() == 0);

        // Positive 2: upstream postselected measurement M 1 unit-propagates
        // through a 2-record difference detector DETECTOR rec[-3] rec[-1] to
        // recover the second-round stabilizer and fold the phase gadget.
        const uint8_t postselected_both_mask[] = {1, 1};
        const clifft::sampling::SamplingPlanOptions postselected_both{
            .postselection_mask = postselected_both_mask,
        };
        const Circuit unit_prop_circuit = clifft::parse(
            "RX 0\n"
            "R 1 2\n"
            "CX 0 1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "DEPOLARIZE1(0.1) 0 1 2\n"
            "CX 0 2\n"
            "T 0\n"
            "T_DAG 2\n"
            "CX 0 2\n"
            "MX 0\n"
            "M 2\n"
            "DETECTOR rec[-3] rec[-1]\n");
        HirModule unit_prop_hir = clifft::trace(unit_prop_circuit);
        unit_prop_hir.postselection.mask = {1, 1};
        PhasePolynomialPass unit_prop_pass;
        unit_prop_pass.run(unit_prop_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(unit_prop_circuit, postselected_both),
            compute_exact_hir_distribution(unit_prop_hir, postselected_both));
        REQUIRE(unit_prop_pass.blocks_reduced() == 1);
        REQUIRE(unit_prop_pass.output_t_count() == 0);

        // Negative 1: prefix-anticommuting EXP_VAL X0 blocks lookahead to M 1.
        const Circuit anticommuting_probe_circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "EXP_VAL X0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        HirModule anticommuting_probe_hir = clifft::trace(anticommuting_probe_circuit);
        anticommuting_probe_hir.postselection.mask = {1};
        PhasePolynomialPass anticommuting_probe_pass;
        anticommuting_probe_pass.run(anticommuting_probe_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(anticommuting_probe_circuit, postselected),
            compute_exact_hir_distribution(anticommuting_probe_hir, postselected));
        REQUIRE(anticommuting_probe_pass.blocks_reduced() == 0);
        REQUIRE(anticommuting_probe_pass.output_t_count() == 2);

        // Negative 2: readout noise on M(0.2) 1 decouples the postselected
        // detector bit from the pre-measurement Z1 eigenvalue.
        const Circuit readout_noise_circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "M(0.2) 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        HirModule readout_noise_hir = clifft::trace(readout_noise_circuit);
        readout_noise_hir.postselection.mask = {1};
        PhasePolynomialPass readout_noise_pass;
        readout_noise_pass.run(readout_noise_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(readout_noise_circuit, postselected),
            compute_exact_hir_distribution(readout_noise_hir, postselected));
        REQUIRE(readout_noise_pass.blocks_reduced() == 0);
        REQUIRE(readout_noise_pass.output_t_count() == 2);

        // Negative 3: anticommuting Pauli noise X_ERROR(0.2) 1 between the phase
        // block and M 1 prevents lookahead from pulling Z1 = +1 back to entry.
        const Circuit pauli_noise_circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "X_ERROR(0.2) 1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        HirModule pauli_noise_hir = clifft::trace(pauli_noise_circuit);
        pauli_noise_hir.postselection.mask = {1};
        PhasePolynomialPass pauli_noise_pass;
        pauli_noise_pass.run(pauli_noise_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(pauli_noise_circuit, postselected),
            compute_exact_hir_distribution(pauli_noise_hir, postselected));
        REQUIRE(pauli_noise_pass.blocks_reduced() == 0);
        REQUIRE(pauli_noise_pass.output_t_count() == 2);

        // Positive 3: an intervening measurement M 1 that commutes with the
        // prefix does not condition the pre-probe state across the rotations,
        // so lookahead across EXP_VAL Z1 remains valid.
        const Circuit commuting_meas_probe_circuit = clifft::parse(
            "H 0 1\n"
            "Z_ERROR(0.1) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "M 1\n"
            "EXP_VAL Z1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n");
        HirModule commuting_meas_probe_hir = clifft::trace(commuting_meas_probe_circuit);
        commuting_meas_probe_hir.postselection.mask = {1};
        PhasePolynomialPass commuting_meas_probe_pass;
        commuting_meas_probe_pass.run(commuting_meas_probe_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(commuting_meas_probe_circuit, postselected),
            compute_exact_hir_distribution(commuting_meas_probe_hir, postselected));
        REQUIRE(commuting_meas_probe_pass.blocks_reduced() == 1);
        REQUIRE(commuting_meas_probe_pass.output_t_count() == 0);

        // Negative 4: mid-circuit reset R 1 before M 1 acts as an instrument
        // barrier that stops postselected lookahead.
        const Circuit reset_barrier_circuit = clifft::parse(
            "H 0 1\n"
            "DEPOLARIZE1(0.05) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "R 1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "H 0\n"
            "M 0\n");
        HirModule reset_barrier_hir = clifft::trace(reset_barrier_circuit);
        reset_barrier_hir.postselection.mask = {1};
        PhasePolynomialPass reset_barrier_pass;
        reset_barrier_pass.run(reset_barrier_hir);
        require_distributions_equal(
            compute_dense_circuit_distribution(reset_barrier_circuit, postselected),
            compute_exact_hir_distribution(reset_barrier_hir, postselected));
        REQUIRE(reset_barrier_pass.blocks_reduced() == 0);
        REQUIRE(reset_barrier_pass.output_t_count() == 2);

        // Negative 5: even when EXP_VAL Z1 commutes with the prefix, an
        // intervening prefix-anticommuting measurement MX 0 conditions the
        // entangled state before the probe (<Z1> = 1/3 on surviving shots), so
        // lookahead must not fold the rotations across EXP_VAL Z1.
        const Circuit conditioned_probe_circuit = clifft::parse(
            "H 0 1\n"
            "Z_ERROR(0.1) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "MX 0\n"
            "EXP_VAL Z1\n"
            "M 1\n"
            "DETECTOR rec[-1]\n");
        HirModule conditioned_probe_hir = clifft::trace(conditioned_probe_circuit);
        conditioned_probe_hir.postselection.mask = {1};
        PhasePolynomialPass conditioned_probe_pass;
        conditioned_probe_pass.run(conditioned_probe_hir);
        const ExactPlanDistribution conditioned_oracle =
            compute_dense_circuit_distribution(conditioned_probe_circuit, postselected);
        require_distributions_equal(conditioned_oracle, compute_exact_hir_distribution(
                                                            conditioned_probe_hir, postselected));
        REQUIRE(conditioned_probe_pass.blocks_reduced() == 0);
        REQUIRE(conditioned_probe_pass.output_t_count() == 2);
        REQUIRE_FALSE(conditioned_oracle.kept_outcomes.empty());
        for (const ExactOutcomeEntry& entry : conditioned_oracle.kept_outcomes) {
            REQUIRE(entry.outcome.exp_vals.size() == 1);
            REQUIRE_THAT(entry.outcome.exp_vals[0], Catch::Matchers::WithinAbs(1.0 / 3.0, 1e-12));
        }

        // Reference syndrome preservation: when the noiseless trajectory lies
        // outside the postselected subspace (M 1 = 1 without X_ERROR),
        // compute_reference_syndrome() on postselection-specialized HIR must
        // still evaluate the un-postselected noiseless reference.
        const Circuit reference_subspace_circuit = clifft::parse(
            "H 0\n"
            "X 1\n"
            "X_ERROR(0.1) 1\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "MX 0\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n");
        HirModule reference_subspace_hir = clifft::trace(reference_subspace_circuit);
        reference_subspace_hir.postselection.mask = {1};
        const ReferenceSyndrome ref_before = compute_reference_syndrome(reference_subspace_hir);
        REQUIRE(ref_before.detectors == std::vector<uint8_t>{1});
        REQUIRE(ref_before.observables == std::vector<uint8_t>{1});
        PhasePolynomialPass reference_subspace_pass;
        reference_subspace_pass.run(reference_subspace_hir);
        REQUIRE(reference_subspace_pass.blocks_reduced() == 1);
        REQUIRE(reference_subspace_hir.num_t_gates() == 0);
        const ReferenceSyndrome ref_after = compute_reference_syndrome(reference_subspace_hir);
        REQUIRE(ref_after.detectors == ref_before.detectors);
        REQUIRE(ref_after.observables == ref_before.observables);

        // Coherent-rotation circuits with zero T_GATE ops must still run internal
        // simplification passes inside compute_reference_syndrome() after noise
        // stripping so cancelling PHASE_ROTATION pairs collapse to width 0.
        std::string coherent_ref_source;
        for (int q = 0; q < 24; ++q) {
            const std::string qs = std::to_string(q);
            coherent_ref_source += "H " + qs + "\n";
            coherent_ref_source += "R_Z(0.07) " + qs + "\n";
            coherent_ref_source += "X_ERROR(0.01) " + qs + "\n";
            coherent_ref_source += "R_Z(-0.07) " + qs + "\n";
            coherent_ref_source += "MX " + qs + "\n";
            coherent_ref_source += "DETECTOR rec[-1]\n";
        }
        const HirModule coherent_ref_hir = clifft::trace(clifft::parse(coherent_ref_source));
        REQUIRE(coherent_ref_hir.num_t_gates() == 0);
        const ReferenceSyndrome coherent_ref = compute_reference_syndrome(coherent_ref_hir);
        REQUIRE(coherent_ref.detectors == std::vector<uint8_t>(24, 0));

        // Cumulative specialization assumptions: optimizing successively under
        // [1, 0] and then [1, 1] (or [0, 1]) retains both detector assumptions
        // and the original un-specialized reference snapshot.
        const Circuit successive_circuit = clifft::parse(
            "H 0 2\n"
            "X 1 3\n"
            "X_ERROR(0.2) 1 3\n"
            "T 0\n"
            "CX 1 0\n"
            "T_DAG 0\n"
            "CX 1 0\n"
            "M 1\n"
            "DETECTOR rec[-1]\n"
            "T 2\n"
            "CX 3 2\n"
            "T_DAG 2\n"
            "CX 3 2\n"
            "M 3\n"
            "DETECTOR rec[-1]\n"
            "MX 0 2\n");
        for (const std::vector<uint8_t>& second_mask :
             {std::vector<uint8_t>{1, 1}, std::vector<uint8_t>{0, 1}}) {
            HirModule successive_hir = clifft::trace(successive_circuit);
            const ReferenceSyndrome successive_ref_before =
                compute_reference_syndrome(successive_hir);
            REQUIRE(successive_ref_before.detectors == std::vector<uint8_t>{1, 1});

            successive_hir.postselection.mask = {1, 0};
            PhasePolynomialPass first_pass;
            first_pass.run(successive_hir);
            REQUIRE(successive_hir.num_t_gates() == 2);

            successive_hir.postselection.mask = second_mask;
            PhasePolynomialPass second_pass;
            second_pass.run(successive_hir);
            REQUIRE(successive_hir.num_t_gates() == 0);
            REQUIRE(compute_reference_syndrome(successive_hir).detectors ==
                    successive_ref_before.detectors);

            successive_hir.postselection.mask = {1, 0};
            REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(successive_hir),
                              std::invalid_argument);
            successive_hir.postselection.mask = {0, 1};
            REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(successive_hir),
                              std::invalid_argument);

            successive_hir.postselection.mask = {1, 1};
            successive_hir.postselection.expected_detectors = {1, 0};
            REQUIRE_THROWS_AS(PhasePolynomialPass().run(successive_hir), std::invalid_argument);
            REQUIRE_THROWS_AS(clifft::sampling::plan_sampling(successive_hir),
                              std::invalid_argument);

            successive_hir.postselection.expected_detectors.clear();
            require_distributions_equal(
                compute_dense_circuit_distribution(successive_circuit, postselected_both),
                compute_exact_hir_distribution(successive_hir, postselected_both));
        }
    }

    SECTION(
        "Post-observer rotation elision fires after terminal measurement and stops at trailing "
        "EXP_VAL") {
        // When M 0 is the final observer in the module and commutes with the
        // synthesized phase polynomial on X0, PhasePolynomialPass pulls M 0
        // ahead of the phase block and elides the remaining post-observer T_DAG.
        const Circuit terminal_meas_circuit = clifft::parse(
            "H 0\n"
            "T 0\n"
            "T 0\n"
            "T 0\n"
            "M 0\n");
        HirModule terminal_meas_hir = clifft::trace(terminal_meas_circuit);
        PhasePolynomialPass terminal_meas_pass;
        terminal_meas_pass.run(terminal_meas_hir);
        require_distributions_equal(compute_dense_circuit_distribution(terminal_meas_circuit),
                                    compute_exact_hir_distribution(terminal_meas_hir));
        REQUIRE(terminal_meas_pass.blocks_reduced() == 1);
        REQUIRE(terminal_meas_pass.output_t_count() == 0);

        // Appending a trailing EXP_VAL Y0 after H 0 extends final_observer_end_
        // past the phase block so the synthesized rotation is preserved.
        const Circuit trailing_probe_circuit = clifft::parse(
            "H 0\n"
            "T 0\n"
            "T 0\n"
            "T 0\n"
            "M 0\n"
            "H 0\n"
            "EXP_VAL Y0\n");
        HirModule trailing_probe_hir = clifft::trace(trailing_probe_circuit);
        PhasePolynomialPass trailing_probe_pass;
        trailing_probe_pass.run(trailing_probe_hir);
        require_distributions_equal(compute_dense_circuit_distribution(trailing_probe_circuit),
                                    compute_exact_hir_distribution(trailing_probe_hir));
        REQUIRE(trailing_probe_pass.blocks_reduced() == 1);
        REQUIRE(trailing_probe_pass.output_t_count() == 1);
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
                HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                const OwnedPlanOptions owned = build_plan_options(original, planted, false);
                HirModule optimized = original;
                PeepholeFusionPass pass;
                pass.run(optimized);

                scaffold_rewrites += pass.fusions() + pass.cancellations();

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
        {.pauli_noise = true,
         .readout_noise = true,
         .exp_val = true,
         .postselect_check_detectors = true},
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
                HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                HirModule optimized = original;
                PhasePolynomialPass pass;
                pass.run(optimized);

                scaffold_reduced_blocks += pass.blocks_reduced();
                if (pass.input_t_count() > pass.output_t_count()) {
                    scaffold_t_saved += pass.input_t_count() - pass.output_t_count();
                }

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
                HirModule original = clifft::trace(circuit);
                assert_requested_hazards_present(planted, hazards, original);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                HirModule optimized = original;
                RotationSimplificationPass pass;
                pass.run(optimized);

                scaffold_regions += pass.regions_reduced();
                scaffold_removed += pass.rotations_removed();

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
                HirModule original = clifft::trace(circuit);
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
                HirModule original = clifft::trace(circuit);
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
                HirModule clean_unoptimized = clifft::trace(circuit);
                RemoveNoisePass().run(clean_unoptimized);
                const ExactPlanDistribution noiseless_dist =
                    compute_exact_hir_distribution(clean_unoptimized);
                auto assert_deterministic_reference = [&](const HirModule& module) {
                    const ReferenceSyndrome ref = compute_reference_syndrome(module);
                    for (uint32_t d = 0; d < module.num_detectors; ++d) {
                        uint64_t seen_mask = 0;
                        for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                            if (entry.probability > 1e-12) {
                                seen_mask |= 1ULL << ((entry.outcome.detectors >> d) & 1ULL);
                            }
                        }
                        if (seen_mask == 1ULL || seen_mask == 2ULL) {
                            const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                            REQUIRE(ref.detectors.at(d) == expected_bit);
                        }
                    }
                    for (uint32_t o = 0; o < module.num_observables; ++o) {
                        uint64_t seen_mask = 0;
                        for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                            if (entry.probability > 1e-12) {
                                seen_mask |= 1ULL << ((entry.outcome.observables >> o) & 1ULL);
                            }
                        }
                        if (seen_mask == 1ULL || seen_mask == 2ULL) {
                            const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                            REQUIRE(ref.observables.at(o) == expected_bit);
                        }
                    }
                };

                HirModule staged = original;
                for (const auto& info : kRegisteredPasses) {
                    if (info.name == "RemoveNoisePass" || info.name == "DropNonUnitaryPass") {
                        continue;
                    }
                    info.make()->run(staged);
                    CAPTURE(info.name);
                    require_distributions_equal(dense_dist,
                                                compute_exact_hir_distribution(staged, options));
                    assert_deterministic_reference(staged);
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

                assert_deterministic_reference(optimized);
                check_exact_equivalent(original, optimized, control_rng, 16, options);
                require_distributions_equal(dense_dist,
                                            compute_exact_hir_distribution(optimized, options));
            }
        }
        REQUIRE(scaffold_width_improvements >= 3);
    }

    REQUIRE(width_improvements >= 20);
}

TEST_CASE(
    "Randomized postselected circuits with interleaved measurements feedback and probes match "
    "dense oracle",
    "[pass_equivalence][optimizer][schedule_pass]") {
    clifft::Xoshiro256PlusPlus rng(0x576101);
    clifft::Xoshiro256PlusPlus control_rng(0x576102);

    static constexpr const char* kProbeAxes[] = {
        "Z0", "X0",    "Y0",    "Z1",    "X1",    "Y1",    "Z2",
        "X2", "Z0*Z1", "X0*Z1", "Y0*Z1", "X0*X1", "Z1*Z2", "X0*Z2",
    };
    static constexpr const char* kInterveningMeasurements[] = {
        "MX 0", "MY 0", "M 0", "M 1", "M 2", "MX 2", "MPP X0*Z2", "MPP Z0*Z2",
    };

    size_t witness_postselected_phase_reduced = 0;
    size_t witness_postselected_rotation_reduced = 0;
    size_t witness_intervening_meas_probe_blocked = 0;
    size_t witness_noiseless_outside_subspace = 0;

    for (int trial = 0; trial < 48; ++trial) {
        const uint32_t upstream_mode = static_cast<uint32_t>(trial % 4);
        const uint32_t phase_mode = static_cast<uint32_t>((trial / 4) % 3);
        const uint32_t window_mode = static_cast<uint32_t>(trial % 6);
        const bool flip_check_qubit = ((trial / 2) & 1) != 0;

        std::vector<std::string> lines;
        lines.push_back("H 0");
        if ((rng() & 1U) != 0) {
            lines.push_back("H 2");
        }
        if (flip_check_qubit) {
            // Put the noiseless trajectory in |1> on qubit 1 while X_ERROR
            // populates the |0> branch with probability 0.25.
            lines.push_back("X 1");
        }
        lines.push_back("X_ERROR(0.25) 1");
        if ((rng() & 1U) != 0) {
            lines.push_back("Z_ERROR(0.1) 0");
        }

        uint32_t num_measurements = 0;
        uint32_t num_detectors = 0;
        int upstream_check_meas = -1;

        // Stage 1: optional upstream measurement, detector, and feedback.
        if (upstream_mode == 1) {
            // Upstream postselected measurement on qubit 1.
            lines.push_back("M 1");
            upstream_check_meas = static_cast<int>(num_measurements++);
            lines.push_back("DETECTOR rec[-1]");
            ++num_detectors;
        } else if (upstream_mode == 2) {
            // Upstream postselected measurement followed by noise on qubit 1,
            // paired with a downstream 2-record difference detector.
            lines.push_back("M 1");
            upstream_check_meas = static_cast<int>(num_measurements++);
            lines.push_back("DETECTOR rec[-1]");
            ++num_detectors;
            lines.push_back("X_ERROR(0.2) 1");
        } else if (upstream_mode == 3) {
            // Upstream measurement on qubit 2 driving classical feedback before
            // the phase region.
            lines.push_back((rng() & 1U) ? "MX 2" : "M 2");
            ++num_measurements;
            lines.push_back((rng() & 1U) ? "CX rec[-1] 0" : "CZ rec[-1] 0");
        }

        // Stage 2: planted phase or arbitrary-angle rotation block whose
        // reduction depends on the Z1 eigenvalue on surviving trajectories.
        const bool negate_pair = (trial % 5) == 1;
        if (phase_mode == 0) {
            lines.push_back("T 0");
            lines.push_back("CX 1 0");
            lines.push_back(negate_pair ? "T 0" : "T_DAG 0");
            lines.push_back("CX 1 0");
            if ((trial & 1) == 0) {
                lines.push_back("T 0");
                lines.push_back("CX 1 0");
                lines.push_back(negate_pair ? "T 0" : "T_DAG 0");
                lines.push_back("CX 1 0");
            }
        } else if (phase_mode == 1) {
            lines.push_back("T 0");
            lines.push_back("CX 1 0");
            lines.push_back(negate_pair ? "T 0" : "T_DAG 0");
            lines.push_back("CX 1 0");
            lines.push_back("TPP Z0*Z2");
            lines.push_back(negate_pair ? "TPP Z0*Z1*Z2" : "TPP_DAG Z0*Z1*Z2");
        } else {
            lines.push_back("R_Z(0.375) 0");
            lines.push_back("CX 1 0");
            lines.push_back(negate_pair ? "R_Z(0.375) 0" : "R_Z(-0.375) 0");
            lines.push_back("CX 1 0");
            lines.push_back("R_Z(0.125) 0");
            lines.push_back("CX 1 0");
            lines.push_back(negate_pair ? "R_Z(0.125) 0" : "R_Z(-0.125) 0");
            lines.push_back("CX 1 0");
        }

        // Stage 3: interleaved lookahead window between the phase block and
        // the downstream postselected detector.
        const std::string probe1 = kProbeAxes[rng() % std::size(kProbeAxes)];
        const std::string probe2 = kProbeAxes[rng() % std::size(kProbeAxes)];
        const std::string inter_meas =
            kInterveningMeasurements[trial % std::size(kInterveningMeasurements)];

        if (window_mode == 0) {
            // phase -> intervening measurement -> probe -> postselected M 1
            lines.push_back(inter_meas);
            ++num_measurements;
            lines.push_back("EXP_VAL " + probe1);
        } else if (window_mode == 1) {
            // phase -> probe -> intervening measurement -> probe -> postselected M 1
            lines.push_back("EXP_VAL " + probe1);
            lines.push_back(inter_meas);
            ++num_measurements;
            lines.push_back("EXP_VAL " + probe2);
        } else if (window_mode == 2) {
            // phase -> intervening measurement -> feedback -> probe -> postselected M 1
            lines.push_back((rng() & 1U) ? "M 2" : "MX 0");
            ++num_measurements;
            lines.push_back((rng() & 1U) ? "CZ rec[-1] 0" : "CX rec[-1] 2");
            lines.push_back("EXP_VAL " + probe1);
        } else if (window_mode == 3) {
            // phase -> Pauli noise -> probe -> postselected M 1
            lines.push_back((rng() & 1U) ? "Z_ERROR(0.15) 1" : "X_ERROR(0.15) 1");
            lines.push_back("EXP_VAL " + probe1);
        } else if (window_mode == 4) {
            // phase -> commuting measurement -> readout noise -> probe -> postselected M 1
            lines.push_back("M 2");
            ++num_measurements;
            lines.push_back("READOUT_NOISE(0.15, 0.05) rec[-1]");
            lines.push_back("EXP_VAL " + probe1);
        } else {
            // phase -> probe -> mid-circuit reset on spectator -> postselected M 1
            lines.push_back("EXP_VAL " + probe1);
            lines.push_back("R 2");
            lines.push_back("H 2");
        }

        // Stage 4: downstream postselected measurement and detector.
        const bool noisy_postselected_readout = (trial % 9) == 8;
        if (noisy_postselected_readout) {
            lines.push_back("M(0.15) 1");
        } else {
            lines.push_back("M 1");
        }
        const int downstream_check_meas = static_cast<int>(num_measurements++);
        if (upstream_mode == 2 && upstream_check_meas >= 0) {
            const int offset = downstream_check_meas - upstream_check_meas + 1;
            lines.push_back("DETECTOR rec[-1] rec[-" + std::to_string(offset) + "]");
        } else {
            lines.push_back("DETECTOR rec[-1]");
        }
        ++num_detectors;

        // Stage 5: terminal measurements, trailing probe, and observable.
        lines.push_back("MX 0");
        ++num_measurements;
        lines.push_back("M 2");
        ++num_measurements;
        if ((rng() & 1U) != 0) {
            lines.push_back("EXP_VAL " + std::string(kProbeAxes[rng() % std::size(kProbeAxes)]));
        }
        lines.push_back("OBSERVABLE_INCLUDE(0) rec[-2]");

        const std::string source = join_lines(lines);
        const Circuit circuit = clifft::parse(source);

        HirModule clean_unoptimized = clifft::trace(circuit);
        RemoveNoisePass().run(clean_unoptimized);
        const ExactPlanDistribution noiseless_dist =
            compute_exact_hir_distribution(clean_unoptimized);
        auto assert_deterministic_reference = [&](const HirModule& module) {
            const ReferenceSyndrome ref = compute_reference_syndrome(module);
            bool any_checked = false;
            for (uint32_t d = 0; d < module.num_detectors; ++d) {
                uint64_t seen_mask = 0;
                for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                    if (entry.probability > 1e-12) {
                        seen_mask |= 1ULL << ((entry.outcome.detectors >> d) & 1ULL);
                    }
                }
                if (seen_mask == 1ULL || seen_mask == 2ULL) {
                    const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                    REQUIRE(ref.detectors.at(d) == expected_bit);
                    any_checked = true;
                }
            }
            for (uint32_t o = 0; o < module.num_observables; ++o) {
                uint64_t seen_mask = 0;
                for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                    if (entry.probability > 1e-12) {
                        seen_mask |= 1ULL << ((entry.outcome.observables >> o) & 1ULL);
                    }
                }
                if (seen_mask == 1ULL || seen_mask == 2ULL) {
                    const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                    REQUIRE(ref.observables.at(o) == expected_bit);
                    any_checked = true;
                }
            }
            REQUIRE(any_checked);
        };

        for (bool normalize : {false, true}) {
            CAPTURE(trial, upstream_mode, phase_mode, window_mode, flip_check_qubit, normalize,
                    source);

            HirModule original = clifft::trace(circuit);
            const ReferenceSyndrome ref_before = compute_reference_syndrome(original);
            const std::vector<uint8_t> postselection_mask(num_detectors, 1);
            const std::vector<uint8_t> expected_detectors =
                normalize ? ref_before.detectors : std::vector<uint8_t>{};
            const std::vector<uint8_t> expected_observables =
                normalize ? ref_before.observables : std::vector<uint8_t>{};

            original.postselection.mask = postselection_mask;
            original.postselection.expected_detectors = expected_detectors;
            original.postselection.expected_observables = expected_observables;

            const clifft::sampling::SamplingPlanOptions options{
                .postselection_mask = postselection_mask,
                .expected_detectors = expected_detectors,
                .expected_observables = expected_observables,
            };

            const ExactPlanDistribution dense_dist =
                compute_dense_circuit_distribution(circuit, options);
            REQUIRE(dense_dist.discarded_probability < 1.0 - 1e-6);
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(original, options));

            // Check witness conditions against un-postselected optimization.
            HirModule unpostselected = clifft::trace(circuit);
            PhasePolynomialPass unpost_phase;
            unpost_phase.run(unpostselected);

            HirModule post_phase_hir = original;
            PhasePolynomialPass post_phase;
            post_phase.run(post_phase_hir);
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(post_phase_hir, options));
            assert_deterministic_reference(post_phase_hir);

            if (post_phase.output_t_count() < unpost_phase.output_t_count()) {
                ++witness_postselected_phase_reduced;
            }

            if (window_mode <= 2 && post_phase.output_t_count() == post_phase.input_t_count() &&
                post_phase.input_t_count() > 0) {
                // Verify whether removing EXP_VAL ops would allow the postselected
                // phase reduction to fire, witnessing that the probe/measurement
                // lookahead guard actively blocked an unsafe rewrite.
                HirModule without_probes = original;
                std::erase_if(without_probes.ops, [](const HeisenbergOp& op) {
                    return op.op_type() == OpType::EXP_VAL;
                });
                without_probes.num_exp_vals = 0;
                without_probes.source_map.clear();
                PhasePolynomialPass probe_free_phase;
                probe_free_phase.run(without_probes);
                if (probe_free_phase.output_t_count() < post_phase.output_t_count()) {
                    ++witness_intervening_meas_probe_blocked;
                }
            }

            HirModule unpost_rot_hir = clifft::trace(circuit);
            RotationSimplificationPass unpost_rot;
            unpost_rot.run(unpost_rot_hir);

            HirModule post_rot_hir = original;
            RotationSimplificationPass post_rot;
            post_rot.run(post_rot_hir);
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(post_rot_hir, options));
            assert_deterministic_reference(post_rot_hir);

            if (post_rot.rotations_removed() > unpost_rot.rotations_removed()) {
                ++witness_postselected_rotation_reduced;
            }

            const bool noiseless_rejected =
                !normalize && std::ranges::any_of(ref_before.detectors,
                                                  [](uint8_t parity) { return parity != 0; });
            if (noiseless_rejected &&
                (post_phase.blocks_reduced() > 0 || post_rot.rotations_removed() > 0)) {
                ++witness_noiseless_outside_subspace;
            }

            // Full pipeline + scheduler equivalence and reference preservation.
            HirModule pipeline_hir = original;
            HirPassManager pm = default_hir_pass_manager();
            pm.add_pass(std::make_unique<ActiveWidthSchedulePass>());
            pm.run(pipeline_hir);
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(pipeline_hir, options));
            assert_deterministic_reference(pipeline_hir);
            check_exact_equivalent(original, pipeline_hir, control_rng, 8, options);
        }
    }

    REQUIRE(witness_postselected_phase_reduced >= 6);
    REQUIRE(witness_postselected_rotation_reduced >= 4);
    REQUIRE(witness_intervening_meas_probe_blocked >= 4);
    REQUIRE(witness_noiseless_outside_subspace >= 6);
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
            HirModule original = clifft::trace(circuit);
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

namespace {

void run_full_hir_pipeline(HirModule& hir) {
    HirPassManager pm = default_hir_pass_manager();
    pm.add_pass(std::make_unique<ActiveWidthSchedulePass>());
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.run(hir);
}

// Constructs HIR modules with a commuting Pauli basis (on either dense qubits
// or multi-word qubit positions) plus known |0> input stabilizers Z_q.
// Rotations are emitted in degree-3 TOHPE cubes and stabilizer-aliased
// multiples so PhasePolynomialPass exercises:
// - KnownStabilizers canonicalization and multi-generator reduction
// - Mod-8 coefficient accumulation across c in {0..7}
// - Quadratic monomial remainders (4*a*b -> S(a)S(b)S_dag(a XOR b))
// - Cubic TOHPE synthesis
// - Observer pullbacks (MEASURE, EXP_VAL, NOISE) and stabilizer invalidation.
HirModule generate_phase_poly_hir(Xoshiro256PlusPlus& rng, std::span<const uint32_t> qubit_map,
                                  uint32_t total_qubits) {
    const size_t n_logical = qubit_map.size();
    REQUIRE(n_logical >= 4);

    HirModule hir(total_qubits, 64, 8);
    hir.final_tableau.emplace(total_qubits);

    const uint32_t q0 = qubit_map[0];
    const uint32_t q1 = qubit_map[1];
    const uint32_t q2 = qubit_map[2];
    const bool use_full_16_cube = (rng() & 1U) != 0 && n_logical >= 5;
    // The fourth cube coordinate cannot also supply a fixed Z stabilizer.
    const uint32_t q_stab = qubit_map[use_full_16_cube ? 4 : 3];
    const uint32_t q_stab2 = n_logical > 4 ? qubit_map[4] : q_stab;

    auto fill_x_parity_with_stab = [&](MutablePauliMaskView slot, uint32_t parity_bits,
                                       bool alias_stab1, bool alias_stab2, bool sign) {
        if (parity_bits & 1U) {
            slot.x().bit_set(q0, true);
        }
        if (parity_bits & 2U) {
            slot.x().bit_set(q1, true);
        }
        if (parity_bits & 4U) {
            slot.x().bit_set(q2, true);
        }
        if (alias_stab1) {
            slot.z().bit_set(q_stab, true);
        }
        if (alias_stab2 && q_stab2 != q_stab) {
            slot.z().bit_set(q_stab2, true);
        }
        slot.set_sign(sign);
    };

    if (use_full_16_cube) {
        const uint32_t q3 = qubit_map[3];
        for (uint32_t term = 1; term <= 16; ++term) {
            const uint32_t parity = (term == 16) ? 15U : term;
            const bool alias = (rng() % 3) == 0;
            hir.append_tgate(false, [&](MutablePauliMaskView slot) {
                if (parity & 1U)
                    slot.x().bit_set(q0, true);
                if (parity & 2U)
                    slot.x().bit_set(q1, true);
                if (parity & 4U)
                    slot.x().bit_set(q2, true);
                if (parity & 8U)
                    slot.x().bit_set(q3, true);
                if (alias)
                    slot.z().bit_set(q_stab, true);
            });
        }
    } else {
        for (uint32_t parity = 1; parity <= 7; ++parity) {
            const uint32_t copies = 1 + static_cast<uint32_t>(rng() % 4);
            for (uint32_t c = 0; c < copies; ++c) {
                const bool alias1 = (rng() & 1U) != 0;
                const bool alias2 = (rng() & 1U) != 0;
                const bool dagger = (rng() % 5) == 0;
                const bool sign = (rng() % 4) == 0;
                hir.append_tgate(dagger, [&](MutablePauliMaskView slot) {
                    fill_x_parity_with_stab(slot, parity, alias1, alias2, sign);
                });
            }
        }
    }

    // Interleave a mid-circuit observer (EXP_VAL, MEASURE, or commuting NOISE)
    // followed by trailing T gates that can merge if their stabilizer aliases
    // survive the observer.
    const uint32_t obs_kind = static_cast<uint32_t>(rng() % 4);
    if (obs_kind == 0) {
        const bool commuting_probe = (rng() % 3) != 0;
        const ExpValIdx exp_idx{hir.num_exp_vals++};
        hir.append_exp_val(exp_idx, [&](MutablePauliMaskView slot) {
            if (commuting_probe) {
                slot.x().bit_set(q0, true);
                if ((rng() & 1U) != 0) {
                    slot.x().bit_set(q1, true);
                }
            } else {
                slot.z().bit_set(q0, true);
            }
            slot.set_sign((rng() & 1U) != 0);
        });
    } else if (obs_kind == 1) {
        const bool measure_commuting = (rng() % 3) != 0;
        const MeasRecordIdx meas_idx{hir.num_measurements++};
        hir.append_measure(meas_idx, [&](MutablePauliMaskView slot) {
            if (measure_commuting) {
                slot.x().bit_set(q0, true);
            } else {
                slot.x().bit_set(q_stab, true);
            }
            slot.set_sign((rng() & 1U) != 0);
        });
    } else if (obs_kind == 2) {
        const PauliMaskHandle ch = hir.claim_empty_noise_channel_mask();
        hir.noise_channel_masks.mut_at(ch).x().bit_set(q1, true);
        const NoiseSiteIdx site_idx{static_cast<uint32_t>(hir.noise_sites.size())};
        hir.noise_sites.push_back({0.15, {{ch, 0.15}}});
        hir.append_noise(site_idx);
    } else {
        const ExpValIdx exp_idx{hir.num_exp_vals++};
        hir.append_exp_val(exp_idx, [&](MutablePauliMaskView slot) {
            slot.x().bit_set(q1, true);
            slot.set_sign((rng() & 1U) != 0);
        });
        const MeasRecordIdx meas_idx{hir.num_measurements++};
        hir.append_measure(meas_idx,
                           [&](MutablePauliMaskView slot) { slot.x().bit_set(q2, true); });
    }

    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t parity = 1 + static_cast<uint32_t>(rng() % 7);
        const bool alias1 = (rng() & 1U) != 0;
        const bool dagger = (rng() & 1U) != 0;
        hir.append_tgate(dagger, [&](MutablePauliMaskView slot) {
            fill_x_parity_with_stab(slot, parity, alias1, false, false);
        });
    }

    // Terminal probe and measurements sensitive to X-basis phase angles and
    // any absorbed degree-1 / degree-2 Clifford remainder.
    const ExpValIdx term_exp{hir.num_exp_vals++};
    hir.append_exp_val(term_exp, [&](MutablePauliMaskView slot) {
        slot.x().bit_set(q0, true);
        slot.z().bit_set(q0, true);
        slot.z().bit_set(q1, true);
    });

    const uint32_t m0 = hir.num_measurements++;
    hir.append_measure(MeasRecordIdx{m0},
                       [&](MutablePauliMaskView slot) { slot.z().bit_set(q0, true); });

    const uint32_t m1 = hir.num_measurements++;
    hir.append_measure(MeasRecordIdx{m1},
                       [&](MutablePauliMaskView slot) { slot.z().bit_set(q1, true); });

    const uint32_t m2 = hir.num_measurements++;
    hir.append_measure(MeasRecordIdx{m2},
                       [&](MutablePauliMaskView slot) { slot.z().bit_set(q2, true); });

    hir.detector_targets.push_back({m0, m1});
    hir.ops.push_back(HeisenbergOp::make_detector(DetectorIdx{hir.num_detectors++}));

    hir.observable_targets.push_back({m0, m1, m2});
    hir.ops.push_back(HeisenbergOp::make_observable(ObservableIdx{0}, 0));
    hir.num_observables = 1;

    return hir;
}

// Builds HIR modules where continuous rotations R_PAULI(alpha) share axes up
// to known |0> stabilizers (or stabilizers surviving across region boundaries)
// and are separated by commuting rotations or anticommuting barriers.
// Absorbing an exposed Clifford can let collection cross a barrier.
HirModule generate_rotation_simp_hir(Xoshiro256PlusPlus& rng, std::span<const uint32_t> qubit_map,
                                     uint32_t total_qubits) {
    REQUIRE(qubit_map.size() >= 4);

    HirModule hir(total_qubits, 48, 4);
    hir.final_tableau.emplace(total_qubits);

    const uint32_t q0 = qubit_map[0];
    const uint32_t q1 = qubit_map[1];
    const uint32_t q_stab1 = qubit_map[2];
    const uint32_t q_stab2 = qubit_map[3];

    static constexpr double kBaseAngles[] = {0.125, 0.25, 0.3, 0.375, 0.5, 0.625, 0.7};
    for (int region = 0; region < 2; ++region) {
        const double a1 = kBaseAngles[rng() % std::size(kBaseAngles)];
        static constexpr double kTargetSums[] = {0.0, 0.25, 0.5, 1.0, 1.5, 0.6};
        const double target_sum = kTargetSums[rng() % std::size(kTargetSums)];
        const double a2 = target_sum - a1;

        const bool alias_first = (rng() & 1U) != 0;
        hir.append_phase_rotation(a1, [&](MutablePauliMaskView slot) {
            slot.x().bit_set(q0, true);
            if (alias_first) {
                slot.z().bit_set(q_stab1, true);
            }
        });

        const bool intervening_commutes = (rng() % 3) != 0;
        const double a_mid = kBaseAngles[rng() % std::size(kBaseAngles)];
        hir.append_phase_rotation(a_mid, [&](MutablePauliMaskView slot) {
            slot.x().bit_set(q1, true);
            if (!intervening_commutes) {
                slot.z().bit_set(q0, true);
            }
        });

        const bool alias_second_1 = !alias_first;
        const bool alias_second_2 = (rng() & 1U) != 0;
        const bool flip_sign = (rng() & 1U) != 0;
        hir.append_phase_rotation(flip_sign ? -a2 : a2, [&](MutablePauliMaskView slot) {
            slot.x().bit_set(q0, true);
            if (alias_second_1) {
                slot.z().bit_set(q_stab1, true);
            }
            if (alias_second_2) {
                slot.z().bit_set(q_stab2, true);
            }
            slot.set_sign(flip_sign);
        });

        if (region == 0) {
            if ((rng() & 1U) != 0) {
                const ExpValIdx exp_idx{hir.num_exp_vals++};
                hir.append_exp_val(exp_idx, [&](MutablePauliMaskView slot) {
                    slot.z().bit_set(q0, true);
                    slot.x().bit_set(q1, true);
                });
            } else {
                const MeasRecordIdx meas_idx{hir.num_measurements++};
                hir.append_measure(meas_idx,
                                   [&](MutablePauliMaskView slot) { slot.z().bit_set(q0, true); });
            }
        }
    }

    const ExpValIdx exp_idx{hir.num_exp_vals++};
    hir.append_exp_val(exp_idx, [&](MutablePauliMaskView slot) {
        slot.z().bit_set(q0, true);
        slot.z().bit_set(q1, true);
    });

    const uint32_t m0 = hir.num_measurements++;
    hir.append_measure(MeasRecordIdx{m0},
                       [&](MutablePauliMaskView slot) { slot.z().bit_set(q0, true); });

    const uint32_t m1 = hir.num_measurements++;
    hir.append_measure(MeasRecordIdx{m1},
                       [&](MutablePauliMaskView slot) { slot.z().bit_set(q1, true); });

    hir.detector_targets.push_back({m0, m1});
    hir.ops.push_back(HeisenbergOp::make_detector(DetectorIdx{hir.num_detectors++}));

    hir.observable_targets.push_back({m0, m1});
    hir.ops.push_back(HeisenbergOp::make_observable(ObservableIdx{0}, 0));
    hir.num_observables = 1;

    return hir;
}

}  // namespace

TEST_CASE(
    "Structured phase polynomial HIR generator preserves exact distributions across word "
    "boundaries",
    "[pass_equivalence][optimizer]") {
    Xoshiro256PlusPlus rng(0x51A5E01);
    Xoshiro256PlusPlus control_rng(0x51A5E02);

    size_t applied_count = 0;
    size_t blocks_reduced_total = 0;
    size_t pullbacks_total = 0;
    size_t t_reduced_cases = 0;

    // Exercise dense single-word qubits and sparse multi-word layouts crossing
    // the 64-bit and 128-bit boundaries through the calibrated exact oracle.
    const std::vector<std::pair<uint32_t, std::vector<uint32_t>>> layouts = {
        {5, {0, 1, 2, 3, 4}},
        {66, {0, 1, 63, 64, 65}},
        {130, {0, 63, 64, 128, 129}},
    };

    for (int trial = 0; trial < 36; ++trial) {
        const auto& [total_qubits, qubit_map] =
            layouts[static_cast<size_t>(trial) % layouts.size()];
        const HirModule original = generate_phase_poly_hir(rng, qubit_map, total_qubits);
        CAPTURE(trial, total_qubits);

        HirModule phase_opt = original;
        PhasePolynomialPass pass;
        pass.run(phase_opt);

        if (pass.applied()) {
            ++applied_count;
            blocks_reduced_total += pass.blocks_reduced();
            pullbacks_total += pass.pauli_pullbacks();
            if (pass.output_t_count() < pass.input_t_count()) {
                ++t_reduced_cases;
            }
        }

        check_exact_equivalent(original, phase_opt, control_rng);

        // Repeated optimization must preserve behavior even if more rewrites become possible.
        pass.run(phase_opt);
        check_exact_equivalent(original, phase_opt, control_rng);

        HirModule full_opt = original;
        run_full_hir_pipeline(full_opt);
        check_exact_equivalent(original, full_opt, control_rng);
    }

    REQUIRE(applied_count >= 24);
    REQUIRE(blocks_reduced_total >= 24);
    REQUIRE(pullbacks_total >= 8);
    REQUIRE(t_reduced_cases >= 20);
}

TEST_CASE(
    "Structured rotation simplification HIR generator preserves exact distributions across word "
    "boundaries",
    "[pass_equivalence][optimizer]") {
    Xoshiro256PlusPlus rng(0x707A7101);
    Xoshiro256PlusPlus control_rng(0x707A7102);

    size_t applied_count = 0;
    size_t regions_reduced_total = 0;
    size_t rotations_removed_total = 0;

    const std::vector<std::pair<uint32_t, std::vector<uint32_t>>> layouts = {
        {4, {0, 1, 2, 3}},
        {66, {0, 63, 64, 65}},
        {130, {0, 63, 64, 129}},
    };

    for (int trial = 0; trial < 36; ++trial) {
        const auto& [total_qubits, qubit_map] =
            layouts[static_cast<size_t>(trial) % layouts.size()];
        const HirModule original = generate_rotation_simp_hir(rng, qubit_map, total_qubits);
        CAPTURE(trial, total_qubits);

        HirModule rot_opt = original;
        RotationSimplificationPass pass;
        pass.run(rot_opt);

        if (pass.applied()) {
            ++applied_count;
            regions_reduced_total += pass.regions_reduced();
            rotations_removed_total += pass.rotations_removed();
        }

        check_exact_equivalent(original, rot_opt, control_rng);

        pass.run(rot_opt);
        check_exact_equivalent(original, rot_opt, control_rng);

        HirModule full_opt = original;
        run_full_hir_pipeline(full_opt);
        check_exact_equivalent(original, full_opt, control_rng);
    }

    REQUIRE(applied_count >= 20);
    REQUIRE(regions_reduced_total >= 20);
    REQUIRE(rotations_removed_total >= 24);
}

TEST_CASE("Varied pass sequences preserve exact distributions across hazards",
          "[pass_equivalence][optimizer]") {
    Xoshiro256PlusPlus rng(0x9E730001);
    Xoshiro256PlusPlus control_rng(0x9E730002);

    const PlantedScaffoldKind scaffolds[] = {
        PlantedScaffoldKind::PhaseSandwich,
        PlantedScaffoldKind::StabilizerReducedPhase,
        PlantedScaffoldKind::FeedbackAndReset,
        PlantedScaffoldKind::MultiRoundSchedule,
    };
    const HazardConfig hazard_configs[] = {
        {},
        {.pauli_noise = true, .readout_noise = true, .exp_val = true},
        {.correlated_noise = true, .resets = true, .postselect_check_detectors = true},
        {.pauli_noise = true,
         .readout_noise = true,
         .exp_val = true,
         .postselect_check_detectors = true},
    };

    std::vector<const PassInfo*> non_destructive_passes;
    for (const PassInfo& info : kRegisteredPasses) {
        const bool is_destructive = std::find(std::begin(kDestructiveRegisteredPasses),
                                              std::end(kDestructiveRegisteredPasses),
                                              info.name) != std::end(kDestructiveRegisteredPasses);
        if (!is_destructive) {
            non_destructive_passes.push_back(&info);
        }
    }

    size_t postselection_specializations = 0;

    for (int trial = 0; trial < 32; ++trial) {
        const PlantedScaffoldKind scaffold =
            scaffolds[static_cast<size_t>(trial) % std::size(scaffolds)];
        const HazardConfig& hazards =
            hazard_configs[(static_cast<size_t>(trial) / std::size(scaffolds)) %
                           std::size(hazard_configs)];
        const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
        const std::string source = join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
        CAPTURE(trial, static_cast<int>(scaffold), source);

        const Circuit circuit = clifft::parse(source);
        HirModule clean_unoptimized = clifft::trace(circuit);
        RemoveNoisePass().run(clean_unoptimized);
        const ExactPlanDistribution noiseless_dist =
            compute_exact_hir_distribution(clean_unoptimized);
        auto assert_deterministic_reference = [&](const HirModule& module) {
            const ReferenceSyndrome ref = compute_reference_syndrome(module);
            for (uint32_t d = 0; d < module.num_detectors; ++d) {
                uint64_t seen_mask = 0;
                for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                    if (entry.probability > 1e-12) {
                        seen_mask |= 1ULL << ((entry.outcome.detectors >> d) & 1ULL);
                    }
                }
                if (seen_mask == 1ULL || seen_mask == 2ULL) {
                    const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                    REQUIRE(ref.detectors.at(d) == expected_bit);
                }
            }
            for (uint32_t o = 0; o < module.num_observables; ++o) {
                uint64_t seen_mask = 0;
                for (const ExactOutcomeEntry& entry : noiseless_dist.kept_outcomes) {
                    if (entry.probability > 1e-12) {
                        seen_mask |= 1ULL << ((entry.outcome.observables >> o) & 1ULL);
                    }
                }
                if (seen_mask == 1ULL || seen_mask == 2ULL) {
                    const uint8_t expected_bit = (seen_mask == 2ULL) ? 1 : 0;
                    REQUIRE(ref.observables.at(o) == expected_bit);
                }
            }
        };

        HirModule original = clifft::trace(circuit);
        const OwnedPlanOptions owned =
            build_plan_options(original, planted, hazards.postselect_check_detectors);
        const auto options = owned.as_span_options();
        const ExactPlanDistribution dense_dist =
            compute_dense_circuit_distribution(circuit, options);

        HirModule candidate = original;
        const size_t pipeline_len = 3 + (rng() % 5);
        for (size_t step = 0; step < pipeline_len; ++step) {
            const size_t choice = rng() % (non_destructive_passes.size() + 1);
            const std::string_view pass_name = choice < non_destructive_passes.size()
                                                   ? non_destructive_passes[choice]->name
                                                   : "ActiveWidthSchedulePass";
            CAPTURE(step, pass_name);
            if (choice < non_destructive_passes.size()) {
                non_destructive_passes[choice]->make()->run(candidate);
            } else {
                ActiveWidthScheduleOptions sched_opts;
                sched_opts.beam_width = 1 + static_cast<uint32_t>(rng() % 4);
                sched_opts.noise_transparent = (rng() & 1U) != 0;
                sched_opts.sink_neutral_rotations = (rng() & 1U) != 0;
                sched_opts.search_budget =
                    (rng() & 1U) != 0 ? std::optional<double>(0.0) : std::optional<double>(8.0);
                ActiveWidthSchedulePass sched(sched_opts);
                sched.run(candidate);
                if (sched.applied()) {
                    REQUIRE(sched.result_peak() <= sched.incumbent_peak());
                }
            }
            require_distributions_equal(dense_dist,
                                        compute_exact_hir_distribution(candidate, options));
            assert_deterministic_reference(candidate);
        }

        if (candidate.postselection.pre_postselection_hir != nullptr) {
            ++postselection_specializations;
        }
        check_exact_equivalent(original, candidate, control_rng, 16, options);
    }

    REQUIRE(postselection_specializations >= 4);
}

TEST_CASE("Multiword postselected circuits preserve survivor distributions across word boundaries",
          "[pass_equivalence][optimizer]") {
    const uint32_t layouts[][3] = {
        {63, 64, 65},
        {63, 127, 128},
    };
    const uint8_t postselect_mask[] = {1};
    const uint8_t expected_dets[] = {0};
    const clifft::sampling::SamplingPlanOptions options{
        .postselection_mask = postselect_mask,
        .expected_detectors = expected_dets,
    };

    auto make_forward_recovery = [](uint32_t q0, uint32_t q1, uint32_t q2) {
        const std::string s0 = std::to_string(q0);
        const std::string s1 = std::to_string(q1);
        const std::string s2 = std::to_string(q2);
        return "H " + s0 + "\nCX " + s0 + " " + s1 + "\nDEPOLARIZE1(0.1) " + s0 + " " + s1 +
               "\nCX " + s0 + " " + s2 + " " + s1 + " " + s2 + "\nM " + s2 +
               "\nDETECTOR rec[-1]\nT " + s0 + "\nT_DAG " + s1 + "\nCX " + s0 + " " + s1 + "\nMX " +
               s0 + "\nM " + s1 + "\nOBSERVABLE_INCLUDE(0) rec[-2]\n";
    };

    auto make_lookahead = [](uint32_t q0, uint32_t q1, uint32_t q2) {
        const std::string s0 = std::to_string(q0);
        const std::string s1 = std::to_string(q1);
        const std::string s2 = std::to_string(q2);
        return "H " + s0 + "\nX_ERROR(0.2) " + s1 + "\nZ_ERROR(0.1) " + s0 + "\nTPP Z" + s0 + "*Z" +
               s2 + "\nTPP_DAG Z" + s0 + "*Z" + s1 + "*Z" + s2 + "\nEXP_VAL Z" + s0 + "\nM " + s1 +
               "\nDETECTOR rec[-1]\nMX " + s0 + "\nM " + s2 + "\nOBSERVABLE_INCLUDE(0) rec[-2]\n";
    };

    auto make_blocking_probe = [](uint32_t q0, uint32_t q1, uint32_t q2) {
        const std::string s0 = std::to_string(q0);
        const std::string s1 = std::to_string(q1);
        const std::string s2 = std::to_string(q2);
        return "H " + s0 + " " + s1 + "\nZ_ERROR(0.1) " + s1 + "\nT " + s0 + "\nCX " + s1 + " " +
               s0 + "\nT_DAG " + s0 + "\nCX " + s1 + " " + s0 + "\nMX " + s0 + "\nEXP_VAL Z" + s1 +
               "\nM " + s1 + "\nM " + s2 + "\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n";
    };

    const ExactPlanDistribution dense_forward =
        compute_dense_circuit_distribution(clifft::parse(make_forward_recovery(0, 1, 2)), options);
    const ExactPlanDistribution dense_lookahead =
        compute_dense_circuit_distribution(clifft::parse(make_lookahead(0, 1, 2)), options);
    const ExactPlanDistribution dense_blocked =
        compute_dense_circuit_distribution(clifft::parse(make_blocking_probe(0, 1, 2)), options);

    for (const auto& layout : layouts) {
        const uint32_t q0 = layout[0];
        const uint32_t q1 = layout[1];
        const uint32_t q2 = layout[2];
        CAPTURE(q0, q1, q2);

        for (bool is_lookahead : {false, true}) {
            const std::string source =
                is_lookahead ? make_lookahead(q0, q1, q2) : make_forward_recovery(q0, q1, q2);
            const ExactPlanDistribution& expected_dist =
                is_lookahead ? dense_lookahead : dense_forward;

            HirModule unpost = clifft::trace(clifft::parse(source));
            PhasePolynomialPass unpost_pass;
            unpost_pass.run(unpost);
            REQUIRE(unpost_pass.output_t_count() > 0);

            HirModule post = clifft::trace(clifft::parse(source));
            post.postselection.mask = {1};
            post.postselection.expected_detectors = {0};
            require_distributions_equal(expected_dist,
                                        compute_exact_hir_distribution(post, options));

            HirModule phase_opt = post;
            PhasePolynomialPass post_pass;
            post_pass.run(phase_opt);
            REQUIRE(post_pass.applied());
            REQUIRE(post_pass.output_t_count() == 0);
            require_distributions_equal(expected_dist,
                                        compute_exact_hir_distribution(phase_opt, options));

            HirModule full_opt = post;
            run_full_hir_pipeline(full_opt);
            require_distributions_equal(expected_dist,
                                        compute_exact_hir_distribution(full_opt, options));
        }

        // Intervening measurement + probe must block lookahead across multiword masks.
        const std::string blocked_source = make_blocking_probe(q0, q1, q2);
        HirModule blocked_hir = clifft::trace(clifft::parse(blocked_source));
        blocked_hir.postselection.mask = {1};
        blocked_hir.postselection.expected_detectors = {0};
        require_distributions_equal(dense_blocked,
                                    compute_exact_hir_distribution(blocked_hir, options));

        HirModule blocked_phase = blocked_hir;
        PhasePolynomialPass blocked_pass;
        blocked_pass.run(blocked_phase);
        REQUIRE(!blocked_pass.applied());
        REQUIRE(blocked_pass.output_t_count() == 2);
        require_distributions_equal(dense_blocked,
                                    compute_exact_hir_distribution(blocked_phase, options));

        HirModule without_probe = blocked_hir;
        std::erase_if(without_probe.ops,
                      [](const HeisenbergOp& op) { return op.op_type() == OpType::EXP_VAL; });
        without_probe.num_exp_vals = 0;
        without_probe.source_map.clear();
        PhasePolynomialPass probe_free_pass;
        probe_free_pass.run(without_probe);
        REQUIRE(probe_free_pass.applied());
        REQUIRE(probe_free_pass.output_t_count() == 0);

        HirModule blocked_full = blocked_hir;
        run_full_hir_pipeline(blocked_full);
        require_distributions_equal(dense_blocked,
                                    compute_exact_hir_distribution(blocked_full, options));
    }
}
