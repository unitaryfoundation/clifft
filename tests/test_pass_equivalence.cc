#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
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

// =============================================================================
// Planted-structure and boundary-hazard exact equivalence for individual passes
// =============================================================================

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

    size_t total_fusions = 0;
    size_t total_cancellations = 0;

    for (PlantedScaffoldKind scaffold : scaffolds) {
        for (const HazardConfig& hazards : hazard_configs) {
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), trial, source);

                const HirModule original = parse_and_trace(source);
                HirModule optimized = original;
                PeepholeFusionPass pass;
                pass.run(optimized);

                total_fusions += pass.fusions();
                total_cancellations += pass.cancellations();

                const OwnedPlanOptions owned = build_plan_options(original, planted, false);
                const auto options = owned.as_span_options();
                check_exact_equivalent(original, optimized, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, optimized, options);
            }
        }
    }

    REQUIRE(total_fusions + total_cancellations >= 30);
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

    size_t reduced_blocks = 0;
    size_t t_saved = 0;

    for (PlantedScaffoldKind scaffold : scaffolds) {
        for (const HazardConfig& hazards : hazard_configs) {
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), trial, source);

                const HirModule original = parse_and_trace(source);
                HirModule optimized = original;
                PhasePolynomialPass pass;
                pass.run(optimized);

                reduced_blocks += pass.blocks_reduced();
                if (pass.input_t_count() > pass.output_t_count()) {
                    t_saved += pass.input_t_count() - pass.output_t_count();
                }

                const OwnedPlanOptions owned = build_plan_options(original, planted, false);
                const auto options = owned.as_span_options();
                check_exact_equivalent(original, optimized, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, optimized, options);
            }
        }
    }

    REQUIRE(reduced_blocks >= 15);
    REQUIRE(t_saved >= 20);
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

    size_t reduced_regions = 0;
    size_t rotations_removed = 0;

    for (PlantedScaffoldKind scaffold : scaffolds) {
        for (const HazardConfig& hazards : hazard_configs) {
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), trial, source);

                const HirModule original = parse_and_trace(source);
                HirModule optimized = original;
                RotationSimplificationPass pass;
                pass.run(optimized);

                reduced_regions += pass.regions_reduced();
                rotations_removed += pass.rotations_removed();

                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                check_exact_equivalent(original, optimized, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, optimized, options);
            }
        }
    }

    REQUIRE(reduced_regions >= 15);
    REQUIRE(rotations_removed >= 25);
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
        for (const HazardConfig& hazards : hazard_configs) {
            for (int trial = 0; trial < 4; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), trial, source);

                const HirModule original = parse_and_trace(source);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                const uint32_t unscheduled_peak =
                    clifft::sampling::plan_sampling(original, options).peak_active_width;

                HirModule squeezed = original;
                StatevectorSqueezePass squeeze;
                squeeze.run(squeezed);
                check_exact_equivalent(original, squeezed, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, squeezed, options);

                HirModule scheduled = squeezed;
                ActiveWidthSchedulePass scheduler;
                scheduler.run(scheduled);
                const uint32_t scheduled_peak =
                    clifft::sampling::plan_sampling(scheduled, options).peak_active_width;

                schedule_applied += scheduler.applied() ? 1 : 0;
                noise_crossed += crossed_noise(scheduled) ? 1 : 0;
                width_reductions += (scheduled_peak < unscheduled_peak) ? 1 : 0;

                check_exact_equivalent(original, scheduled, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, scheduled, options);
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
        for (const HazardConfig& hazards : hazard_configs) {
            for (int trial = 0; trial < 3; ++trial) {
                const PlantedCircuit planted = generate_planted_scaffold(scaffold, rng, 4);
                const std::string source =
                    join_lines(inject_boundary_hazards(planted, hazards, rng, 4));
                CAPTURE(static_cast<int>(scaffold), trial, source);

                const HirModule original = parse_and_trace(source);
                const OwnedPlanOptions owned =
                    build_plan_options(original, planted, hazards.postselect_check_detectors);
                const auto options = owned.as_span_options();
                const uint32_t base_peak =
                    clifft::sampling::plan_sampling(original, options).peak_active_width;

                HirModule optimized = original;
                HirPassManager pm = default_hir_pass_manager();
                pm.add_pass(std::make_unique<ActiveWidthSchedulePass>());
                pm.run(optimized);

                const uint32_t opt_peak =
                    clifft::sampling::plan_sampling(optimized, options).peak_active_width;
                if (opt_peak < base_peak) {
                    ++width_improvements;
                }

                check_exact_equivalent(original, optimized, control_rng, 16, options);
                check_unconditional_exact_equivalent(original, optimized, options);
            }
        }
    }

    REQUIRE(width_improvements >= 20);
}
