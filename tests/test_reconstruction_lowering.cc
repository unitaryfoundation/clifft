#include "clifft/circuit/parser.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/reconstruction_lowering.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/sampler.h"

#include "reconstruction_test_helpers.h"

#include <catch2/catch_test_macros.hpp>

namespace clifft::test {

TEST_CASE("Reconstructed HIR preserves complete noisy output laws", "[reconstruction]") {
    for (const auto& text : reconstruction_law_cases()) {
        INFO(text);
        const auto circuit = parse(text);
        const auto expected = compute_dense_circuit_distribution(circuit);
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(circuit, mode);
            REQUIRE(r.reconstruction.has_value());
            auto lowered = lower_global_reconstruction(*r.reconstruction);
            INFO(lowered.fallback_reason);
            REQUIRE(lowered.hir.has_value());
            // Check both the bridge and the passes used by Program selection.
            for (bool optimize : {false, true}) {
                if (optimize)
                    default_hir_pass_manager().run(*lowered.hir);
                const auto actual =
                    compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
                require_same_law(output_law(actual, r.reconstruction->outputs),
                                 output_law(expected, r.reconstruction->outputs));
            }
        }
    }
}

TEST_CASE("Reconstructed output snapshots do not require the source circuit", "[reconstruction]") {
    for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
        const auto r =
            reconstruct_global_sampling(parse("M 0\nDETECTOR rec[-1]\nREADOUT_NOISE(0.25) "
                                              "rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-1]\n"),
                                        mode);
        REQUIRE(r.reconstruction.has_value());
        const auto& layout = r.reconstruction->outputs;
        REQUIRE(layout.num_records == (mode == ReconstructionOutputs::Records ? 1 : 0));
        REQUIRE(layout.num_detectors == 1);
        REQUIRE(layout.num_observables == 3);
        const auto lowered = lower_global_reconstruction(*r.reconstruction);
        REQUIRE(lowered.hir.has_value());
        const auto actual = compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
        // The detector saw zero before readout noise. The final record and third
        // observable see the same flip, while the undeclared observables stay zero.
        const uint64_t flipped = mode == ReconstructionOutputs::Records ? 17 : 8;
        require_same_law(output_law(actual, layout), {{0, 0.75}, {flipped, 0.25}});
    }
}

TEST_CASE("Global reconstruction lowering preserves reference and rejection mass",
          "[reconstruction]") {
    const auto source = parse(
        "H 0\nT 0\nMX 0\nX_ERROR(0.2) 1\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] "
        "rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-2]\n");
    const auto expected_detectors = std::vector<uint8_t>{1, 0};
    const auto expected_observables = std::vector<uint8_t>{1, 1};
    for (const auto& mask :
         {std::vector<uint8_t>{0, 0}, std::vector<uint8_t>{1, 0}, std::vector<uint8_t>{1, 1}}) {
        const auto expected = compute_dense_circuit_distribution(
            source, {mask, expected_detectors, expected_observables});
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(source, mode);
            REQUIRE(r.reconstruction.has_value());
            PostselectionMetadata meta;
            meta.mask = mask;
            meta.expected_detectors = expected_detectors;
            meta.expected_observables = expected_observables;
            const auto lowered = lower_global_reconstruction(*r.reconstruction, meta);
            REQUIRE(lowered.hir.has_value());
            const auto actual =
                compute_exact_plan_distribution(sampling::plan_sampling(*lowered.hir));
            REQUIRE_THAT(actual.discarded_probability,
                         Catch::Matchers::WithinAbs(expected.discarded_probability, 2e-12));
            std::map<uint64_t, double> want, got;
            auto key = [&](const ExactShotOutcome& o) {
                return (mode == ReconstructionOutputs::Records ? o.visible_records : 0) |
                       (o.detectors << 2) | (o.observables << 4);
            };
            for (const auto& e : expected.kept_outcomes)
                want[key(e.outcome)] += e.probability;
            for (const auto& e : actual.kept_outcomes)
                got[key(e.outcome)] += e.probability;
            for (auto [k, p] : want) {
                REQUIRE_THAT(got[k], Catch::Matchers::WithinAbs(p, 2e-12));
                got.erase(k);
            }
            for (auto [k, p] : got)
                REQUIRE_THAT(p, Catch::Matchers::WithinAbs(0, 2e-12));
        }
    }
}

TEST_CASE("Reconstruction lowering reports bounded preparation failures", "[reconstruction]") {
    const auto source = parse("H 0\nT 0\nZ_ERROR(0.2) 0\nZ_ERROR(0.3) 0\nMX 0\nDETECTOR rec[-1]\n");
    ReconstructionLimits limits;
    const auto r = reconstruct_global_sampling(source, ReconstructionOutputs::Records);
    REQUIRE(r.reconstruction.has_value());
    for (uint32_t limit = 0; limit < 3; ++limit) {
        limits = {};
        if (limit == 0)
            limits.max_lowering_bytes = 0;
        if (limit == 1)
            limits.max_channel_products = 0;
        if (limit == 2)
            limits.max_channel_support = 0;
        const auto lowered = lower_global_reconstruction(*r.reconstruction, {}, limits);
        REQUIRE_FALSE(lowered.hir);
        REQUIRE_FALSE(lowered.fallback_reason.empty());
    }
}

TEST_CASE("Global reconstruction lowers complete noisy cultivation circuits",
          "[reconstruction][expensive]") {
    for (uint32_t distance : {5, 7}) {
        const auto source = parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/reconstruction_msc_d" +
                                       std::to_string(distance) + ".stim");
        for (auto mode : {ReconstructionOutputs::Records, ReconstructionOutputs::Detectors}) {
            const auto r = reconstruct_global_sampling(source, mode);
            INFO(r.fallback_reason);
            REQUIRE(r.reconstruction.has_value());
            REQUIRE(r.reconstruction->peak_active_width == (distance == 5 ? 6 : 8));
            auto lowered = lower_global_reconstruction(*r.reconstruction);
            INFO(lowered.fallback_reason);
            REQUIRE(lowered.hir.has_value());
            default_hir_pass_manager().run(*lowered.hir);
            const sampling::ExecutablePlan executable(sampling::plan_sampling(*lowered.hir));
            REQUIRE(executable.peak_active_width() <= (distance == 5 ? 6 : 8));
            REQUIRE(executable.num_visible_records() ==
                    (mode == ReconstructionOutputs::Records ? source.num_measurements : 0));
            REQUIRE(executable.num_detectors() == source.num_detectors);
            REQUIRE(executable.num_observables() == source.num_observables);
            REQUIRE_FALSE(executable.supports_final_state_queries());
            const auto scalar = sampling::sample(executable, 65, 9, 1, {}, 1);
            const auto packed = sampling::sample(executable, 65, 9, 1, {}, 64);
            REQUIRE(scalar.measurements.size() == 65 * executable.num_visible_records());
            REQUIRE(packed.measurements.size() == scalar.measurements.size());
            REQUIRE(packed.detectors.size() == 65 * source.num_detectors);
            REQUIRE(packed.observables.size() == 65 * source.num_observables);
        }
    }
}

}  // namespace clifft::test
