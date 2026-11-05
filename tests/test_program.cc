#include "clifft/api/program.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"

#include <catch2/catch_test_macros.hpp>

namespace clifft::test {

TEST_CASE("Program preserves the original plan across bounded reconstruction failures",
          "[reconstruction]") {
    const auto source = parse("H 0 1\nT 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir);
    REQUIRE(original.original().peak_active_width() == 2);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(original.uses_reconstruction(keep_records));
        REQUIRE(&original.sampling_plan(keep_records) == &original.original());
    }
    const auto selected = Program::compile(source, hir, true);
    for (bool keep_records : {false, true}) {
        REQUIRE(selected.uses_reconstruction(keep_records));
        REQUIRE(selected.sampling_plan(keep_records).peak_active_width() == 1);
    }
    REQUIRE(selected.original().inspect() == original.original().inspect());
    auto limits = ReconstructionLimits{};
    limits.max_dilation_qubits = 1;
    const auto bounded = Program::compile(source, hir, true, true, limits);
    const auto transformed = Program::compile(source, hir, true, false);
    for (const auto* fallback : {&bounded, &transformed}) {
        for (bool keep_records : {false, true}) {
            REQUIRE_FALSE(fallback->uses_reconstruction(keep_records));
            REQUIRE_FALSE(fallback->reconstruction_fallback_reason(keep_records).empty());
            REQUIRE(&fallback->sampling_plan(keep_records) == &fallback->original());
            REQUIRE(fallback->original().inspect() == original.original().inspect());
        }
    }
}

TEST_CASE("Program skips reconstruction lowering without a scheduled width reduction",
          "[reconstruction]") {
    const auto source = parse("H 0\nT 0\nT_DAG 0\nH 0\nM 0\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir);
    REQUIRE(original.original().peak_active_width() == 1);

    // Later passes can improve this candidate, but the scheduled width does
    // not justify their compilation cost under the default selection policy.
    for (auto outputs : {ReconstructionOutputs::Detectors, ReconstructionOutputs::Records}) {
        const auto result = reconstruct_global_sampling(source, outputs);
        REQUIRE(result.reconstruction);
        REQUIRE(result.reconstruction->peak_active_width == 1);
        auto lowered = lower_global_reconstruction(*result.reconstruction);
        REQUIRE(lowered.hir);
        default_hir_pass_manager().run(*lowered.hir);
        REQUIRE(analyze_active_width(*lowered.hir).peak_width == 0);
    }

    auto limits = ReconstructionLimits{};
    // A zero lowering budget makes an attempted lowering fail, so the fallback
    // reason also checks that the width guard runs before that work.
    limits.max_lowering_bytes = 0;
    const auto guarded = Program::compile(source, hir, true, true, limits);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(guarded.uses_reconstruction(keep_records));
        REQUIRE(guarded.reconstruction_fallback_reason(keep_records) ==
                "reconstruction schedule does not reduce peak active width");
        REQUIRE(&guarded.sampling_plan(keep_records) == &guarded.original());
    }
    REQUIRE(guarded.original().inspect() == original.original().inspect());
}

}  // namespace clifft::test
