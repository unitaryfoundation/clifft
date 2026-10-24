#include "clifft/api/program.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/reconstruction_lowering.h"
#include "clifft/sampling/planner.h"

#include <utility>

namespace clifft {

Program::Program(sampling::ExecutablePlan original) : original_(std::move(original)) {}

Program Program::compile(const Circuit& source, const HirModule& original_hir, bool reconstruction,
                         bool passes_preserve_sampling, const ReconstructionLimits& limits) {
    Program program{sampling::ExecutablePlan(sampling::plan_sampling(original_hir))};
    if (!reconstruction) {
        return program;
    }
    for (bool keep_records : {false, true}) {
        auto& variant = program.variants_[keep_records];
        if (!passes_preserve_sampling) {
            variant.fallback_reason =
                "HIR passes do not certify preservation of the sampling distribution";
            continue;
        }
        if (program.original_.num_exp_vals() != 0) {
            variant.fallback_reason = "EXP_VAL requires the original state evolution";
            continue;
        }
        if (program.original_.peak_active_width() == 0) {
            variant.fallback_reason = "original active width is already zero";
            continue;
        }
        if ((keep_records ? source.num_measurements : 0) + source.num_detectors +
                source.num_observables ==
            0) {
            variant.fallback_reason = "no requested sampling outputs";
            continue;
        }
        auto result = reconstruct_global_sampling(
            source,
            keep_records ? ReconstructionOutputs::Records : ReconstructionOutputs::Detectors,
            limits);
        if (!result.reconstruction) {
            variant.fallback_reason = std::move(result.fallback_reason);
            continue;
        }
        auto lowered =
            lower_global_reconstruction(*result.reconstruction, original_hir.postselection, limits);
        if (!lowered.hir) {
            variant.fallback_reason = std::move(lowered.fallback_reason);
            continue;
        }
        // The reconstructed representation has its own optimization pipeline;
        // user pass objects and their diagnostic counters belong to the original.
        auto passes = default_hir_pass_manager();
        passes.run(*lowered.hir);
        // Reject a wider candidate before planning, which enforces the dense
        // backend's width limit and can otherwise reject an eligible source.
        if (analyze_active_width(*lowered.hir).peak_width >=
            program.original_.peak_active_width()) {
            variant.fallback_reason = "reconstruction does not reduce peak active width";
            continue;
        }
        auto plan = sampling::plan_sampling(*lowered.hir);
        variant.plan.emplace(plan);
        variant.fallback_reason.clear();
    }
    return program;
}

const sampling::ExecutablePlan& Program::sampling_plan(bool keep_records) const noexcept {
    const auto& candidate = variants_[keep_records].plan;
    return candidate ? *candidate : original_;
}

bool Program::uses_reconstruction(bool keep_records) const noexcept {
    return variants_[keep_records].plan.has_value();
}

const std::string& Program::reconstruction_fallback_reason(bool keep_records) const {
    return variants_[keep_records].fallback_reason;
}

}  // namespace clifft
