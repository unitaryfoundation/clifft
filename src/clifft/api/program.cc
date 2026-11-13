#include "clifft/api/program.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include <array>
#include <optional>
#include <utility>

namespace clifft {

struct Program::Data {
    struct SamplingVariant {
        std::optional<sampling::ExecutablePlan> plan;
        std::string fallback_reason = "reconstruction is disabled";
    };

    explicit Data(sampling::ExecutablePlan plan) : original(std::move(plan)) {}

    sampling::ExecutablePlan original;
    // Prepared once during compilation, then immutable for concurrent sampling.
    std::array<SamplingVariant, 2> variants;
    // Output choice, fixed-fault plan choice, survivor summaries, and early checks.
    std::array<
        std::array<std::array<std::array<detail::LazyPrepared<SamplingPreparation>, 2>, 2>, 2>, 4>
        preparations;
};

Program::Program(sampling::ExecutablePlan original)
    : data_(std::make_shared<const Data>(std::move(original))) {}

Program::Program(std::shared_ptr<const Data> data) : data_(std::move(data)) {}

Program Program::compile(const Circuit& source, const HirModule& original_hir, bool reconstruction,
                         bool passes_preserve_sampling, const ReconstructionLimits& limits) {
    auto data =
        std::make_shared<Data>(sampling::ExecutablePlan(sampling::plan_sampling(original_hir)));
    if (!reconstruction) {
        return Program(std::move(data));
    }
    const auto& original = data->original;
    std::optional<GlobalSamplingReconstructor> reconstructor;
    for (bool keep_records : {false, true}) {
        auto& variant = data->variants[keep_records];
        if (!passes_preserve_sampling) {
            variant.fallback_reason =
                "HIR passes do not certify preservation of the sampling distribution";
            continue;
        }
        if (original.num_exp_vals() != 0) {
            variant.fallback_reason = "EXP_VAL requires the original state evolution";
            continue;
        }
        if (original.peak_active_width() == 0) {
            variant.fallback_reason = "original active width is already zero";
            continue;
        }
        if ((keep_records ? source.num_measurements : 0) + source.num_detectors +
                source.num_observables ==
            0) {
            variant.fallback_reason = "no requested sampling outputs";
            continue;
        }
        if (!reconstructor) {
            reconstructor.emplace(source, limits);
        }
        auto result = reconstructor->reconstruct(keep_records ? ReconstructionOutputs::Records
                                                              : ReconstructionOutputs::Detectors);
        if (keep_records) {
            reconstructor.reset();
        }
        if (!result.reconstruction) {
            variant.fallback_reason = std::move(result.fallback_reason);
            continue;
        }
        // Avoid expensive lowering and HIR passes without a scheduled width
        // reduction. This cost policy can skip reductions found by later passes.
        if (result.reconstruction->peak_active_width >= original.peak_active_width()) {
            variant.fallback_reason = "reconstruction schedule does not reduce peak active width";
            continue;
        }
        auto lowered =
            lower_global_reconstruction(*result.reconstruction, original_hir.postselection, limits);
        result.reconstruction.reset();
        if (!lowered.hir) {
            variant.fallback_reason = std::move(lowered.fallback_reason);
            continue;
        }
        // Reconstructed HIR runs its own default pass pipeline so caller-owned
        // pass objects and their counters reflect only the original HIR.
        auto passes = default_hir_pass_manager();
        passes.run(*lowered.hir);
        // Check peak width before plan_sampling(), which enforces the active-width
        // cap and would otherwise throw on a wide candidate.
        if (summarize_active_width(*lowered.hir).peak_width >= original.peak_active_width()) {
            variant.fallback_reason = "reconstruction does not reduce peak active width";
            continue;
        }
        auto plan = sampling::plan_sampling(*lowered.hir);
        lowered.hir.reset();
        variant.plan.emplace(std::move(plan));
        variant.fallback_reason.clear();
    }
    return Program(std::move(data));
}

const sampling::ExecutablePlan& Program::original() const noexcept {
    return data_->original;
}

const sampling::ExecutablePlan& Program::sampling_plan(bool keep_records) const noexcept {
    const auto& candidate = data_->variants[keep_records].plan;
    return candidate ? *candidate : data_->original;
}

bool Program::uses_reconstruction(bool keep_records) const noexcept {
    return data_->variants[keep_records].plan.has_value();
}

const std::string& Program::reconstruction_fallback_reason(bool keep_records) const {
    return data_->variants[keep_records].fallback_reason;
}

const SamplingPreparation& Program::sampling_preparation(sampling::SamplingOutputs outputs,
                                                         SamplingPreparationOptions options) const {
    const bool fixed_faults = options.fixed_faults && uses_reconstruction(outputs.measurements());
    const auto& plan = fixed_faults ? original() : sampling_plan(outputs.measurements());
    const bool early = options.early_postselection && plan.has_postselection();
    const bool survivor_counts =
        !outputs.rows() ||
        (outputs.records == sampling::RecordOutputs::Measurements && options.survivor_counts);
    return data_
        ->preparations[static_cast<size_t>(outputs.records)][fixed_faults][survivor_counts][early]
        .get(plan, outputs, early, survivor_counts);
}

}  // namespace clifft
