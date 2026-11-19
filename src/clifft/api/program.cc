#include "clifft/api/program.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace clifft {

struct Program::Data {
    struct SamplingVariant {
        std::optional<sampling::ExecutablePlan> plan;
        std::string fallback_reason = "reconstruction is disabled";
    };

    struct EarlyCheckPreparations {
        std::array<detail::LazyPrepared<SamplingPreparation>, 2> by_early_postselection;
    };

    struct SummaryPreparations {
        std::array<EarlyCheckPreparations, 2> by_survivor_counts;
    };

    struct OutputPreparations {
        std::array<SummaryPreparations, 2> by_fixed_faults;
    };

    explicit Data(sampling::ExecutablePlan plan)
        : num_qubits(plan.num_qubits()),
          num_visible_records(plan.num_visible_records()),
          num_hidden_records(plan.num_hidden_records()),
          num_detectors(plan.num_detectors()),
          num_observables(plan.num_observables()),
          num_exp_vals(plan.num_exp_vals()),
          has_postselection(plan.has_postselection()),
          original(std::move(plan)) {}

    Data(const HirModule& hir, std::string overflow_reason)
        : num_qubits(hir.num_qubits),
          num_visible_records(hir.num_measurements),
          num_hidden_records(hir.num_hidden_measurements),
          num_detectors(hir.num_detectors),
          num_observables(hir.num_observables),
          num_exp_vals(hir.num_exp_vals),
          has_postselection(std::ranges::any_of(
              std::span(hir.postselection.mask)
                  .first(std::min<size_t>(hir.num_detectors, hir.postselection.mask.size())),
              [](uint8_t flag) { return flag != 0; })),
          original_overflow_reason(std::move(overflow_reason)) {}

    uint32_t num_qubits = 0;
    uint32_t num_visible_records = 0;
    uint32_t num_hidden_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    bool has_postselection = false;

    std::optional<sampling::ExecutablePlan> original;
    std::string original_overflow_reason;
    // Prepared once during compilation, then immutable for concurrent sampling.
    std::array<SamplingVariant, 2> variants;
    std::array<OutputPreparations, 4> preparations;
};

Program::Program(sampling::ExecutablePlan original)
    : data_(std::make_shared<const Data>(std::move(original))) {}

Program::Program(std::shared_ptr<const Data> data) : data_(std::move(data)) {}

Program Program::compile(const Circuit& source, const HirModule& original_hir,
                         std::optional<bool> reconstruction, bool passes_preserve_sampling,
                         const ReconstructionLimits& limits) {
    if (reconstruction == false) {
        return Program(sampling::ExecutablePlan(sampling::plan_sampling(original_hir)));
    }
    std::shared_ptr<Data> data;
    try {
        data =
            std::make_shared<Data>(sampling::ExecutablePlan(sampling::plan_sampling(original_hir)));
    } catch (const std::overflow_error& overflow) {
        data = std::make_shared<Data>(original_hir, overflow.what());
    }

    const uint32_t width_limit =
        data->original ? data->original->peak_active_width() : kDenseActiveWidthLimit;
    // Spend the full scheduling budget when a dense original state already costs
    // over a MiB in coefficients, or reconstruction is the only sampling path.
    auto budget = limits;
    constexpr uint32_t kSmallOriginalWidth = 16;
    constexpr uint32_t kSmallScheduleBudget = 2048;
    if (!reconstruction.has_value() && width_limit <= kSmallOriginalWidth) {
        budget.max_schedule_operations =
            std::min(budget.max_schedule_operations, kSmallScheduleBudget);
    }
    const bool compare_cost = !reconstruction.has_value() && data->original.has_value();
    const bool has_syndrome_outputs = source.num_detectors != 0 || source.num_observables != 0;
    std::optional<GlobalSamplingReconstructor> reconstructor;
    for (bool keep_records : {false, true}) {
        auto& variant = data->variants[keep_records];
        if (!passes_preserve_sampling) {
            variant.fallback_reason =
                "HIR passes do not certify preservation of the sampling distribution";
            continue;
        }
        if (data->num_exp_vals != 0) {
            variant.fallback_reason = "EXP_VAL requires the original state evolution";
            continue;
        }
        if (data->original && data->original->peak_active_width() == 0) {
            variant.fallback_reason = "original active width is already zero";
            continue;
        }
        if (!has_syndrome_outputs && (!keep_records || source.num_measurements == 0)) {
            variant.fallback_reason = "no requested sampling outputs";
            continue;
        }
        if (!reconstructor) {
            reconstructor.emplace(source, budget);
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
        // A tied schedule can become narrower after HIR optimization. Skip wider
        // schedules when an original plan exists. On overflow, the only viable
        // plan can emerge after lowering even if its schedule is too wide.
        if (data->original && result.reconstruction->peak_active_width > width_limit) {
            variant.fallback_reason = "reconstruction schedule does not reduce peak active width";
            continue;
        }
        auto lowered =
            lower_global_reconstruction(*result.reconstruction, original_hir.postselection, budget);
        result.reconstruction.reset();
        if (!lowered.hir) {
            variant.fallback_reason = std::move(lowered.fallback_reason);
            continue;
        }
        // Reconstructed HIR is already scheduled; run the remaining default
        // passes so caller-owned pass objects reflect only the original HIR.
        auto passes = default_hir_pass_manager(/*include_schedule=*/false);
        passes.run(*lowered.hir);
        // Check peak width before plan_sampling(), which enforces the active-width
        // cap and would otherwise throw on a wide candidate.
        const uint32_t width = summarize_active_width(*lowered.hir).peak_width;
        if (width > width_limit || (width == width_limit && !compare_cost)) {
            variant.fallback_reason = data->original
                                          ? "reconstruction does not reduce peak active width"
                                          : "reconstruction reaches the dense active-width limit";
            continue;
        }
        sampling::ExecutablePlan executable(sampling::plan_sampling(*lowered.hir));
        lowered.hir.reset();
        // Under postselection on narrow originals, small width drops that
        // substantially increase plan actions add more trajectory splits than
        // the narrower state saves.
        constexpr uint32_t kMinPostselectedWidthReduction = 3;
        if (compare_cost && data->has_postselection && width != 0 && width < width_limit &&
            width_limit <= kSmallOriginalWidth &&
            width_limit - width < kMinPostselectedWidthReduction &&
            uint64_t{executable.num_actions()} * 4 > uint64_t{data->original->num_actions()} * 5) {
            variant.fallback_reason =
                "postselected reconstruction saves at most two qubits while increasing actions";
            continue;
        }
        if (compare_cost && width == width_limit) {
            const auto work = [](const sampling::ExecutablePlan& p) {
                return saturating_add_u64(
                    p.estimated_batch_lane_work(sampling::BatchOutputMode::AggregateSurvivors),
                    uint64_t{p.num_actions()} + p.num_noise_sites());
            };
            if (executable.num_actions() > data->original->num_actions() ||
                work(executable) >= work(*data->original)) {
                variant.fallback_reason = "original plan has lower or equal estimated work";
                continue;
            }
        }
        variant.plan.emplace(std::move(executable));
        variant.fallback_reason.clear();
    }

    if (!data->original && !data->variants[false].plan && !data->variants[true].plan) {
        const std::string& det_reason = data->variants[false].fallback_reason;
        const std::string& rec_reason = data->variants[true].fallback_reason;
        const std::string detail = (det_reason == rec_reason || !has_syndrome_outputs)
                                       ? rec_reason
                                       : "syndromes: " + det_reason + "; records: " + rec_reason;
        throw std::overflow_error(data->original_overflow_reason +
                                  " (reconstruction fallback: " + detail + ")");
    }
    return Program(std::move(data));
}

bool Program::has_original() const noexcept {
    return data_->original.has_value();
}

bool Program::has_sampling_plan(bool keep_records) const noexcept {
    return data_->variants[keep_records].plan.has_value() || data_->original.has_value();
}

const sampling::ExecutablePlan& Program::original() const {
    if (!data_->original) {
        throw std::overflow_error(
            "original plan is unavailable: " + data_->original_overflow_reason +
            "; exact queries, replay, and fixed-fault sampling require the original plan");
    }
    return *data_->original;
}

const sampling::ExecutablePlan& Program::sampling_plan(bool keep_records) const {
    if (const auto& candidate = data_->variants[keep_records].plan) {
        return *candidate;
    }
    if (data_->original) {
        return *data_->original;
    }
    const char* variant_name =
        keep_records ? "full-record sampling plan" : "detector/observable sampling plan";
    const char* alternative = keep_records ? "keep_records=False or keep_records='syndromes'"
                                           : "keep_records=True or keep_records='measurements'";
    throw std::overflow_error(
        std::string(variant_name) + " is unavailable: " + data_->original_overflow_reason +
        " (reconstruction fallback: " + data_->variants[keep_records].fallback_reason + "); use " +
        alternative);
}

bool Program::uses_reconstruction(bool keep_records) const noexcept {
    return data_->variants[keep_records].plan.has_value();
}

const std::string& Program::reconstruction_fallback_reason(bool keep_records) const {
    return data_->variants[keep_records].fallback_reason;
}

uint32_t Program::num_qubits() const noexcept {
    return data_->num_qubits;
}

uint32_t Program::num_visible_records() const noexcept {
    return data_->num_visible_records;
}

uint32_t Program::num_hidden_records() const noexcept {
    return data_->num_hidden_records;
}

uint32_t Program::num_detectors() const noexcept {
    return data_->num_detectors;
}

uint32_t Program::num_observables() const noexcept {
    return data_->num_observables;
}

uint32_t Program::num_exp_vals() const noexcept {
    return data_->num_exp_vals;
}

bool Program::has_postselection() const noexcept {
    return data_->has_postselection;
}

const SamplingPreparation& Program::sampling_preparation(sampling::SamplingOutputs outputs,
                                                         SamplingPreparationOptions options) const {
    const auto& plan = options.fixed_faults ? original() : sampling_plan(outputs.measurements());
    const bool separate_fixed_slot =
        options.fixed_faults && uses_reconstruction(outputs.measurements());
    const bool early = options.early_postselection && plan.has_postselection();
    const bool survivor_counts =
        !outputs.rows() ||
        (outputs.records == sampling::RecordOutputs::Measurements && options.survivor_counts);
    return data_->preparations[static_cast<size_t>(outputs.records)]
        .by_fixed_faults[separate_fixed_slot]
        .by_survivor_counts[survivor_counts]
        .by_early_postselection[early]
        .get(plan, outputs, early, survivor_counts);
}

}  // namespace clifft
