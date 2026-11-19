#include "clifft/api/reference_syndrome.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {
namespace {

// One scalar shot through width 21 uses at most 48 MiB for coefficients and
// collapse scratch. Keep the inexpensive sequential reference and its seeded
// outcomes. For wider states, try a smaller detector plan before allocating.
constexpr uint32_t kMaxDirectReferenceWidth = 21;

ReconstructionLoweringResult reconstruct_noiseless_detectors(const Circuit& source,
                                                             const ReconstructionLimits& limits) {
    Circuit noiseless = source.metadata_only_copy();
    noiseless.nodes.reserve(source.nodes.size());
    for (const AstNode& node : source.nodes) {
        if (!is_noise_gate(node.gate)) {
            noiseless.nodes.push_back(node);
        }
    }
    auto result = reconstruct_global_sampling(noiseless, ReconstructionOutputs::Detectors, limits);
    if (!result.reconstruction) {
        return {std::nullopt, std::move(result.fallback_reason)};
    }
    return lower_global_reconstruction(*result.reconstruction, {}, limits);
}

}  // namespace

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir,
                                             const Circuit* reconstruction_source,
                                             const ReconstructionLimits& limits) {
    const HirModule& source_hir =
        hir.postselection.pre_postselection_hir ? *hir.postselection.pre_postselection_hir : hir;

    if (source_hir.num_detectors == 0 && source_hir.num_observables == 0) {
        return {};
    }

    // Make a clean copy and strip all noise
    HirModule clean_hir = source_hir;
    clean_hir.postselection.clear();
    RemoveNoisePass strip;
    strip.run(clean_hir);
    const bool has_non_clifford = std::ranges::any_of(clean_hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
    });
    if (has_non_clifford) {
        default_hir_pass_manager(/*include_schedule=*/false).run(clean_hir);
    }

    // Compile without postselection or expected parities.
    std::optional<sampling::ExecutablePlan> clean_program;
    std::string overflow_reason;
    try {
        clean_program.emplace(sampling::plan_sampling(clean_hir));
    } catch (const std::overflow_error& overflow) {
        if (reconstruction_source == nullptr) {
            throw;
        }
        if (clean_hir.num_exp_vals != 0) {
            throw std::overflow_error(
                std::string(overflow.what()) +
                " (reconstruction fallback: EXP_VAL requires the original state evolution)");
        }
        overflow_reason = overflow.what();
    }

    const uint32_t width_limit =
        clean_program ? clean_program->peak_active_width() : kDenseActiveWidthLimit;
    if (reconstruction_source && clean_hir.num_exp_vals == 0 &&
        width_limit > kMaxDirectReferenceWidth) {
        auto lowered = reconstruct_noiseless_detectors(*reconstruction_source, limits);
        if (lowered.hir) {
            default_hir_pass_manager(/*include_schedule=*/false).run(*lowered.hir);
            if (summarize_active_width(*lowered.hir).peak_width < width_limit) {
                clean_program.emplace(sampling::plan_sampling(*lowered.hir));
            } else if (!clean_program) {
                lowered.fallback_reason = "reconstruction reaches the dense active-width limit";
            }
        }
        if (!clean_program) {
            throw std::overflow_error(
                overflow_reason + " (reconstruction fallback: " + lowered.fallback_reason + ")");
        }
    }

    // A fixed seed selects a reproducible shot, not necessarily fixed parities.
    auto clean_res = sampling::sample(*clean_program, 1, uint64_t{0});
    return {std::move(clean_res.detectors), std::move(clean_res.observables)};
}

}  // namespace clifft
