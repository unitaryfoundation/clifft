#include "clifft/api/reference_syndrome.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/frontend/frontend.h"
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
                                             const ReconstructionLimits& limits, bool runtime_frame,
                                             const Circuit* frame_source) {
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
        overflow_reason = overflow.what();
    }
    if (runtime_frame &&
        (!clean_program || clean_program->peak_active_width() > kMaxDirectReferenceWidth)) {
        // Measurement outcomes still sign the T regions of a noiseless
        // circuit, and the frame planner reduces them where the sequential
        // plan cannot. It reads the circuit without the optimizations above,
        // and from before the caller's passes where the source is given.
        HirModule frame_hir = frame_source != nullptr ? trace(*frame_source) : source_hir;
        frame_hir.postselection.clear();
        strip.run(frame_hir);
        try {
            sampling::ExecutablePlan frame(
                sampling::plan_sampling(frame_hir, {.runtime_frame = true}));
            if (!clean_program || frame.peak_active_width() < clean_program->peak_active_width()) {
                clean_program.emplace(std::move(frame));
            }
        } catch (const std::overflow_error&) {}
    }
    if (!clean_program) {
        if (reconstruction_source == nullptr) {
            throw std::overflow_error(overflow_reason);
        }
        if (clean_hir.num_exp_vals != 0) {
            throw std::overflow_error(
                overflow_reason +
                " (reconstruction fallback: EXP_VAL requires the original state evolution)");
        }
    }

    // A frame plan's states start at the width its static actions need, and
    // a noiseless shot seldom grows them toward the plan's bound.
    const uint32_t width_limit =
        clean_program ? clean_program->initial_capacity_width() : kDenseActiveWidthLimit;
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
