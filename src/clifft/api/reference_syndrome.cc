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

namespace clifft {

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir,
                                             const Circuit* reconstruction_source,
                                             const ReconstructionLimits& limits) {
    ReferenceSyndrome ref;

    const HirModule& source_hir =
        hir.postselection.pre_postselection_hir ? *hir.postselection.pre_postselection_hir : hir;

    // Make a clean copy and strip all noise
    HirModule clean_hir = source_hir;
    clean_hir.postselection.clear();
    RemoveNoisePass strip;
    strip.run(clean_hir);
    const bool has_non_clifford = std::ranges::any_of(clean_hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
    });
    if (has_non_clifford) {
        default_hir_pass_manager().run(clean_hir);
    }

    // Compile without postselection or expected parities.
    std::optional<sampling::ExecutablePlan> clean_program;
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
        if (clean_hir.num_detectors == 0 && clean_hir.num_observables == 0) {
            return ref;
        }
        Circuit noiseless_source = reconstruction_source->metadata_only_copy();
        noiseless_source.nodes.reserve(reconstruction_source->nodes.size());
        for (const AstNode& node : reconstruction_source->nodes) {
            if (!is_noise_gate(node.gate)) {
                noiseless_source.nodes.push_back(node);
            }
        }
        const auto recon =
            reconstruct_global_sampling(noiseless_source, ReconstructionOutputs::Detectors, limits);
        if (!recon.reconstruction) {
            throw std::overflow_error(std::string(overflow.what()) +
                                      " (reconstruction fallback: " + recon.fallback_reason + ")");
        }
        auto lowered = lower_global_reconstruction(*recon.reconstruction, {}, limits);
        if (!lowered.hir) {
            throw std::overflow_error(std::string(overflow.what()) + " (reconstruction fallback: " +
                                      lowered.fallback_reason + ")");
        }
        default_hir_pass_manager().run(*lowered.hir);
        if (summarize_active_width(*lowered.hir).peak_width >= kDenseActiveWidthLimit) {
            throw std::overflow_error(
                std::string(overflow.what()) +
                " (reconstruction fallback: reconstruction reaches the dense active-width limit)");
        }
        clean_program.emplace(sampling::plan_sampling(*lowered.hir));
    }

    if (clean_program->num_visible_records() == 0 && clean_program->num_detectors() == 0 &&
        clean_program->num_observables() == 0) {
        return ref;
    }

    // A fixed seed selects a reproducible shot, not necessarily fixed parities.
    auto clean_res = sampling::sample(*clean_program, 1, uint64_t{0});
    ref.detectors = std::move(clean_res.detectors);
    ref.observables = std::move(clean_res.observables);
    return ref;
}

}  // namespace clifft
