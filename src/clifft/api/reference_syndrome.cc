#include "clifft/api/reference_syndrome.h"

#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/phase_polynomial_pass.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/rotation_simplification_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

namespace clifft {

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir) {
    ReferenceSyndrome ref;

    const HirModule& source_hir = hir.pre_postselection_hir ? *hir.pre_postselection_hir : hir;
    const bool had_noise = !source_hir.noise_sites.empty() || !source_hir.readout_noise.empty();

    // Make a clean copy and strip all noise
    HirModule clean_hir = source_hir;
    clean_hir.pre_postselection_hir.reset();
    clean_hir.postselection_mask.clear();
    clean_hir.expected_detectors.clear();
    clean_hir.expected_observables.clear();
    RemoveNoisePass strip;
    strip.run(clean_hir);
    if ((had_noise || clean_hir.num_t_gates() > 64) && clean_hir.num_t_gates() > 0) {
        HirPassManager passes;
        passes.add_pass(std::make_unique<PeepholeFusionPass>());
        passes.add_pass(std::make_unique<PhasePolynomialPass>());
        passes.add_pass(std::make_unique<RotationSimplificationPass>());
        passes.add_pass(std::make_unique<StatevectorSqueezePass>());
        passes.run(clean_hir);
    }

    // Compile without postselection or expected parities.
    sampling::ExecutablePlan clean_program(sampling::plan_sampling(clean_hir));

    if (clean_program.num_visible_records() == 0 && clean_program.num_detectors() == 0 &&
        clean_program.num_observables() == 0) {
        return ref;
    }

    // A fixed seed selects a reproducible shot, not necessarily fixed parities.
    auto clean_res = sampling::sample(clean_program, 1, uint64_t{0});
    ref.detectors = std::move(clean_res.detectors);
    ref.observables = std::move(clean_res.observables);
    return ref;
}

}  // namespace clifft
