#include "clifft/api/reference_syndrome.h"

#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

namespace clifft {

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir) {
    ReferenceSyndrome ref;

    // Make a clean copy and strip all noise
    HirModule clean_hir = hir;
    RemoveNoisePass strip;
    strip.run(clean_hir);

    // Compile without postselection or expected parities.
    sampling::ExecutablePlan clean_program(sampling::plan_sampling(clean_hir));

    if (clean_program.num_visible_records() == 0 && clean_program.num_detectors() == 0 &&
        clean_program.num_observables() == 0) {
        return ref;
    }

    // Run exactly one deterministic shot (seed=0)
    auto clean_res = sampling::sample(clean_program, 1, uint64_t{0});
    ref.detectors = std::move(clean_res.detectors);
    ref.observables = std::move(clean_res.observables);
    return ref;
}

}  // namespace clifft
