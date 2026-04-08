#include "clifft/backend/reference_syndrome.h"

#include "clifft/backend/backend.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/svm/svm.h"

namespace clifft {

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir) {
    ReferenceSyndrome ref;

    // Make a clean copy and strip all noise
    HirModule clean_hir = hir;
    RemoveNoisePass strip;
    strip.run(clean_hir);

    // Lower without postselection or expected parities
    auto clean_prog = lower(clean_hir);

    if (clean_prog.num_measurements == 0 && clean_prog.num_detectors == 0 &&
        clean_prog.num_observables == 0) {
        return ref;
    }

    // Run exactly one deterministic shot (seed=0)
    auto clean_res = sample(clean_prog, 1, uint64_t{0});
    ref.detectors = std::move(clean_res.detectors);
    ref.observables = std::move(clean_res.observables);
    return ref;
}

}  // namespace clifft
