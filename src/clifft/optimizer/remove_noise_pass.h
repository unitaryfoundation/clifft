#pragma once

#include "clifft/optimizer/hir_pass.h"

namespace clifft {

/// Strips all stochastic noise and readout noise ops from the HIR.
/// Also clears the noise_sites and readout_noise side-tables while filtering
/// a parallel source map when one is available.
///
/// This pass is NOT included in the default pass list. It is used
/// internally by compute_reference_syndrome() to produce a clean
/// noiseless copy of the circuit for reference-shot extraction.
class RemoveNoisePass : public HirPass {
  public:
    void run(HirModule& hir) override;
};

}  // namespace clifft
