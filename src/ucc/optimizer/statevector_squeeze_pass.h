#pragma once

#include "ucc/optimizer/hir_pass.h"

namespace ucc {

/// Minimizes the active spacetime volume by reordering HIR operations:
/// - Sweep 1 (leftward): bubbles MEASURE ops as early as possible
/// - Sweep 2 (rightward): bubbles T_GATE and PHASE_ROTATION
///   ops as late as possible
///
/// This reduces peak_rank by compacting qubit lifetimes: measurements free
/// active dimensions sooner, and non-Clifford expansions are deferred.
class StatevectorSqueezePass : public HirPass {
  public:
    void run(HirModule& hir) override;
};

}  // namespace ucc
