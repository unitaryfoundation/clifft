#pragma once

#include "clifft/optimizer/hir_pass.h"

namespace clifft {

/// Minimizes the active spacetime volume by reordering HIR operations:
/// - Sweep 1 (leftward): bubbles MEASURE ops as early as possible
/// - Sweep 2 (rightward): bubbles T_GATE and PHASE_ROTATION
///   ops as late as possible
///
/// This reduces peak active width by compacting qubit lifetimes: measurements free
/// active coordinates sooner, and non-Clifford expansions are deferred.
class StatevectorSqueezePass : public HirPass {
  public:
    StatevectorSqueezePass() = default;

    [[nodiscard]] static StatevectorSqueezePass with_reversed_commuting_expansions() {
        return StatevectorSqueezePass{true};
    }

    void run(HirModule& hir) override;

  private:
    explicit StatevectorSqueezePass(bool reverse_commuting_expansions)
        : reverse_commuting_expansions_(reverse_commuting_expansions) {}

    bool reverse_commuting_expansions_ = false;
};

}  // namespace clifft
