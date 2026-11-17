#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Peephole fusion pass: scans the HIR to algebraically cancel or fuse
/// T/T_dag gates acting on the same virtual Pauli axis, and removes Pauli
/// phases consumed by a later same-axis measurement.
class PeepholeFusionPass : public HirPass {
  public:
    void run(HirModule& hir) override;
    bool preserves_sampling_distribution() const override { return true; }

    /// Statistics from the last run.
    size_t cancellations() const { return cancellations_; }
    size_t fusions() const { return fusions_; }

  private:
    size_t cancellations_ = 0;
    size_t fusions_ = 0;
};

}  // namespace clifft
