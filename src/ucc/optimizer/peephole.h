#pragma once

#include "ucc/optimizer/pass.h"

#include <cstddef>

namespace ucc {

/// Peephole fusion pass: scans the HIR to algebraically cancel or fuse
/// T/T_dag gates acting on the same virtual Pauli axis using the
/// symplectic inner product as a commutation check.
class PeepholeFusionPass : public Pass {
  public:
    void run(HirModule& hir) override;

    /// Statistics from the last run.
    size_t cancellations() const { return cancellations_; }
    size_t fusions() const { return fusions_; }

  private:
    size_t cancellations_ = 0;
    size_t fusions_ = 0;
};

}  // namespace ucc
