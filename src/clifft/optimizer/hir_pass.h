#pragma once

#include "clifft/frontend/hir.h"

namespace clifft {

/// Abstract base class for HIR optimization passes.
///
/// Each pass receives a mutable HirModule and may rewrite, reorder,
/// or remove operations. Passes operate on the timeless Heisenberg IR
/// using purely symplectic geometry -- no DAG or time assumptions.
class HirPass {
  public:
    virtual void run(HirModule& hir) = 0;
    virtual ~HirPass() = default;
};

}  // namespace clifft
