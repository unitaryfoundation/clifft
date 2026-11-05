#pragma once

#include "clifft/frontend/hir.h"

namespace clifft {

/// Abstract base class for HIR optimization passes.
class HirPass {
  public:
    virtual void run(HirModule& hir) = 0;
    // Source-level reconstruction requires passes to preserve the sampling
    // distribution. Passes that remove noise or have unknown behavior opt out.
    virtual bool preserves_sampling_distribution() const { return false; }
    virtual ~HirPass() = default;
};

}  // namespace clifft
