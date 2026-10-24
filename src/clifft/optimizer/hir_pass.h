#pragma once

#include "clifft/frontend/hir.h"

namespace clifft {

/// Abstract base class for HIR optimization passes.
class HirPass {
  public:
    virtual void run(HirModule& hir) = 0;
    // Source-level reconstruction is unsafe after an intentional change of law
    // such as removing noise. Unknown passes must opt in to this contract.
    virtual bool preserves_sampling_distribution() const { return false; }
    virtual ~HirPass() = default;
};

}  // namespace clifft
