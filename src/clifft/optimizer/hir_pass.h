#pragma once

#include "clifft/frontend/hir.h"

namespace clifft {

/// Abstract base class for HIR optimization passes.
class HirPass {
  public:
    virtual void run(HirModule& hir) = 0;
    virtual ~HirPass() = default;
};

}  // namespace clifft
