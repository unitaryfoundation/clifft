#pragma once

#include "clifft/optimizer/hir_pass.h"

namespace clifft {

/// Drops non-unitary HIR operations so the remaining circuit can be queried
/// as a unitary skeleton. This is not semantics-preserving.
class MakeUnitaryPass : public HirPass {
  public:
    void run(HirModule& hir) override;
};

}  // namespace clifft
