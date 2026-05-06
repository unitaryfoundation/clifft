#pragma once

#include "clifft/optimizer/hir_pass.h"

namespace clifft {

/// Drops non-evolution HIR operations so the remaining circuit can be queried
/// as a unitary skeleton. This is not semantics-preserving.
class DropNonUnitaryPass : public HirPass {
  public:
    void run(HirModule& hir) override;
};

}  // namespace clifft
