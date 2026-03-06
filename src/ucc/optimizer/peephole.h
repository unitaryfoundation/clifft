#pragma once

#include "ucc/optimizer/pass.h"

namespace ucc {

/// Peephole fusion pass: scans the HIR to algebraically cancel or fuse
/// T/T_dag gates acting on the same virtual Pauli axis using the
/// symplectic inner product as a commutation check.
class PeepholeFusionPass : public Pass {
  public:
    void run(HirModule& hir) override;
};

}  // namespace ucc
