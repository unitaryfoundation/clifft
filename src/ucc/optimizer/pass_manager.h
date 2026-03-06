#pragma once

#include "ucc/optimizer/pass.h"

#include <memory>
#include <vector>

namespace ucc {

/// Runs a sequence of optimization passes over an HirModule.
///
/// Passes execute in the order they were added. Each pass receives
/// the HirModule mutated by all prior passes.
class PassManager {
  public:
    void add_pass(std::unique_ptr<Pass> pass);
    void run(HirModule& hir);

  private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace ucc
