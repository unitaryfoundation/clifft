#pragma once

#include "ucc/optimizer/hir_pass.h"

#include <memory>
#include <vector>

namespace ucc {

/// Runs a sequence of optimization passes over an HirModule.
///
/// Passes execute in the order they were added. Each pass receives
/// the HirModule mutated by all prior passes.
class HirPassManager {
  public:
    HirPassManager() = default;
    HirPassManager(HirPassManager&&) = default;
    HirPassManager& operator=(HirPassManager&&) = default;
    HirPassManager(const HirPassManager&) = delete;
    HirPassManager& operator=(const HirPassManager&) = delete;

    void add_pass(std::unique_ptr<HirPass> pass);
    void run(HirModule& hir);

  private:
    std::vector<std::unique_ptr<HirPass>> passes_;
};

}  // namespace ucc
