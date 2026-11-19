#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <memory>
#include <vector>

namespace clifft {

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
    // The caller keeps a borrowed pass alive for the lifetime of the manager.
    void add_pass(HirPass& pass);
    void run(HirModule& hir);
    [[nodiscard]] bool preserves_sampling_distribution() const;

  private:
    std::vector<std::unique_ptr<HirPass>> owned_;
    std::vector<HirPass*> passes_;
};

}  // namespace clifft
