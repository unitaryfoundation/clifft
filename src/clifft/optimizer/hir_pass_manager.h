#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <functional>
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
    void run(HirModule& hir);

    /// Run the pass sequence per fence-delimited segment. Ops for which
    /// `is_fence` returns true are structural optimization barriers: they
    /// are never handed to a pass, and no pass can observe, fuse, or move
    /// operations across one, because each maximal fence-free segment is
    /// presented to the passes as if it were the module's entire op stream.
    /// Module-level state (arenas, side tables, counters, global weight)
    /// stays in place, so mask handles and side-table indices remain valid.
    /// Equivalent to run() when no op is a fence.
    void run_segmented(HirModule& hir, const std::function<bool(const HeisenbergOp&)>& is_fence);

  private:
    std::vector<std::unique_ptr<HirPass>> passes_;
};

}  // namespace clifft
