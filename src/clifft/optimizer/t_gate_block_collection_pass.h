#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Opt-in pre-pass for exact phase-polynomial T-count reduction.
///
/// The pass uses only adjacent swaps approved by can_swap() to pull later
/// T_GATE ops into the current commuting T block. It does not rewrite gates or
/// change T count by itself; it only exposes larger contiguous blocks for a
/// later T-count pass.
class TGateBlockCollectionPass : public HirPass {
  public:
    /// Maximum number of consecutive non-T ops inspected while looking for each
    /// next T candidate to pull into the current block.
    explicit TGateBlockCollectionPass(size_t max_scan = 64) : max_scan_(max_scan) {}

    void run(HirModule& hir) override;

    [[nodiscard]] size_t max_scan() const { return max_scan_; }
    [[nodiscard]] size_t blocks_collected() const { return blocks_collected_; }
    [[nodiscard]] size_t t_gates_moved() const { return t_gates_moved_; }
    [[nodiscard]] size_t adjacent_swaps() const { return adjacent_swaps_; }

  private:
    size_t max_scan_;
    size_t blocks_collected_ = 0;
    size_t t_gates_moved_ = 0;
    size_t adjacent_swaps_ = 0;
};

}  // namespace clifft
