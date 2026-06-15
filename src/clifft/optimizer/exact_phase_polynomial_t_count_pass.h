#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>

namespace clifft {

/// Experimental exact T-count minimization for small commuting phase-polynomial blocks.
///
/// The pass is intentionally bounded: it optimizes only contiguous commuting T-gate
/// blocks whose independent Pauli-axis rank is at most `max_rank`. Within that
/// bound it exhaustively searches lower odd-parity representatives and accepts
/// only rewrites whose residual phase is Clifford.
class ExactPhasePolynomialTCountPass : public HirPass {
  public:
    explicit ExactPhasePolynomialTCountPass(uint8_t max_rank = 4);

    void run(HirModule& hir) override;

    [[nodiscard]] uint8_t max_rank() const { return max_rank_; }
    [[nodiscard]] size_t blocks_considered() const { return blocks_considered_; }
    [[nodiscard]] size_t blocks_optimized() const { return blocks_optimized_; }
    [[nodiscard]] size_t t_removed() const { return t_removed_; }

  private:
    uint8_t max_rank_;
    size_t blocks_considered_ = 0;
    size_t blocks_optimized_ = 0;
    size_t t_removed_ = 0;
};

}  // namespace clifft
