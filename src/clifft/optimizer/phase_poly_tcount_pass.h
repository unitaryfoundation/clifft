#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/mcr_tcount.h"

#include <cstddef>

namespace clifft {

/// Experimental global T-count pass for issue #40: bounded MCR reordering plus
/// size-capped TODD on commuting phase-polynomial clusters. Opt-in only.
class ExperimentalGlobalTcountPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    [[nodiscard]] size_t t_gates_before() const { return t_gates_before_; }
    [[nodiscard]] size_t t_gates_after() const { return t_gates_after_; }
    [[nodiscard]] const McrTcountStats& mcr_stats() const { return mcr_stats_; }
    [[nodiscard]] size_t todd_blocks_optimized() const { return todd_blocks_; }
    [[nodiscard]] size_t todd_t_removed() const { return todd_t_removed_; }

  private:
    size_t t_gates_before_ = 0;
    size_t t_gates_after_ = 0;
    McrTcountStats mcr_stats_{};
    size_t todd_blocks_ = 0;
    size_t todd_t_removed_ = 0;
};

}  // namespace clifft
