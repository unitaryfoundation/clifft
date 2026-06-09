#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/mcr_reorder.h"

#include <cstddef>

namespace clifft {

class GlobalTcountPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    [[nodiscard]] size_t t_gates_before() const { return t_before_; }
    [[nodiscard]] size_t t_gates_after() const { return t_after_; }
    [[nodiscard]] const McrReorderStats& mcr_stats() const { return mcr_stats_; }
    [[nodiscard]] size_t todd_blocks() const { return todd_blocks_; }
    [[nodiscard]] size_t todd_t_removed() const { return todd_t_removed_; }

  private:
    size_t t_before_ = 0;
    size_t t_after_ = 0;
    McrReorderStats mcr_stats_{};
    size_t todd_blocks_ = 0;
    size_t todd_t_removed_ = 0;
};

}  // namespace clifft
