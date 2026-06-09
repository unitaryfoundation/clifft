#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

struct McrReorderStats {
    size_t window_scans = 0;
    size_t quadruples_found = 0;
    size_t swaps_applied = 0;
    size_t merges = 0;
    size_t t_removed = 0;
};

void run_mcr_reorder(HirModule& hir, McrReorderStats& stats);

class McrReorderPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    [[nodiscard]] const McrReorderStats& stats() const { return stats_; }

  private:
    McrReorderStats stats_{};
};

}  // namespace clifft
