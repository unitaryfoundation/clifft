#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

class ToddPhasePass : public HirPass {
  public:
    void run(HirModule& hir) override;

    [[nodiscard]] size_t blocks_optimized() const { return blocks_; }
    [[nodiscard]] size_t t_removed() const { return t_removed_; }

  private:
    size_t blocks_ = 0;
    size_t t_removed_ = 0;
};

}  // namespace clifft
