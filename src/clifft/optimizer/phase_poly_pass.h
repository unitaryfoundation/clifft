#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/mcr_tcount.h"

#include <cstddef>

namespace clifft {

struct TohpePhaseStats {
    size_t t_reductions = 0;
    size_t blocks_optimized = 0;
};

/// Size-capped TOHPE duplicate-and-destroy on commuting T_GATE blocks.
void run_tohpe_phase(HirModule& hir, TohpePhaseStats& stats);

/// Opt-in HIR pass wrapper around run_tohpe_phase for per-phase evaluation.
class TohpePhasePass : public HirPass {
  public:
    void run(HirModule& hir) override;

    size_t t_reductions() const { return stats_.t_reductions; }
    size_t blocks_optimized() const { return stats_.blocks_optimized; }

  private:
    TohpePhaseStats stats_{};
};

/// Experimental global T-count pass: bounded MCR reordering on contiguous
/// T windows, then size-capped TOHPE on commuting T_GATE blocks. Opt-in only;
/// run between PeepholeFusionPass sweeps.
class PhasePolynomialPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    size_t t_reductions() const { return t_reductions_; }
    size_t blocks_optimized() const { return blocks_optimized_; }
    size_t t_gates_before() const { return t_gates_before_; }
    size_t t_gates_after() const { return t_gates_after_; }
    const McrTcountStats& mcr_stats() const { return mcr_stats_; }

  private:
    size_t t_reductions_ = 0;
    size_t blocks_optimized_ = 0;
    size_t t_gates_before_ = 0;
    size_t t_gates_after_ = 0;
    McrTcountStats mcr_stats_{};
};

}  // namespace clifft
