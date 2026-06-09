#include "clifft/optimizer/global_tcount_pass.h"

#include "clifft/optimizer/todd_phase_pass.h"

namespace clifft {

namespace {

void run_todd_phase(HirModule& hir, size_t& blocks, size_t& t_removed) {
    ToddPhasePass pass;
    pass.run(hir);
    blocks = pass.blocks_optimized();
    t_removed = pass.t_removed();
}

}  // namespace

void GlobalTcountPass::run(HirModule& hir) {
    t_before_ = hir.num_t_gates();
    mcr_stats_ = McrReorderStats{};
    todd_blocks_ = 0;
    todd_t_removed_ = 0;

    run_mcr_reorder(hir, mcr_stats_);
    run_todd_phase(hir, todd_blocks_, todd_t_removed_);

    t_after_ = hir.num_t_gates();
}

}  // namespace clifft
