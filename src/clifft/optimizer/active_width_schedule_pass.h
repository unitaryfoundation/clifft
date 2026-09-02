#pragma once

// State-aware HIR scheduling pass.
//
// Objective, lexicographic: first minimize peak active width, then minimize
// estimate_dense_work's planning proxy for the dense work a trace performs
// along the way. Closure theorem (active_width_search.h, restated in one
// sentence): some peak-minimizing schedule executes every ready
// non-expanding op as soon as it is ready, so a scheduler only has to
// choose, at each step, which ready expanding rotation or instrument fires
// next. active_width_search.cc answers that question exactly, with a node
// budget and a certificate; this pass instead keeps a beam of the best few
// partial choices, trading the certificate for a result that scales past
// circuits the exact search's budget cannot finish, then optionally spends
// a bounded amount of exact search polishing the beam's own answer.
//
// This pass must run last in a pipeline, after PeepholeFusionPass and
// StatevectorSqueezePass. PeepholeFusionPass refuses to fuse an operation
// across a NOISE op once HirModule::logical_noise_prefix disagrees with
// that operation's schedule position (see schedule_dependence.h) -- which
// is exactly the state a noise-transparent reorder leaves behind. Running
// this pass earlier would therefore either do nothing useful (peephole
// already ran, nothing left to fuse) or leave the HIR in a shape peephole
// can no longer safely optimize.
//
// "Never worse than the incumbent" means: compute analyze_active_width on
// the input HIR before touching it (the incumbent) and again on the
// pass's own candidate order, compare the two lexicographically by (peak,
// estimate_dense_work), and apply the candidate only if it is strictly
// better by that order. Anything else -- including a candidate that only
// ties the incumbent -- leaves the HIR completely untouched: not even
// reordered to a same-cost permutation.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"

#include <cstdint>

namespace clifft {

struct ActiveWidthScheduleOptions {
    // Passed through to ScheduleDependence::build. See schedule_dependence.h
    // for why crossing a NOISE op is sound under this relaxation.
    bool noise_transparent = true;

    // Number of partial schedules kept at each beam-search step. 1
    // degenerates to the greedy closure scheduler (always take the single
    // best-looking ready expanding op); larger values explore more of the
    // choice tree at proportionally higher cost.
    uint32_t beam_width = 8;

    // Node budget for the optional exact-repair step that runs
    // search_width_schedule on the beam's own result. 0 disables repair,
    // and is the default: measured against the beam search alone on the
    // clifft-paper corpus in a Release build, exact repair never lowered a
    // peak the beam had not already reached and cost seconds of Release
    // wall time on the larger fixtures (ScheduleDependence::build is a
    // documented O(N^2) scan, and repair pays for a second one against the
    // beam's own result), so it is opt-in rather than on by default. The
    // knob stays because a circuit the beam handles worse than this corpus
    // could still benefit from it.
    uint64_t exact_node_budget = 0;

    // Whether to bubble width-neutral rotations rightward past independent
    // non-expanding ops after scheduling, clustering them just before the
    // next expansion for the executable-plan rotation fusion.
    bool sink_neutral_rotations = true;
};

class ActiveWidthSchedulePass : public HirPass {
  public:
    explicit ActiveWidthSchedulePass(ActiveWidthScheduleOptions options = {});

    void run(HirModule& hir) override;

    // Statistics from the last run() call. All read as zero/false before
    // the first call.
    [[nodiscard]] uint32_t incumbent_peak() const { return incumbent_peak_; }
    [[nodiscard]] uint32_t result_peak() const { return result_peak_; }
    [[nodiscard]] double incumbent_dense_work() const { return incumbent_dense_work_; }
    [[nodiscard]] double result_dense_work() const { return result_dense_work_; }
    [[nodiscard]] bool applied() const { return applied_; }

  private:
    ActiveWidthScheduleOptions options_;
    uint32_t incumbent_peak_ = 0;
    uint32_t result_peak_ = 0;
    double incumbent_dense_work_ = 0.0;
    double result_dense_work_ = 0.0;
    bool applied_ = false;
};

}  // namespace clifft
