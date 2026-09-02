#pragma once

// State-aware HIR scheduling pass.
//
// Objective, lexicographic: first minimize peak active width, then minimize
// estimate_dense_work's planning proxy for the dense work a trace performs
// along the way. Closure theorem, restated in one sentence (see
// active_width_closure.h for the full argument and its confluence
// corollary): some peak-minimizing schedule executes every ready
// non-expanding op as soon as it is ready, so a scheduler only has to
// choose, at each step, which ready expanding rotation fires next. This
// pass keeps a beam of the best few partial choices at each such step,
// closing each one out with the shared closure sweep, and reports whichever
// completed beam member scores best.
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
//
// Early exit: with a 0 incumbent peak, or no T_GATE or PHASE_ROTATION op in
// the HIR, there is no ready expanding rotation for a scheduler to choose
// among -- an expanding INSTRUMENT is always the only ready op when it
// fires (detail::ScheduleDependence treats it as a positional barrier), so
// it offers no scheduling freedom either -- and run() reports the incumbent
// unchanged without paying for detail::ScheduleDependence::build's O(N^2)
// scan.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"

#include <cstdint>

namespace clifft {

struct ActiveWidthScheduleOptions {
    // Passed through to detail::ScheduleDependence::build. See
    // schedule_dependence.h for why crossing a NOISE op is sound under this
    // relaxation.
    bool noise_transparent = true;

    // Number of partial schedules kept at each beam-search step. 1
    // degenerates to the greedy closure scheduler (always take the single
    // best-looking ready expanding op); larger values explore more of the
    // choice tree at proportionally higher cost. 0 leaves nothing for the
    // beam to keep, so the constructor rejects it.
    uint32_t beam_width = 8;

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

    // True when the last run() call built the detail::ScheduleDependence
    // relation, i.e. did not take the early exit above. A structural
    // witness that the exit actually happened, for tests that would
    // otherwise have to infer it from a timing side-channel.
    [[nodiscard]] bool built_dependence() const { return built_dependence_; }

  private:
    ActiveWidthScheduleOptions options_;
    uint32_t incumbent_peak_ = 0;
    uint32_t result_peak_ = 0;
    double incumbent_dense_work_ = 0.0;
    double result_dense_work_ = 0.0;
    bool applied_ = false;
    bool built_dependence_ = false;
};

}  // namespace clifft
