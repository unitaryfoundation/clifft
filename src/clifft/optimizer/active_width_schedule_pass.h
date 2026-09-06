#pragma once

// State-aware HIR scheduling pass.
//
// Objective, lexicographic: first minimize peak active width, then minimize
// estimate_dense_work's planning proxy for the dense work a trace performs
// along the way. Closure theorem, restated in one sentence (see
// docs/theory/active-width.md for the full argument and its confluence
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
// unchanged without paying to build a detail::ScheduleDependence at all.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"
#include "clifft/util/numeric.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace clifft {

namespace detail {

// True for a finite value that is not negative (0.0 and -0.0 both count as
// non-negative). Built on is_finite_robust and opaque_binary64_bits rather
// than std::isnan, std::isinf, or a direct comparison against 0.0, because
// this project builds with -ffast-math (finite-math-only): under that model
// the compiler is entitled to assume no value is ever NaN or infinity, so a
// comparison or standard-library classification involving a non-finite
// value is not reliable at that optimization level -- the same failure
// mode that ruled out infinity itself as ActiveWidthScheduleOptions's "no
// budget" sentinel below, in favor of an empty std::optional. See
// numeric.h's own comments for why even a direct bit-cast exponent check
// can still be folded away under that model, which is why this goes
// through its opaque helper rather than inspecting the bits directly.
inline bool is_finite_non_negative(double value) {
    if (!is_finite_robust(value)) {
        return false;
    }
    // A negative number has the sign bit set and a nonzero magnitude in
    // the remaining (exponent and mantissa) bits; -0.0 also has the sign
    // bit set, but its magnitude is all zero, and counts as non-negative.
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    const uint64_t bits = opaque_binary64_bits(value);
    return (bits & kSignMask) == 0 || (bits & ~kSignMask) == 0;
}

}  // namespace detail

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

    // Work the beam search may spend, in units of ops executed through
    // closure sweeps per op in the HIR (so 16 means about sixteen full
    // replays of the circuit). Counted in swept ops rather than wall-clock
    // time so a compiled plan does not depend on the machine that compiled
    // it: the beam-search cost is circuit-shape dependent, and a
    // wall-clock cutoff would make the schedule (and therefore the plan)
    // vary run to run.
    //
    // This one budget backs two graduated responses, both measured against
    // the same running swept-op count and both implemented in
    // run_beam_search (see its own comment for the exact mechanics). Once
    // the count exceeds half of *search_budget * hir.ops.size(), the beam
    // narrows to a single surviving parent for every remaining step. Once
    // it exceeds the full *search_budget * hir.ops.size(), that surviving
    // parent also stops comparing its own ready expanding rotations and
    // simply takes the lowest-index one at every further step. Narrowing
    // the beam first, while still letting the single survivor's candidates
    // compete fairly for a while longer, keeps the search's quality close
    // to an unbounded one on circuits where a handful of ready rotations
    // recur at every step, while the second, blunter response bounds the
    // cost even on circuits where that count itself grows with the circuit
    // size. The result is still a legal schedule, and the never-worse
    // guard in run() still applies, so narrowing can only give up some of
    // the beam search's improvement over the incumbent, never regress past
    // it. No bound is expressed by leaving this empty (std::nullopt),
    // never by an infinite double: this project builds with -ffast-math,
    // under which a non-finite sentinel like infinity is not reliably
    // comparable (see detail::is_finite_non_negative above), so an empty
    // optional is the only value that reliably means "unbounded" here. An
    // empty budget leaves this pass's behavior unchanged: the search
    // always runs to completion at the full beam_width.
    //
    // The default, 16, comes from measuring the unbounded search against
    // narrower budgets over a varied circuit corpus: every budget of 8 or
    // more (a beam-narrowing threshold of 4 traces or more) reached the
    // unbounded search's own peak on every circuit measured, and 16 (a
    // beam-narrowing threshold of 8 traces) was the smallest that also
    // kept every dense-work gain the unbounded search found. The unbounded
    // search's own cost varied enormously by circuit shape, while a budget
    // of 16 kept every measured circuit to a small, comparable multiple of
    // that cost. Revisit this default if a production circuit's shape
    // falls outside what was measured.
    //
    // The constructor rejects a negative or non-finite value.
    std::optional<double> search_budget = 16.0;

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

    // Ops executed through closure sweeps and candidate replays during the
    // last run() call: the initial closure, every scoring sweep in
    // score_candidates (including candidates later discarded), and every
    // materialize_candidate replay. This is the quantity search_budget
    // bounds. Reset to zero at the start of each run(), and left at zero
    // when the early exit fires.
    [[nodiscard]] size_t swept_ops() const { return swept_ops_; }

  private:
    ActiveWidthScheduleOptions options_;
    uint32_t incumbent_peak_ = 0;
    uint32_t result_peak_ = 0;
    double incumbent_dense_work_ = 0.0;
    double result_dense_work_ = 0.0;
    bool applied_ = false;
    bool built_dependence_ = false;
    size_t swept_ops_ = 0;
};

}  // namespace clifft
