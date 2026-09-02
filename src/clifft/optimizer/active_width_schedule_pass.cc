#include "clifft/optimizer/active_width_schedule_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// ---------------------------------------------------------------------------
// Closure and readiness bookkeeping for the beam search below.
//
// Invariants (see docs/theory/active-width.md for the closure theorem's
// full argument and its confluence corollary):
//   ready       every predecessor of the op, per ScheduleDependence, has
//               already executed.
//   expanding   executing the op would raise the active width: a T_GATE or
//               PHASE_ROTATION whose axis does not commute with every
//               generator of the current DormantSubspace, or an
//               INSTRUMENT that takes the Activate branch. Every other
//               ready op is non-expanding.
//   closure     repeatedly executing the lowest-index ready non-expanding
//               op until none remains never raises the peak a schedule
//               could otherwise reach, so a scheduler only has to choose
//               which ready expanding op fires next; closure fills in the
//               rest deterministically.
//   confluence  the subspace reached by executing a given set of ops does
//               not depend on the order they executed in, so a scheduling
//               state is fully determined by its executed-op set -- the
//               identity the beam search's dedup step below relies on.
// ---------------------------------------------------------------------------

// True when executing `op` against `subspace` would raise the active width:
// a T_GATE/PHASE_ROTATION whose axis does not commute with every generator
// of S, or an INSTRUMENT that takes the Activate branch (see
// active_width_analysis.h's WidthEffect). Pure query, no mutation, so a
// caller can test every ready op before committing to one.
bool is_expanding(const HirModule& hir, const HeisenbergOp& op, const DormantSubspace& subspace) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            return !subspace.commutes_with_all(hir.destab_mask(op), hir.stab_mask(op));
        case OpType::INSTRUMENT: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (subspace.commutes_with_all(x, z)) {
                return false;  // Classical or Active: non-expanding.
            }
            const InstrumentSite& site =
                hir.instrument_sites.at(static_cast<uint32_t>(op.instrument_site_idx()));
            const bool traps = hir.neglect_instrument_damping ||
                               site.probabilities.p_fire[0] == site.probabilities.p_fire[1];
            return !traps;  // Activate iff it does not trap.
        }
        default:
            return false;
    }
}

// One execute()/undo() bracket: the op that executed, and how many
// successors newly became ready as a result. The count indexes into the
// caller's own shared newly-ready log (see SearchFrontier::execute), so
// backtracking never pays for a fresh heap allocation per step the way a
// per-step std::vector<uint32_t> would.
struct UndoStep {
    uint32_t op = 0;
    uint32_t newly_ready_count = 0;
};

// Tracks which ops are executed and which are ready (every predecessor
// executed) as one mutable structure a caller updates via execute()/undo()
// pairs bracketing each recursive or speculative step. Readiness and the
// executed set are both pure functions of the executed-op set alone (see
// the confluence invariant above), so undo() only has to reverse exactly
// what its matching execute() call did, never recompute anything from
// scratch. Copyable: a caller exploring several independent speculative
// continuations (rather than backtracking one shared instance) can clone a
// frontier instead of using execute()/undo().
//
// Readiness lives in a flat ready_flag_ array rather than a std::set: it
// churns on every execute()/undo() call in the beam search's inner loop,
// and a flag flip is both allocation-free and branch-cheap where a tree
// insert or erase is neither. lowest_ready_hint_ is a safe lower bound on
// the smallest ready op index (never above the true minimum, though it can
// undershoot after a removal until the next scan walks past the gap),
// letting find_ready_non_expanding's ascending scan skip the already-known-
// empty prefix below it while still visiting ready ops in the same
// lowest-index-first order a std::set would and stopping at the first
// non-expanding one -- which is what makes each avoided is_expanding() call
// (a GF(2) commutation scan against every generator of S) worth avoiding.
class SearchFrontier {
  public:
    explicit SearchFrontier(const detail::ScheduleDependence& dependence);

    [[nodiscard]] bool is_ready(uint32_t op) const { return ready_flag_[op] != 0; }
    [[nodiscard]] uint32_t num_ops() const { return static_cast<uint32_t>(ready_flag_.size()); }
    [[nodiscard]] uint32_t lowest_ready_hint() const { return lowest_ready_hint_; }
    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] size_t executed_count() const { return executed_count_; }

    // Marks `op` executed. Appends each successor that newly became ready to
    // the end of `newly_ready_log` (grown, never cleared here, so a caller
    // threading one log across a whole closure sweep or search amortizes
    // its allocations instead of paying for a fresh vector per call) and
    // returns how many entries it appended -- store this (typically in an
    // UndoStep) to undo exactly this call later.
    uint32_t execute(uint32_t op, std::vector<uint32_t>& newly_ready_log);

    // Reverses exactly an execute() call that appended `newly_ready_count`
    // entries to the end of `newly_ready_log`, popping them back off.
    void undo(uint32_t op, uint32_t newly_ready_count, std::vector<uint32_t>& newly_ready_log);

  private:
    void mark_ready(uint32_t op);
    void mark_not_ready(uint32_t op);

    const detail::ScheduleDependence* dependence_;
    std::vector<uint64_t> executed_;
    std::vector<uint32_t> remaining_preds_;
    std::vector<uint8_t> ready_flag_;
    uint32_t lowest_ready_hint_ = 0;
    size_t executed_count_ = 0;
};

void bitset_set(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] |= (uint64_t{1} << (index % 64));
}

void bitset_clear(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] &= ~(uint64_t{1} << (index % 64));
}

SearchFrontier::SearchFrontier(const detail::ScheduleDependence& dependence)
    : dependence_(&dependence),
      executed_((dependence.num_ops() + 63) / 64, 0),
      remaining_preds_(dependence.num_ops()),
      ready_flag_(dependence.num_ops(), 0) {
    for (uint32_t op = 0; op < dependence.num_ops(); ++op) {
        remaining_preds_[op] = static_cast<uint32_t>(dependence.predecessors(op).size());
        if (remaining_preds_[op] == 0) {
            mark_ready(op);
        }
    }
}

void SearchFrontier::mark_ready(uint32_t op) {
    ready_flag_[op] = 1;
    lowest_ready_hint_ = std::min(lowest_ready_hint_, op);
}

void SearchFrontier::mark_not_ready(uint32_t op) {
    ready_flag_[op] = 0;
    // Only a cheap, exact-match bump: the op just removed was the hint's own
    // witness, so the hint must move at least past it, but the true new
    // minimum among whatever remains ready is not known without a scan --
    // find_ready_non_expanding's own scan discovers it lazily instead.
    if (op == lowest_ready_hint_) {
        ++lowest_ready_hint_;
    }
}

uint32_t SearchFrontier::execute(uint32_t op, std::vector<uint32_t>& newly_ready_log) {
    assert(is_ready(op) && "execute() called on a non-ready op");
    mark_not_ready(op);
    bitset_set(executed_, op);
    ++executed_count_;

    uint32_t count = 0;
    for (uint32_t succ : dependence_->successors(op)) {
        if (--remaining_preds_[succ] == 0) {
            mark_ready(succ);
            newly_ready_log.push_back(succ);
            ++count;
        }
    }
    return count;
}

void SearchFrontier::undo(uint32_t op, uint32_t newly_ready_count,
                          std::vector<uint32_t>& newly_ready_log) {
    for (uint32_t succ : dependence_->successors(op)) {
        ++remaining_preds_[succ];
    }

    assert(newly_ready_log.size() >= newly_ready_count &&
           "undo() asked to reverse more newly-ready entries than the log holds");
    for (uint32_t i = 0; i < newly_ready_count; ++i) {
        mark_not_ready(newly_ready_log.back());
        newly_ready_log.pop_back();
    }

    bitset_clear(executed_, op);
    --executed_count_;
    // op is ready again, and may be lower than the current hint (it was
    // ready before this undo's matching execute() call raised the hint past
    // it), so this goes through mark_ready rather than a raw flag set.
    mark_ready(op);
}

// Undoes every step in `log`, most recent first, against the shared
// `newly_ready_log` every step's SearchFrontier::execute() call appended to.
void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log,
              std::vector<uint32_t>& newly_ready_log) {
    for (auto it = log.rbegin(); it != log.rend(); ++it) {
        frontier.undo(it->op, it->newly_ready_count, newly_ready_log);
    }
}

// Lowest-index ready op that is not expanding, or nullopt when every
// currently ready op (if any) is expanding.
std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                 const SearchFrontier& frontier,
                                                 const DormantSubspace& subspace) {
    // Ascending scan starting from the hint (a safe lower bound, never above
    // the true minimum ready index): a cheap is_ready() check skips every
    // not-ready index, so is_expanding() only ever runs on an actual ready
    // op, in the same lowest-index-first order a sorted container would
    // visit them, stopping at the first non-expanding one.
    for (uint32_t op = frontier.lowest_ready_hint(); op < frontier.num_ops(); ++op) {
        if (!frontier.is_ready(op)) {
            continue;
        }
        if (!is_expanding(hir, hir.ops[op], subspace)) {
            return op;
        }
    }
    return std::nullopt;
}

// Executes every ready non-expanding op, lowest index first, until none is
// ready: the closure step the closure invariant above justifies. Appends
// each executed op, in execution order, to `order` and logs it in `log`
// (backed by `newly_ready_log`, which every logged step's execute() call
// appends to) for the caller to undo later if needed. When `transitions` is
// non-null, each executed op's classification (before/after width and
// effect) is appended to it in the same order, so a caller that needs
// per-op dense-work contributions (the scheduling pass) can accumulate them
// without a second pass over the same ops; a caller that only needs width
// leaves this null.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<uint32_t>& newly_ready_log,
                 std::vector<WidthTransition>* transitions = nullptr) {
    while (const std::optional<uint32_t> op = find_ready_non_expanding(hir, frontier, subspace)) {
        const uint32_t newly_ready_count = frontier.execute(*op, newly_ready_log);
        log.push_back(UndoStep{*op, newly_ready_count});
        order.push_back(*op);
        const WidthTransition transition = classify_and_apply(hir, hir.ops[*op], subspace);
        assert(!is_expanding_effect(transition.effect) &&
               "find_ready_non_expanding chose an op classify_and_apply treats as expanding");
        if (transitions != nullptr) {
            transitions->push_back(transition);
        }
    }
}

// ---------------------------------------------------------------------------
// Beam state: one partial schedule, closed (every ready non-expanding op
// already executed), tracking everything needed to score and extend it
// further without replaying the ops executed so far.
// ---------------------------------------------------------------------------

struct BeamState {
    BeamState(SearchFrontier frontier_in, DormantSubspace subspace_in)
        : frontier(std::move(frontier_in)), subspace(std::move(subspace_in)) {}

    SearchFrontier frontier;
    DormantSubspace subspace;
    uint32_t peak = 0;
    double dense_work = 0.0;
    std::vector<uint32_t> order;
};

void absorb_transitions(uint32_t& peak, double& dense_work,
                        const std::vector<WidthTransition>& transitions) {
    for (const WidthTransition& transition : transitions) {
        peak = std::max(peak, transition.after);
        dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
}

void absorb_closure_transitions(BeamState& state, const std::vector<WidthTransition>& transitions) {
    absorb_transitions(state.peak, state.dense_work, transitions);
}

BeamState make_initial_beam_state(const HirModule& hir,
                                  const detail::ScheduleDependence& dependence) {
    BeamState state(SearchFrontier(dependence), DormantSubspace(hir.num_qubits));
    std::vector<UndoStep> discarded_log;
    std::vector<uint32_t> discarded_newly_ready;
    std::vector<WidthTransition> transitions;
    run_closure(hir, state.frontier, state.subspace, state.order, discarded_log,
                discarded_newly_ready, &transitions);
    absorb_closure_transitions(state, transitions);
    return state;
}

// Copies out the parent's current ready set before any candidate scoring
// below mutates it: score_candidates' loop executes and undoes each
// candidate against `parent.frontier` in turn, so iterating a live view of
// its ready ops while that same loop body edits them is unsafe. The scan is
// ascending (lowest_ready_hint() up to num_ops()), so the candidate list
// score_candidates below produces is ordered the same way regardless of the
// history of execute()/undo() calls that got `frontier` to its current
// state -- which matters because the beam ranking sort's key does not fully
// order every pair of candidates, and an input built in a different order
// could lead an otherwise-tied sort to a different, equally-ranked winner.
// This runs once per beam member per beam step, not once per closure step,
// so a scan over a few thousand ops here is negligible next to the cost of
// scoring and closing each candidate.
std::vector<uint32_t> ready_ops_snapshot(const SearchFrontier& frontier) {
    std::vector<uint32_t> ready;
    for (uint32_t op = frontier.lowest_ready_hint(); op < frontier.num_ops(); ++op) {
        if (frontier.is_ready(op)) {
            ready.push_back(op);
        }
    }
    return ready;
}

// A scored but not yet materialized child of beam[parent_index]: committing
// to `first_op` (a ready expanding op) and closing would append `ops`
// (first_op followed by its closure sweep, in execution order) to the
// parent's own order and reach `width_after_closure`/peak/dense_work.
// executed_bits is the parent's post-candidate executed-op bitset, captured
// for deduplication without needing a per-candidate SearchFrontier clone.
// Scored by (peak, width_after_closure, -ops.size(), first_op) ascending:
// among candidates tied on peak and width_after_closure, the one whose
// closure swept the most operations into place (typically measurements)
// ranks first, ahead of dense_work_so_far. The intuition: a wide sweep
// usually means several commuting measurements collapsed the subspace
// immediately, which is a stronger signal that this branch is heading
// toward a state with few remaining live directions than the dense work
// spent to get there so far -- measured on the clifft-paper corpus, this
// tie-break reaches lower summed dense work than tying on dense_work_so_far
// (with or without also considering sweep count).
struct ScoredCandidate {
    uint32_t parent_index = 0;
    std::vector<uint32_t> ops;
    std::vector<uint64_t> executed_bits;
    uint32_t width_after_closure = 0;
    uint32_t peak = 0;
    double dense_work = 0.0;
    uint32_t first_op = 0;
};

// Scores every ready expanding op of `parent` without materializing a full
// child BeamState for each: `parent.frontier` is mutated and restored via
// paired execute()/undo() calls, and only `subspace` (which has no cheap
// undo) is cloned, once per candidate. This is the expensive
// step's cost reduction: on a fixture with many simultaneously-ready
// independent expanding ops, most candidates are discarded after scoring,
// so paying only for a DormantSubspace clone and a small ops list here --
// not a SearchFrontier clone and a full copy of the (potentially
// near-complete) order vector -- is what keeps this affordable. `parent` is
// left exactly as found on return.
std::vector<ScoredCandidate> score_candidates(const HirModule& hir, BeamState& parent,
                                              uint32_t parent_index) {
    std::vector<ScoredCandidate> scored;
    for (uint32_t op : ready_ops_snapshot(parent.frontier)) {
        if (!is_expanding(hir, hir.ops[op], parent.subspace)) {
            continue;
        }

        DormantSubspace scratch(parent.subspace);
        std::vector<UndoStep> log;
        std::vector<uint32_t> newly_ready_log;

        log.push_back(UndoStep{op, parent.frontier.execute(op, newly_ready_log)});
        ScoredCandidate candidate;
        candidate.parent_index = parent_index;
        candidate.first_op = op;
        candidate.ops.push_back(op);

        const WidthTransition first_transition = classify_and_apply(hir, hir.ops[op], scratch);
        assert(is_expanding_effect(first_transition.effect) &&
               "score_candidates chose a ready op is_expanding did not classify as expanding");
        candidate.peak = std::max(parent.peak, first_transition.after);
        candidate.dense_work =
            parent.dense_work + detail::dense_work_contribution(first_transition.effect,
                                                                first_transition.before,
                                                                first_transition.after);

        std::vector<WidthTransition> swept;
        run_closure(hir, parent.frontier, scratch, candidate.ops, log, newly_ready_log, &swept);
        absorb_transitions(candidate.peak, candidate.dense_work, swept);

        candidate.width_after_closure = scratch.active_width();
        candidate.executed_bits = parent.frontier.executed_bits();

        undo_all(parent.frontier, log, newly_ready_log);
        scored.push_back(std::move(candidate));
    }
    return scored;
}

// Materializes a beam_width survivor: clones its parent once (the only
// clone this candidate ever needed) and replays `candidate.ops` -- already
// known from scoring, so this is a deterministic replay rather than a new
// search -- through the clone's own frontier and subspace. The assertions
// cross-check that this replay reaches exactly what score_candidates
// predicted, since the two are computed by independent code paths that
// must agree bit-for-bit (dense_work accumulates the same transitions in
// the same order in both places, so the floating-point sums match exactly,
// not just numerically).
BeamState materialize_candidate(const HirModule& hir, const std::vector<BeamState>& beam,
                                const ScoredCandidate& candidate) {
    const BeamState& parent = beam[candidate.parent_index];
    BeamState state(parent.frontier, parent.subspace);
    state.peak = parent.peak;
    state.dense_work = parent.dense_work;
    state.order = parent.order;
    state.order.reserve(state.order.size() + candidate.ops.size());

    // This replay never backtracks, so the newly-ready ops execute() reports
    // are never read; one scratch vector reused for the whole loop avoids
    // both a fresh allocation per op and the earlier per-call
    // std::vector<uint32_t> return this replaced.
    std::vector<uint32_t> discarded_newly_ready;
    for (uint32_t op : candidate.ops) {
        state.frontier.execute(op, discarded_newly_ready);
        const WidthTransition transition = classify_and_apply(hir, hir.ops[op], state.subspace);
        state.peak = std::max(state.peak, transition.after);
        state.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
        state.order.push_back(op);
    }

    assert(state.subspace.active_width() == candidate.width_after_closure &&
           "materialize_candidate's replay disagrees with score_candidates' speculative width");
    assert(state.peak == candidate.peak &&
           "materialize_candidate's replay disagrees with score_candidates' speculative peak");
    assert(
        state.dense_work == candidate.dense_work &&
        "materialize_candidate's replay disagrees with score_candidates' speculative dense work");

    return state;
}

// Among completed (fully executed) beam states, the one with the smallest
// (peak, dense_work), tied by comparing the executed order itself so the
// choice is fully deterministic even when two structurally different
// schedules happen to cost exactly the same.
const BeamState* pick_best_completed(const std::vector<BeamState>& completed) {
    const BeamState* best = nullptr;
    for (const BeamState& state : completed) {
        if (best == nullptr) {
            best = &state;
            continue;
        }
        if (state.peak != best->peak) {
            if (state.peak < best->peak) {
                best = &state;
            }
            continue;
        }
        if (state.dense_work != best->dense_work) {
            if (state.dense_work < best->dense_work) {
                best = &state;
            }
            continue;
        }
        if (state.order < best->order) {
            best = &state;
        }
    }
    return best;
}

// Beam search over the closure/readiness machinery above. beam_width == 1
// degenerates to the greedy closure scheduler: at every step, take
// whichever single ready expanding op's own closure sweep scores best. The
// constructor rejects beam_width == 0, so at least one beam member always
// survives to complete a schedule.
//
// Two-phase per step: score_candidates ranks every ready expanding op of
// every current beam state cheaply (see its own comment), then only the
// surviving beam_width candidates -- picked by the same dedup-then-rank
// rule the single-phase version used -- pay to materialize a full BeamState
// via materialize_candidate. A wide, mostly-discarded generation is common
// on fixtures with many independent expanding rotations, so this ordering
// (score everything, materialize only the winners) is what makes the beam
// width affordable to scale.
std::vector<uint32_t> run_beam_search(const HirModule& hir,
                                      const detail::ScheduleDependence& dependence,
                                      uint32_t beam_width) {
    std::vector<BeamState> beam;
    beam.push_back(make_initial_beam_state(hir, dependence));

    std::vector<BeamState> completed;
    while (!beam.empty()) {
        std::vector<ScoredCandidate> generation;
        std::vector<bool> parent_has_candidates(beam.size(), false);
        for (uint32_t i = 0; i < beam.size(); ++i) {
            std::vector<ScoredCandidate> scored = score_candidates(hir, beam[i], i);
            if (!scored.empty()) {
                parent_has_candidates[i] = true;
                for (ScoredCandidate& candidate : scored) {
                    generation.push_back(std::move(candidate));
                }
            }
        }

        for (uint32_t i = 0; i < beam.size(); ++i) {
            if (parent_has_candidates[i]) {
                continue;
            }
            // Closure already consumed every ready non-expanding op, so no
            // ready expanding candidates left means no ready ops left at
            // all, which is only possible once every op has executed (see
            // the confluence invariant above).
            assert(beam[i].frontier.executed_count() == dependence.num_ops() &&
                   "a closed beam state with no ready expanding op must have executed every op");
            completed.push_back(std::move(beam[i]));
        }

        if (generation.empty()) {
            break;
        }

        // Deduplicate by executed-op bitset: by confluence, the same set of
        // executed ops always reaches the same subspace and width, so only
        // one survivor is worth keeping. The peak reached along the way is
        // not confluent, though: it is a running max over a path-dependent
        // sequence of transitions, so two candidates that converge on the
        // same executed set can still disagree on peak. Keeping whichever
        // has the smaller dense_work, regardless of peak, can therefore
        // discard the strictly better duplicate -- lower peak but higher
        // dense_work -- since peak can only stay the same or grow over the
        // rest of the schedule while dense_work is only a secondary
        // objective. Rank by the pass's own lexicographic objective,
        // (peak, dense_work), instead, with first_op breaking a remaining
        // tie deterministically.
        std::ranges::sort(generation, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.executed_bits < b.executed_bits;
        });
        std::vector<ScoredCandidate> deduped;
        for (ScoredCandidate& candidate : generation) {
            if (!deduped.empty() && deduped.back().executed_bits == candidate.executed_bits) {
                const ScoredCandidate& kept = deduped.back();
                const bool candidate_is_better =
                    std::tie(candidate.peak, candidate.dense_work, candidate.first_op) <
                    std::tie(kept.peak, kept.dense_work, kept.first_op);
                if (candidate_is_better) {
                    deduped.back() = std::move(candidate);
                }
                continue;
            }
            deduped.push_back(std::move(candidate));
        }

        std::ranges::sort(deduped, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            if (a.peak != b.peak) {
                return a.peak < b.peak;
            }
            if (a.width_after_closure != b.width_after_closure) {
                return a.width_after_closure < b.width_after_closure;
            }
            if (a.ops.size() != b.ops.size()) {
                return a.ops.size() > b.ops.size();  // -swept_count ascending.
            }
            return a.first_op < b.first_op;
        });

        std::vector<BeamState> next_beam;
        next_beam.reserve(std::min<size_t>(deduped.size(), beam_width));
        for (size_t i = 0; i < deduped.size() && i < beam_width; ++i) {
            next_beam.push_back(materialize_candidate(hir, beam, deduped[i]));
        }
        beam = std::move(next_beam);
    }

    const BeamState* best = pick_best_completed(completed);
    assert(best != nullptr &&
           "beam_width >= 1 (the constructor rejects 0) guarantees at least one completed state");
    return best->order;
}

// ---------------------------------------------------------------------------
// Neutral-rotation sinking: a rightward bubble per RotationNeutral op.
// ---------------------------------------------------------------------------

bool independent(const detail::ScheduleDependence& dependence, uint32_t a, uint32_t b) {
    return !std::ranges::binary_search(dependence.successors(a), b) &&
           !std::ranges::binary_search(dependence.predecessors(a), b);
}

// Each op's WidthEffect when `order` is replayed from a fresh subspace,
// indexed by op index (not position) so it stays valid as `order` is
// mutated afterward: a RotationNeutral op is a complete no-op on the
// subspace (see active_width_schedule_pass.h), so removing or relocating
// one never changes any other op's classification. Every RotationNeutral
// crossing performed below therefore reads a classification that remains
// accurate no matter how many other neutral rotations have already moved.
std::vector<WidthEffect> effect_by_op_index(const HirModule& hir,
                                            const std::vector<uint32_t>& order) {
    std::vector<WidthEffect> effect(hir.ops.size(), WidthEffect::None);
    DormantSubspace subspace(hir.num_qubits);
    for (uint32_t op : order) {
        effect[op] = classify_and_apply(hir, hir.ops[op], subspace).effect;
    }
    return effect;
}

// Bubbles every RotationNeutral op in `order` as far right as it can legally
// go, one op at a time in original schedule order, stopping each bubble
// before the first op that is dependent (per `dependence`) or expanding.
// Crossing only independent, non-expanding ops never raises the width the
// rotation runs at, and never reorders any pair of non-sunk ops relative to
// each other, so every intermediate and final order stays a legal linear
// extension of `dependence`. Does not move RotationStabilizer ops: they
// emit no planner action, so sinking one would not change dense work.
void sink_neutral_rotations(const HirModule& hir, const detail::ScheduleDependence& dependence,
                            std::vector<uint32_t>& order) {
    const std::vector<WidthEffect> effect = effect_by_op_index(hir, order);

    std::vector<uint32_t> to_sink;
    for (uint32_t op : order) {
        if (effect[op] == WidthEffect::RotationNeutral) {
            to_sink.push_back(op);
        }
    }
    if (to_sink.empty()) {
        return;
    }

    // position[op] tracks op's live index in `order`, kept in sync with
    // every swap below so each rotation's bubble starts from wherever it
    // actually sits, including any shift an earlier rotation's own bubble
    // caused.
    std::vector<uint32_t> position(hir.ops.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    for (uint32_t op : to_sink) {
        size_t curr = position[op];
        while (curr + 1 < order.size()) {
            const uint32_t next_op = order[curr + 1];
            if (!independent(dependence, op, next_op) || is_expanding_effect(effect[next_op])) {
                break;
            }
            std::swap(order[curr], order[curr + 1]);
            position[order[curr]] = static_cast<uint32_t>(curr);
            position[order[curr + 1]] = static_cast<uint32_t>(curr + 1);
            ++curr;
        }
    }
}

// True when `hir` has any op a beam search could ever branch on. T_GATE and
// PHASE_ROTATION are the only op types is_expanding ever calls genuinely
// discretionary: an expanding INSTRUMENT is always the sole ready op when it
// fires (detail::ScheduleDependence treats it as a positional barrier), so
// its presence or absence changes nothing a scheduler could choose
// differently.
bool has_rotation_op(const HirModule& hir) {
    return std::ranges::any_of(hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
    });
}

}  // namespace

ActiveWidthSchedulePass::ActiveWidthSchedulePass(ActiveWidthScheduleOptions options)
    : options_(options) {
    if (options_.beam_width == 0) {
        throw std::invalid_argument("ActiveWidthSchedulePass: beam_width must be positive");
    }
}

void ActiveWidthSchedulePass::run(HirModule& hir) {
    built_dependence_ = false;

    const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
    incumbent_peak_ = incumbent_trace.peak_width;
    incumbent_dense_work_ = estimate_dense_work(incumbent_trace);

    // See the header comment's "Early exit": with nothing for a scheduler to
    // choose among, report the incumbent unchanged rather than pay for
    // ScheduleDependence::build's O(N^2) scan.
    if (incumbent_peak_ == 0 || !has_rotation_op(hir)) {
        result_peak_ = incumbent_peak_;
        result_dense_work_ = incumbent_dense_work_;
        applied_ = false;
        return;
    }

    detail::ScheduleDependenceOptions dependence_options;
    dependence_options.noise_transparent = options_.noise_transparent;
    const detail::ScheduleDependence dependence =
        detail::ScheduleDependence::build(hir, dependence_options);
    built_dependence_ = true;

    std::vector<uint32_t> order = run_beam_search(hir, dependence, options_.beam_width);

    if (options_.sink_neutral_rotations) {
        sink_neutral_rotations(hir, dependence, order);
    }

    HirModule candidate = hir;
    detail::apply_schedule(candidate, dependence, order);
    const ActiveWidthTrace candidate_trace = analyze_active_width(candidate);
    const double candidate_dense_work = estimate_dense_work(candidate_trace);

    const bool better = (candidate_trace.peak_width < incumbent_peak_) ||
                        (candidate_trace.peak_width == incumbent_peak_ &&
                         candidate_dense_work < incumbent_dense_work_);

    if (better) {
        hir = std::move(candidate);
        result_peak_ = candidate_trace.peak_width;
        result_dense_work_ = candidate_dense_work;
        applied_ = true;
    } else {
        result_peak_ = incumbent_peak_;
        result_dense_work_ = incumbent_dense_work_;
        applied_ = false;
    }
}

}  // namespace clifft
