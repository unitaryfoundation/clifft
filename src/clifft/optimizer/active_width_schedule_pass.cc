#include "clifft/optimizer/active_width_schedule_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
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
//
// A closure sweep also memoizes which ready ops it has already found
// expanding (see SearchFrontier below), so a later step of the same sweep
// does not re-run is_expanding on an op whose verdict cannot have changed.
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
// Readiness lives in a flat ready_bits_ bitset rather than a std::set: it
// churns on every execute()/undo() call in the beam search's inner loop,
// and a bit flip is both allocation-free and branch-cheap where a tree
// insert or erase is neither. lowest_ready_hint_ is a mutable cache of the
// smallest ready op index, not just a lower bound: lowest_ready() tightens
// it on demand by scanning upward from the cached value's word to the first
// set bit at or above it (or num_ops() if none), then stores that exact
// result back before returning it. Recomputing the cache is legal from a
// const method because it is a pure function of ready_bits_ alone -- the
// same value a caller would get by scanning from zero every time -- so
// caching it only changes how much of the bitset a later scan has to
// revisit, never which op that scan returns. Without the tightening,
// mark_not_ready's one-step bump is the only thing that ever advances the
// hint, so after a long run of executed ops above it, lowest_ready's and
// ready_ops_snapshot's ascending scans would re-walk the same stale
// not-ready prefix on every call.
//
// The frontier also memoizes, for the closure sweep in progress, which
// ready ops is_expanding has already found expanding, as a second bitset
// (known_expanding_bits_) rather than a generation-stamped array: a rotation
// that anticommutes with some element of S keeps anticommuting with it while
// S only grows or stays put, since the new span contains the old one.
// Inside a sweep every executed op is non-expanding, and of the effects such
// an op can have only MeasureDormantRandom shrinks S; RotationPromote and
// InstrumentActivate shrink it too, but they are expanding, so they run only
// just before a sweep starts. run_closure therefore calls
// reset_expanding_memo() on entry and after every MeasureDormantRandom step,
// clearing known_expanding_bits_ with std::ranges::fill -- O(n/64) words,
// far cheaper than the O(n) walk it replaces -- and resets happen once per
// sweep entry and once per MeasureDormantRandom step, not once per closure
// step, so the clear is cheap relative to what it replaces.
//
// candidate_hint_ is a lower bound on the smallest index that is ready and
// not known-expanding, maintained the same way lowest_ready_hint_ is:
// mark_ready lowers both hints, mark_not_ready and note_expanding each bump
// their own hint by one on an exact-index match, and reset_expanding_memo
// resets candidate_hint_ to lowest_ready_hint_ (still a valid lower bound,
// since every candidate is also ready). first_candidate() tightens it by
// scanning ready_bits_[w] & ~known_expanding_bits_[w] a word at a time from
// the hint's word, so a caller walking successive candidates within one
// sweep pays for the words it skips, not the ops: a sweep over n ops with k
// memo resets costs O(n/64 * (1 + k)) word operations plus the is_expanding
// calls it genuinely needs, instead of O(n * distance) op-at-a-time steps.
class SearchFrontier {
  public:
    explicit SearchFrontier(const detail::ScheduleDependence& dependence);

    [[nodiscard]] bool is_ready(uint32_t op) const {
        return (ready_bits_[op / 64] & (uint64_t{1} << (op % 64))) != 0;
    }
    [[nodiscard]] uint32_t num_ops() const {
        return static_cast<uint32_t>(remaining_preds_.size());
    }

    // Tightens lowest_ready_hint_ to the exact smallest ready op index (or
    // num_ops() if none is ready) and returns it. See the class comment
    // above for why a const method may cache this and why the cache never
    // changes which op a scan returns.
    [[nodiscard]] uint32_t lowest_ready() const;

    // Expanding memo for the closure sweep in progress; see the class
    // comment for why a hit is exact and when it must be reset.
    void reset_expanding_memo();
    [[nodiscard]] bool known_expanding(uint32_t op) const {
        return (known_expanding_bits_[op / 64] & (uint64_t{1} << (op % 64))) != 0;
    }
    void note_expanding(uint32_t op);

    // Lowest-index op that is ready and not known-expanding, or nullopt if
    // none remains, tightening candidate_hint_ to the exact result (or
    // num_ops() when nullopt) the same way lowest_ready() tightens its own
    // hint. See the class comment above for the word-at-a-time scan this
    // replaces a one-op-at-a-time walk with.
    [[nodiscard]] std::optional<uint32_t> first_candidate();

    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] const std::vector<uint64_t>& ready_bits() const { return ready_bits_; }
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
    std::vector<uint64_t> ready_bits_;
    mutable uint32_t lowest_ready_hint_ = 0;
    size_t executed_count_ = 0;
    std::vector<uint64_t> known_expanding_bits_;
    uint32_t candidate_hint_ = 0;
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
      ready_bits_((dependence.num_ops() + 63) / 64, 0),
      known_expanding_bits_((dependence.num_ops() + 63) / 64, 0) {
    for (uint32_t op = 0; op < dependence.num_ops(); ++op) {
        remaining_preds_[op] = static_cast<uint32_t>(dependence.predecessors(op).size());
        if (remaining_preds_[op] == 0) {
            mark_ready(op);
        }
    }
}

void SearchFrontier::mark_ready(uint32_t op) {
    bitset_set(ready_bits_, op);
    lowest_ready_hint_ = std::min(lowest_ready_hint_, op);
    candidate_hint_ = std::min(candidate_hint_, op);
}

void SearchFrontier::mark_not_ready(uint32_t op) {
    bitset_clear(ready_bits_, op);
    // Only a cheap, exact-match bump: the op just removed was the hint's own
    // witness, so the hint must move at least past it, but the true new
    // minimum among whatever remains ready is not known without a scan --
    // lowest_ready()'s and first_candidate()'s own scans discover it lazily
    // instead.
    if (op == lowest_ready_hint_) {
        ++lowest_ready_hint_;
    }
    if (op == candidate_hint_) {
        ++candidate_hint_;
    }
}

uint32_t SearchFrontier::lowest_ready() const {
    uint32_t word = lowest_ready_hint_ / 64;
    if (word < ready_bits_.size()) {
        uint64_t pending = ready_bits_[word] & (~uint64_t{0} << (lowest_ready_hint_ % 64));
        while (pending == 0) {
            ++word;
            if (word >= ready_bits_.size()) {
                lowest_ready_hint_ = num_ops();
                pending = 0;
                break;
            }
            pending = ready_bits_[word];
        }
        if (pending != 0) {
            lowest_ready_hint_ = 64 * word + static_cast<uint32_t>(std::countr_zero(pending));
        }
    } else {
        lowest_ready_hint_ = num_ops();
    }
    assert((lowest_ready_hint_ == num_ops() || is_ready(lowest_ready_hint_)) &&
           "lowest_ready() must return num_ops() or an actually ready op");
    return lowest_ready_hint_;
}

void SearchFrontier::reset_expanding_memo() {
    // O(n/64) words, not the O(n) walk a per-op reset would cost; this runs
    // once per closure sweep entry and once per MeasureDormantRandom step
    // inside a sweep (see the class comment), far less often than the scans
    // it clears the way for.
    std::ranges::fill(known_expanding_bits_, uint64_t{0});
    // lowest_ready_hint_ is already a valid lower bound on the smallest
    // ready index, and every candidate is also ready, so it remains a valid
    // lower bound on the smallest ready-and-not-known-expanding index too.
    candidate_hint_ = lowest_ready_hint_;
}

void SearchFrontier::note_expanding(uint32_t op) {
    bitset_set(known_expanding_bits_, op);
    if (op == candidate_hint_) {
        ++candidate_hint_;
    }
}

std::optional<uint32_t> SearchFrontier::first_candidate() {
    uint32_t word = candidate_hint_ / 64;
    if (word >= ready_bits_.size()) {
        candidate_hint_ = num_ops();
        return std::nullopt;
    }
    uint64_t pending = (ready_bits_[word] & ~known_expanding_bits_[word]) &
                       (~uint64_t{0} << (candidate_hint_ % 64));
    while (pending == 0) {
        ++word;
        if (word >= ready_bits_.size()) {
            candidate_hint_ = num_ops();
            return std::nullopt;
        }
        pending = ready_bits_[word] & ~known_expanding_bits_[word];
    }
    candidate_hint_ = 64 * word + static_cast<uint32_t>(std::countr_zero(pending));
    return candidate_hint_;
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
std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir, SearchFrontier& frontier,
                                                 const DormantSubspace& subspace) {
    // first_candidate() scans ready_bits_ & ~known_expanding_bits_ a word at
    // a time (see SearchFrontier), so it already excludes every op the memo
    // has found expanding; the op it returns is the same lowest-index
    // ready-and-not-yet-classified op an unmemoized linear scan would reach
    // next. Because known-expanding ops never come back around, there is no
    // "memo hit" branch left to cross-check here: every op this loop sees is
    // one is_expanding has not yet classified this sweep, so it always runs
    // fresh.
    while (const std::optional<uint32_t> op = frontier.first_candidate()) {
        if (is_expanding(hir, hir.ops[*op], subspace)) {
            frontier.note_expanding(*op);
            continue;
        }
        return op;
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
//
// The frontier's expanding memo is reset on entry, since the caller may
// have just executed an expanding op (which shrinks the subspace) or be
// sweeping a different subspace than the frontier's last sweep saw, and
// again after every MeasureDormantRandom step, the only shrinking effect a
// sweep can execute; see SearchFrontier for why every other effect keeps a
// memoized verdict valid.
//
// `swept_ops` accumulates one count per op this sweep executes: the budget
// ActiveWidthScheduleOptions::search_budget bounds is measured in exactly
// this quantity, summed across every closure sweep and candidate replay in
// a run_beam_search call (see run_beam_search), so that the search's cost
// limit is reproducible across machines instead of depending on wall-clock
// speed.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<uint32_t>& newly_ready_log, size_t& swept_ops,
                 std::vector<WidthTransition>* transitions = nullptr) {
    frontier.reset_expanding_memo();
    while (const std::optional<uint32_t> op = find_ready_non_expanding(hir, frontier, subspace)) {
        const uint32_t newly_ready_count = frontier.execute(*op, newly_ready_log);
        log.push_back(UndoStep{*op, newly_ready_count});
        order.push_back(*op);
        ++swept_ops;
        const WidthTransition transition = classify_and_apply(hir, hir.ops[*op], subspace);
        assert(!is_expanding_effect(transition.effect) &&
               "find_ready_non_expanding chose an op classify_and_apply treats as expanding");
        if (transition.effect == WidthEffect::MeasureDormantRandom) {
            frontier.reset_expanding_memo();
        }
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
                                  const detail::ScheduleDependence& dependence, size_t& swept_ops) {
    BeamState state(SearchFrontier(dependence), DormantSubspace(hir.num_qubits));
    std::vector<UndoStep> discarded_log;
    std::vector<uint32_t> discarded_newly_ready;
    std::vector<WidthTransition> transitions;
    run_closure(hir, state.frontier, state.subspace, state.order, discarded_log,
                discarded_newly_ready, swept_ops, &transitions);
    absorb_closure_transitions(state, transitions);
    return state;
}

// Copies out the parent's current ready set before any candidate scoring
// below mutates it: score_candidates' loop executes and undoes each
// candidate against `parent.frontier` in turn, so iterating a live view of
// its ready ops while that same loop body edits them is unsafe. The scan is
// ascending (lowest_ready() up to num_ops()), so the candidate list
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
    const uint32_t start = frontier.lowest_ready();
    if (start >= frontier.num_ops()) {
        return ready;
    }
    const std::vector<uint64_t>& bits = frontier.ready_bits();
    uint32_t word = start / 64;
    uint64_t pending = bits[word] & (~uint64_t{0} << (start % 64));
    for (;;) {
        while (pending != 0) {
            ready.push_back(64 * word + static_cast<uint32_t>(std::countr_zero(pending)));
            pending &= pending - 1;
        }
        ++word;
        if (word >= bits.size()) {
            break;
        }
        pending = bits[word];
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
// near-complete) order vector -- is what keeps this affordable. `swept_ops`
// counts each candidate's first op plus its closure sweep, including
// candidates later discarded here or in run_beam_search's dedup/rank step:
// the work search_budget bounds is the work this function actually spends
// finding out a candidate is worth discarding, not just the work spent on
// eventual survivors. `parent` is
// left exactly as found on return.
//
// `candidate_budget_ops`, when set, is the second, higher threshold
// run_beam_search computes (the full *search_budget * hir.ops.size(), where
// the beam-narrowing threshold it checks between parents is only half of
// that -- see run_beam_search's comment for why the two differ). Once a
// scored candidate pushes swept_ops past it, the remaining ready expanding
// ops of this parent are left unscored: ready_ops_snapshot visits ops in
// ascending index, so the candidate that crosses the threshold is always
// the lowest-index one still unscored, and every later step of a search
// this far over budget scores only that one candidate too, since the
// count only grows from here.
std::vector<ScoredCandidate> score_candidates(const HirModule& hir, BeamState& parent,
                                              uint32_t parent_index,
                                              std::optional<double> candidate_budget_ops,
                                              size_t& swept_ops) {
    std::vector<ScoredCandidate> scored;
    for (uint32_t op : ready_ops_snapshot(parent.frontier)) {
        if (!is_expanding(hir, hir.ops[op], parent.subspace)) {
            continue;
        }

        DormantSubspace scratch(parent.subspace);
        std::vector<UndoStep> log;
        std::vector<uint32_t> newly_ready_log;

        log.push_back(UndoStep{op, parent.frontier.execute(op, newly_ready_log)});
        ++swept_ops;
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
        run_closure(hir, parent.frontier, scratch, candidate.ops, log, newly_ready_log, swept_ops,
                    &swept);
        absorb_transitions(candidate.peak, candidate.dense_work, swept);

        candidate.width_after_closure = scratch.active_width();
        candidate.executed_bits = parent.frontier.executed_bits();

        undo_all(parent.frontier, log, newly_ready_log);
        scored.push_back(std::move(candidate));

        if (candidate_budget_ops && static_cast<double>(swept_ops) > *candidate_budget_ops) {
            break;
        }
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
// not just numerically). `swept_ops` counts this replay's ops too: it
// redoes real work (score_candidates' own execute()/undo() bracket left no
// trace behind to reuse), so it counts against the same budget.
BeamState materialize_candidate(const HirModule& hir, const std::vector<BeamState>& beam,
                                const ScoredCandidate& candidate, size_t& swept_ops) {
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
        ++swept_ops;
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

// True when candidate a's complete prospective order -- its parent's
// already-executed order followed by the ops this step adds -- is
// lexicographically less than candidate b's. a and b are assumed to share
// an executed-op set (the caller only ever compares within one dedup
// group), so the two sequences are permutations of the same op indices and
// therefore always differ at some position; the length fallback below is
// only a defensive tie-break for two literally identical sequences.
// Comparing the full sequence, rather than just first_op, keeps the result
// independent of which parent produced which candidate -- and therefore
// independent of the order `generation` happened to be built in -- since
// two candidates from different parents can otherwise share a first_op.
bool prospective_order_less(const std::vector<BeamState>& beam, const ScoredCandidate& a,
                            const ScoredCandidate& b) {
    const std::vector<uint32_t>& a_parent = beam[a.parent_index].order;
    const std::vector<uint32_t>& b_parent = beam[b.parent_index].order;
    const size_t a_len = a_parent.size() + a.ops.size();
    const size_t b_len = b_parent.size() + b.ops.size();
    for (size_t i = 0; i < a_len && i < b_len; ++i) {
        const uint32_t a_val = i < a_parent.size() ? a_parent[i] : a.ops[i - a_parent.size()];
        const uint32_t b_val = i < b_parent.size() ? b_parent[i] : b.ops[i - b_parent.size()];
        if (a_val != b_val) {
            return a_val < b_val;
        }
    }
    return a_len < b_len;
}

// True when `dominator` dominates `dominated` on the pass's own
// lexicographic objective: no worse on both peak and dense_work, strictly
// better on at least one. Two candidates tied on both never dominate each
// other under this definition -- append_pareto_front collapses those to one
// canonical survivor via prospective_order_less before dominance is ever
// checked.
bool dominates(const ScoredCandidate& dominator, const ScoredCandidate& dominated) {
    const bool no_worse =
        dominator.peak <= dominated.peak && dominator.dense_work <= dominated.dense_work;
    const bool strictly_better =
        dominator.peak < dominated.peak || dominator.dense_work < dominated.dense_work;
    return no_worse && strictly_better;
}

// Reduces one executed-bitset group, generation[begin, end), to its
// non-dominated (Pareto) front over (peak, dense_work), moving the
// survivors onto the end of `deduped`. See the dedup comment in
// run_beam_search for why a single lexicographic winner is not enough here.
// Two passes over the group: first collapse every exact (peak, dense_work)
// tie to one canonical representative, picked by prospective_order_less so
// the result does not depend on `generation`'s order; then drop every
// representative some other representative dominates. Both passes are a
// plain O(k^2) scan over the group -- the front is expected to stay tiny in
// practice (one candidate per distinct peak reached so far), so a sorted or
// indexed structure would only add bookkeeping for no measurable benefit.
void append_pareto_front(const std::vector<BeamState>& beam,
                         std::vector<ScoredCandidate>& generation, size_t begin, size_t end,
                         std::vector<ScoredCandidate>& deduped) {
    std::vector<size_t> representatives;
    for (size_t i = begin; i < end; ++i) {
        bool collapsed = false;
        for (size_t& rep : representatives) {
            if (generation[rep].peak == generation[i].peak &&
                generation[rep].dense_work == generation[i].dense_work) {
                if (prospective_order_less(beam, generation[i], generation[rep])) {
                    rep = i;
                }
                collapsed = true;
                break;
            }
        }
        if (!collapsed) {
            representatives.push_back(i);
        }
    }

    for (size_t rep : representatives) {
        bool is_dominated = false;
        for (size_t other : representatives) {
            if (other != rep && dominates(generation[other], generation[rep])) {
                is_dominated = true;
                break;
            }
        }
        if (!is_dominated) {
            deduped.push_back(std::move(generation[rep]));
        }
    }
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
//
// search_budget bounds the beam-search cost as a multiple of hir.ops.size(),
// counted in swept_ops (see run_closure, score_candidates, and
// materialize_candidate) rather than wall-clock time, so the point at which
// the search narrows is the same on every machine and a compiled plan is
// reproducible. It backs two graduated, independently-triggered responses,
// both checked against the same running swept_ops count:
//
//   1. Beam narrowing, at half the budget (swept_ops exceeds
//      parent_budget_ops == 0.5 * *search_budget * hir.ops.size()). Inside
//      the scoring loop below, the remaining lower-ranked parents are
//      dropped unscored as soon as the count is over this threshold (beam
//      is always in ranked order, since next_beam is filled from the
//      ranked deduped list, so the parents dropped are the weakest), and
//      the cut after the generation keeps a single survivor from then on.
//   2. Candidate narrowing, at the full budget (swept_ops exceeds
//      candidate_budget_ops == *search_budget * hir.ops.size(), twice the
//      first threshold). Inside score_candidates itself, the remaining
//      ready expanding ops of whichever parent is being scored when the
//      count crosses this threshold are left unscored, so every later step
//      of an already-narrowed, already-over-this-threshold search scores
//      only its one surviving parent's lowest-index ready candidate.
//
// Splitting the single search_budget into two thresholds this way lets the
// beam narrow to its cheapest useful shape (one parent) well before that
// surviving parent's own candidates stop being compared to each other, so
// a circuit whose ready-candidate count per step stays small keeps
// exploring properly-ranked choices for the second half of the budget too
// -- which is what keeps this search's schedule quality close to an
// unbounded search's on such circuits. Candidate narrowing is the
// backstop for circuits where that count does not stay small: without it,
// a single surviving parent with many simultaneously ready, mutually
// independent expanding rotations would still re-score all of them at
// every remaining step, making the remaining cost grow with the square of
// that count instead of staying linear in it. Checking before each parent
// and before each candidate, rather than only between steps or only once a
// whole generation finishes, also bounds the worst-case overshoot past
// either threshold to one parent's or one candidate's own sweep, instead
// of an entire generation's cost. Altogether, total cost stays near the
// full budget itself plus about four traces of the circuit: one for the
// unconditional initial closure before the loop starts, one for the
// candidate whose scoring first crosses candidate_budget_ops (its own
// sweep still runs to completion before either check can fire), and one
// more each for the sweep and the replay every remaining single-beam step
// performs. Narrowing rather than aborting outright keeps every guarantee
// this function already gives: every order the search can still produce
// is a legal linear extension of `dependence`, and
// ActiveWidthSchedulePass::run's incumbent comparison still applies to
// whatever this returns, so a narrowed search can only give up some of the
// wide beam's improvement over the incumbent, never regress past it.
std::vector<uint32_t> run_beam_search(const HirModule& hir,
                                      const detail::ScheduleDependence& dependence,
                                      uint32_t beam_width, std::optional<double> search_budget,
                                      size_t& swept_ops) {
    swept_ops = 0;
    const std::optional<double> candidate_budget_ops =
        search_budget ? std::optional<double>(*search_budget * static_cast<double>(hir.ops.size()))
                      : std::nullopt;
    const std::optional<double> parent_budget_ops =
        candidate_budget_ops ? std::optional<double>(0.5 * *candidate_budget_ops) : std::nullopt;

    std::vector<BeamState> beam;
    beam.push_back(make_initial_beam_state(hir, dependence, swept_ops));

    std::vector<BeamState> completed;
    while (!beam.empty()) {
        std::vector<ScoredCandidate> generation;
        std::vector<bool> parent_has_candidates(beam.size(), false);
        for (uint32_t i = 0; i < beam.size(); ++i) {
            if (i > 0 && parent_budget_ops && static_cast<double>(swept_ops) > *parent_budget_ops) {
                // Over the beam-narrowing threshold: drop the remaining
                // parents unscored, so the loop below does not mistake them
                // for completed states. erase(), not resize(): BeamState
                // has no default constructor.
                beam.erase(beam.begin() + i, beam.end());
                parent_has_candidates.resize(beam.size());
                break;
            }
            std::vector<ScoredCandidate> scored =
                score_candidates(hir, beam[i], i, candidate_budget_ops, swept_ops);
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
        // executed ops always reaches the same subspace and width, so a
        // candidate outside its own set's Pareto front never has to be
        // materialized. But peak-so-far and dense-work-so-far are path
        // properties, not confluent ones -- peak is a running max over a
        // path-dependent sequence of transitions, and dense_work is the
        // pass's own secondary objective -- so within one executed set
        // neither key dominates the other until the rest of the schedule is
        // known. A candidate reached with lower peak but higher dense work
        // can still lose to one reached with higher peak but lower dense
        // work, if the shared suffix later forces both up to that same
        // higher peak anyway: at that point only the lower-dense_work
        // duplicate was worth keeping, and a single lexicographic winner
        // picked here cannot see that coming. So this keeps every candidate
        // in a set that no other candidate of the same set dominates (peak
        // no worse and dense work no worse, at least one strictly better),
        // collapsing candidates tied on both to one canonical survivor (see
        // append_pareto_front). The front is expected to be tiny in
        // practice (one candidate per distinct peak), so no special data
        // structure is needed. The global beam_width cut below then ranks
        // over every surviving candidate across every set, exactly as it
        // did when dedup left one candidate per set.
        std::ranges::sort(generation, [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.executed_bits < b.executed_bits;
        });
        std::vector<ScoredCandidate> deduped;
        for (size_t group_begin = 0; group_begin < generation.size();) {
            size_t group_end = group_begin + 1;
            while (group_end < generation.size() &&
                   generation[group_end].executed_bits == generation[group_begin].executed_bits) {
                ++group_end;
            }
            append_pareto_front(beam, generation, group_begin, group_end, deduped);
            group_begin = group_end;
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
        const size_t width =
            parent_budget_ops && static_cast<double>(swept_ops) > *parent_budget_ops ? 1
                                                                                     : beam_width;
        next_beam.reserve(std::min<size_t>(deduped.size(), width));
        for (size_t i = 0; i < deduped.size() && i < width; ++i) {
            next_beam.push_back(materialize_candidate(hir, beam, deduped[i], swept_ops));
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

// Every call site below passes ops that are currently adjacent in `order`,
// which is exactly the case the adjacency lemma in schedule_dependence.h
// covers: a missing edge between two ops that are actually next to each
// other in a schedule cannot be standing in for an implied constraint (that
// would require some other op strictly between them, contradicting
// adjacency), so checking direct edges here is as good as checking the
// full closure.
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
    if (options_.search_budget && !detail::is_finite_non_negative(*options_.search_budget)) {
        throw std::invalid_argument(
            "ActiveWidthSchedulePass: search_budget must be a finite, non-negative value");
    }
}

void ActiveWidthSchedulePass::run(HirModule& hir) {
    built_dependence_ = false;
    swept_ops_ = 0;

    const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
    incumbent_peak_ = incumbent_trace.peak_width;
    incumbent_dense_work_ = estimate_dense_work(incumbent_trace);

    // See the header comment's "Early exit": with nothing for a scheduler to
    // choose among, report the incumbent unchanged rather than pay to build
    // a ScheduleDependence at all. swept_ops_ stays zero: the beam search
    // that would have spent budget never runs.
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

    std::vector<uint32_t> order =
        run_beam_search(hir, dependence, options_.beam_width, options_.search_budget, swept_ops_);

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
