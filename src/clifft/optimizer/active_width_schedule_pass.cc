#include "clifft/optimizer/active_width_schedule_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/schedule_dependence.h"
#include "clifft/util/numeric.h"

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

// Search branches only on ready operations that increase active width.
// Closure eagerly runs the remaining ready ops in original-index order.
// Moving a ready non-expanding op ahead of independent ops cannot worsen the
// attainable peak: measurements cannot raise width, and neutral rotations stay
// neutral across independent updates. This does not guarantee minimum dense work.
//
// Legal prefixes with the same executed-op set reach the same dormant subspace:
// updates for independent operations commute. Their peak and dense-work histories
// can differ, so candidate deduplication must retain both costs.

// Only executions control beam narrowing. Count closure probes separately:
// an expansion can be queried repeatedly without executing.
struct SearchWork {
    std::optional<double> limit;
    size_t swept_ops = 0;
    size_t probes = 0;
    // Hints are rechecked against each current basis, so speculative branches
    // can share them without copying or undoing them with the search state.
    std::vector<uint32_t> anticommuting_rows;

    [[nodiscard]] bool greedy() const { return limit && static_cast<double>(swept_ops) > *limit; }
    [[nodiscard]] bool narrow() const {
        return limit && static_cast<double>(swept_ops) > 0.5 * *limit;
    }
};

// Offsets into a shared undo log avoid allocating a successor list per step.
struct UndoStep {
    uint32_t op = 0;
    uint32_t newly_ready_count = 0;
};

// Tracks executed and ready ops for speculative execute/undo steps. Bitsets
// avoid per-update allocations; scan hints skip prefixes already known to be empty.
//
// Within a closure sweep, an expanding rotation stays expanding while S only
// grows. A dormant-random measurement replaces a generator and can invalidate
// that result, so run_closure clears the expansion memo after such measurements.
class SearchFrontier {
  public:
    explicit SearchFrontier(const detail::ScheduleDependence& dependence);

    [[nodiscard]] bool is_ready(uint32_t op) const {
        return (ready_bits_[op / 64] & (uint64_t{1} << (op % 64))) != 0;
    }
    [[nodiscard]] uint32_t num_ops() const {
        return static_cast<uint32_t>(remaining_preds_.size());
    }

    // Returns the first ready op, or num_ops(), and caches the scan position.
    [[nodiscard]] uint32_t lowest_ready() const;

    [[nodiscard]] bool has_single_ready() const;

    // Cached expansion results are valid only within the current closure sweep.
    void reset_expanding_memo();
    void note_expanding(uint32_t op);

    // First ready op not already known to expand, or nullopt if none remains.
    [[nodiscard]] std::optional<uint32_t> first_candidate();

    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] const std::vector<uint64_t>& ready_bits() const { return ready_bits_; }
    [[nodiscard]] size_t executed_count() const { return executed_count_; }

    // Append newly ready successors to the shared log and return their count
    // so undo() can reverse this step without a separate allocation.
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
    // Keep hints as lower bounds; defer finding the next ready bit until queried.
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

bool SearchFrontier::has_single_ready() const {
    const uint32_t first = lowest_ready();
    if (first == num_ops() || !std::has_single_bit(ready_bits_[first / 64])) {
        return false;
    }
    // Stop on a second nonempty word instead of allocating a ready-op snapshot.
    for (size_t word = first / 64 + 1; word < ready_bits_.size(); ++word) {
        if (ready_bits_[word] != 0) {
            return false;
        }
    }
    return true;
}

void SearchFrontier::reset_expanding_memo() {
    std::ranges::fill(known_expanding_bits_, uint64_t{0});
    // Every candidate is ready, so the ready hint is also a candidate lower bound.
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
    // Restoring an earlier op may lower both scan hints.
    mark_ready(op);
}

void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log,
              std::vector<uint32_t>& newly_ready_log) {
    for (auto it = log.rbegin(); it != log.rend(); ++it) {
        frontier.undo(it->op, it->newly_ready_count, newly_ready_log);
    }
}

// Lowest-index ready op that is not expanding, or nullopt when every
// currently ready op is expanding.
std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir, SearchFrontier& frontier,
                                                 const DormantSubspace& subspace,
                                                 SearchWork& work) {
    // The bitset scan skips ops whose expansion verdict is still valid.
    while (const std::optional<uint32_t> op = frontier.first_candidate()) {
        ++work.probes;
        if (detail::is_expanding(hir, hir.ops[*op], subspace, work.anticommuting_rows[*op])) {
            frontier.note_expanding(*op);
            continue;
        }
        return op;
    }
    return std::nullopt;
}

// Log executions for speculative undo. Reset the expansion memo on entry
// and after generator replacement, when an earlier verdict may no longer hold.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log,
                 std::vector<uint32_t>& newly_ready_log, SearchWork& work, uint32_t& peak,
                 double& dense_work) {
    frontier.reset_expanding_memo();
    while (const std::optional<uint32_t> op =
               find_ready_non_expanding(hir, frontier, subspace, work)) {
        const uint32_t newly_ready_count = frontier.execute(*op, newly_ready_log);
        log.push_back(UndoStep{*op, newly_ready_count});
        order.push_back(*op);
        ++work.swept_ops;
        const WidthTransition transition = detail::apply_non_expanding(hir, hir.ops[*op], subspace);
        assert(!is_expanding_effect(transition.effect) &&
               "find_ready_non_expanding chose an op classify_and_apply treats as expanding");
        if (transition.effect == WidthEffect::MeasureDormantRandom) {
            frontier.reset_expanding_memo();
        }
        peak = std::max(peak, transition.after);
        dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
}

// A partial schedule after closure, with accumulated costs and replay state.

struct BeamState {
    BeamState(SearchFrontier frontier_in, DormantSubspace subspace_in)
        : frontier(std::move(frontier_in)), subspace(std::move(subspace_in)) {}

    SearchFrontier frontier;
    DormantSubspace subspace;
    uint32_t peak = 0;
    double dense_work = 0.0;
    std::vector<uint32_t> order;
};

BeamState make_initial_beam_state(const HirModule& hir,
                                  const detail::ScheduleDependence& dependence, SearchWork& work) {
    BeamState state(SearchFrontier(dependence), DormantSubspace(hir.num_qubits));
    std::vector<UndoStep> discarded_log;
    std::vector<uint32_t> discarded_newly_ready;
    run_closure(hir, state.frontier, state.subspace, state.order, discarded_log,
                discarded_newly_ready, work, state.peak, state.dense_work);
    return state;
}

// Scoring mutates and undoes the frontier, so iterate a snapshot. Original-index
// order makes exploration reproducible even when the budget ends mid-search.
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

// A prospective child, scored before cloning the parent's frontier and prefix.
// The executed bitset supports deduplication; ops contains the expansion and closure.
struct ScoredCandidate {
    uint32_t parent_index = 0;
    std::vector<uint32_t> ops;
    std::vector<uint64_t> executed_bits;
    uint32_t width_after_closure = 0;
    uint32_t peak = 0;
    double dense_work = 0.0;
    uint32_t first_op = 0;
};

// A closed parent has only expanding ready ops; no second classification is
// needed. Clone its subspace, but mutate and undo its frontier so candidates
// later discarded do not each need a full frontier and prefix copy.
// Stop comparing alternatives at the swept-op threshold.
std::vector<ScoredCandidate> score_candidates(const HirModule& hir, BeamState& parent,
                                              uint32_t parent_index, SearchWork& work) {
    std::vector<ScoredCandidate> scored;
    for (uint32_t op : ready_ops_snapshot(parent.frontier)) {
        DormantSubspace scratch(parent.subspace);
        std::vector<UndoStep> log;
        std::vector<uint32_t> newly_ready_log;

        log.push_back(UndoStep{op, parent.frontier.execute(op, newly_ready_log)});
        ++work.swept_ops;
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

        run_closure(hir, parent.frontier, scratch, candidate.ops, log, newly_ready_log, work,
                    candidate.peak, candidate.dense_work);

        candidate.width_after_closure = scratch.active_width();
        candidate.executed_bits = parent.frontier.executed_bits();

        undo_all(parent.frontier, log, newly_ready_log);
        scored.push_back(std::move(candidate));

        if (work.greedy()) {
            break;
        }
    }
    return scored;
}

// Replay only surviving candidates. Scoring and replay accumulate identical
// transitions in identical order, so even floating-point costs agree exactly.
// Charge replay to the swept-op budget as well as speculative execution.
BeamState materialize_candidate(const HirModule& hir, const std::vector<BeamState>& beam,
                                const ScoredCandidate& candidate, SearchWork& work) {
    const BeamState& parent = beam[candidate.parent_index];
    BeamState state(parent.frontier, parent.subspace);
    state.peak = parent.peak;
    state.dense_work = parent.dense_work;
    state.order = parent.order;
    state.order.reserve(state.order.size() + candidate.ops.size());

    // Reuse one log buffer even though replay never needs to undo its entries.
    std::vector<uint32_t> discarded_newly_ready;
    for (uint32_t op : candidate.ops) {
        state.frontier.execute(op, discarded_newly_ready);
        const WidthTransition transition = classify_and_apply(hir, hir.ops[op], state.subspace);
        state.peak = std::max(state.peak, transition.after);
        state.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
        state.order.push_back(op);
        ++work.swept_ops;
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

// Break equal-cost ties by the full order for deterministic selection.
bool completed_beats(const BeamState& candidate, const BeamState& incumbent) {
    if (candidate.peak != incumbent.peak) {
        return candidate.peak < incumbent.peak;
    }
    if (candidate.dense_work != incumbent.dense_work) {
        return candidate.dense_work < incumbent.dense_work;
    }
    return candidate.order < incumbent.order;
}

// Compare complete prospective orders to resolve equal-cost duplicates
// independently of which parent generated them.
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

// Strict Pareto dominance: no worse on either cost and better on at least one.
bool dominates(const ScoredCandidate& dominator, const ScoredCandidate& dominated) {
    const bool no_worse =
        dominator.peak <= dominated.peak && dominator.dense_work <= dominated.dense_work;
    const bool strictly_better =
        dominator.peak < dominated.peak || dominator.dense_work < dominated.dense_work;
    return no_worse && strictly_better;
}

// Keep the Pareto front within one executed-set group, with a canonical order
// for equal costs. Groups are small enough for pairwise comparisons.
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

// Narrow the beam at half the swept-op budget and choose the lowest ready
// expansion after the full swept-op budget. Closure still runs to completion:
// the budget bounds executions, not classification probes or wall time.
std::vector<uint32_t> run_beam_search(const HirModule& hir,
                                      const detail::ScheduleDependence& dependence,
                                      uint32_t beam_width, SearchWork& work) {
    std::vector<BeamState> beam;
    beam.push_back(make_initial_beam_state(hir, dependence, work));

    std::optional<BeamState> best;
    while (!beam.empty()) {
        // Avoid speculation when the choice is fixed. Before a finite budget
        // is exhausted, retain replay accounting so narrowing happens at the same point.
        if (beam.size() == 1 && beam.front().frontier.lowest_ready() < dependence.num_ops() &&
            (work.greedy() || (!work.limit && beam.front().frontier.has_single_ready()))) {
            BeamState& state = beam.front();
            const uint32_t op = state.frontier.lowest_ready();
            std::vector<uint32_t> newly_ready;
            state.frontier.execute(op, newly_ready);
            state.order.push_back(op);
            ++work.swept_ops;
            const WidthTransition transition = classify_and_apply(hir, hir.ops[op], state.subspace);
            state.peak = std::max(state.peak, transition.after);
            state.dense_work += detail::dense_work_contribution(
                transition.effect, transition.before, transition.after);
            std::vector<UndoStep> log;
            run_closure(hir, state.frontier, state.subspace, state.order, log, newly_ready, work,
                        state.peak, state.dense_work);
            continue;
        }

        std::vector<ScoredCandidate> generation;
        std::vector<bool> parent_has_candidates(beam.size(), false);
        for (uint32_t i = 0; i < beam.size(); ++i) {
            if (i > 0 && work.narrow()) {
                // Remove unscored parents so they cannot be mistaken for completed states.
                beam.erase(beam.begin() + i, beam.end());
                parent_has_candidates.resize(beam.size());
                break;
            }
            std::vector<ScoredCandidate> scored = score_candidates(hir, beam[i], i, work);
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
            // In a DAG, an unfinished closed state must have a ready expansion.
            assert(beam[i].frontier.executed_count() == dependence.num_ops() &&
                   "a closed beam state with no ready expanding op must have executed every op");
            if (!best || completed_beats(beam[i], *best)) {
                best = std::move(beam[i]);
            }
        }

        if (generation.empty()) {
            break;
        }

        // Equal executed sets have the same subspace but may have different costs.
        // Keep their Pareto front: a shared suffix can erase a lower-peak advantage,
        // making the candidate with lower accumulated dense work preferable.
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

        // Prefer lower peak and current width, then longer closures as a
        // heuristic for progress toward finishing the remaining operations.
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
            if (a.first_op != b.first_op) {
                return a.first_op < b.first_op;
            }
            // Each parent contributes one candidate per first_op, so its
            // deterministic rank completes the ordering without a prefix scan.
            return a.parent_index < b.parent_index;
        });

        std::vector<BeamState> next_beam;
        const size_t width = work.narrow() ? 1 : beam_width;
        next_beam.reserve(std::min<size_t>(deduped.size(), width));
        for (size_t i = 0; i < deduped.size() && i < width; ++i) {
            next_beam.push_back(materialize_candidate(hir, beam, deduped[i], work));
        }
        beam = std::move(next_beam);
    }

    assert(best.has_value() &&
           "beam_width >= 1 (the constructor rejects 0) guarantees at least one completed state");
    return best->order;
}

// ---------------------------------------------------------------------------
// Neutral-rotation sinking: a rightward bubble per RotationNeutral op.
// ---------------------------------------------------------------------------

// For adjacent ops in a legal order, any dependency has a direct edge;
// see ScheduleDependence. Callers must only pass adjacent ops.
bool independent(const detail::ScheduleDependence& dependence, uint32_t a, uint32_t b) {
    return !std::ranges::binary_search(dependence.successors(a), b) &&
           !std::ranges::binary_search(dependence.predecessors(a), b);
}

// Index effects by op so they survive permutation. Sinking a neutral rotation
// does not change S, so the other operations retain their classifications.
std::vector<WidthEffect> effect_by_op_index(const HirModule& hir,
                                            const std::vector<uint32_t>& order) {
    std::vector<WidthEffect> effect(hir.ops.size(), WidthEffect::None);
    DormantSubspace subspace(hir.num_qubits);
    for (uint32_t op : order) {
        effect[op] = classify_and_apply(hir, hir.ops[op], subspace).effect;
    }
    return effect;
}

// Sink neutral rotations across independent, non-expanding ops, which cannot
// increase the width they run at. Stabilizer rotations need no action or sinking.
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

    // Earlier bubbles can shift a rotation before its own turn to move.
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

// Only rotations offer a choice of expansion: an expanding instrument is a
// barrier and is always the sole ready op.
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
    if (options_.search_budget && !is_finite_non_negative(*options_.search_budget)) {
        throw std::invalid_argument(
            "ActiveWidthSchedulePass: search_budget must be a finite, non-negative value");
    }
}

void ActiveWidthSchedulePass::run(HirModule& hir) {
    swept_ops_ = 0;
    classification_probes_ = 0;

    const ActiveWidthTrace incumbent_trace = analyze_active_width(hir);
    incumbent_peak_ = incumbent_trace.peak_width;
    incumbent_dense_work_ = estimate_dense_work(incumbent_trace);

    // Avoid graph construction when no rotation can benefit from scheduling.
    // The incumbent statistics still describe the input.
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

    SearchWork work{
        .limit = options_.search_budget
                     ? std::optional<double>(*options_.search_budget * hir.ops.size())
                     : std::nullopt,
        .anticommuting_rows = std::vector<uint32_t>(hir.ops.size(), hir.num_qubits),
    };
    std::vector<uint32_t> order = run_beam_search(hir, dependence, options_.beam_width, work);
    swept_ops_ = work.swept_ops;
    classification_probes_ = work.probes;

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
