#include "clifft/optimizer/active_width_search.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/tableau/pauli_string.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// ---------------------------------------------------------------------------
// Op-level width effect, mirroring analyze_active_width's own dispatch
// (active_width_analysis.cc) against the same public DormantSubspace
// surface. That function threads a single ActiveWidthTrace through a fixed
// op order; this search instead needs to classify a ready op without
// committing to it (closure and candidate gathering try several before
// picking one) and, separately, to commit once a choice is made, so the
// dispatch is split into a pure predicate and a mutating apply rather than
// reusing analyze_active_width's trace-building form directly.
// ---------------------------------------------------------------------------

PauliString pauli_body(const HirModule& hir, const HeisenbergOp& op) {
    PauliString result(hir.num_qubits);
    result.mut_x().xor_with(hir.destab_mask(op));
    result.mut_z().xor_with(hir.stab_mask(op));
    return result;
}

// True when executing `op` against `subspace` would raise the active width:
// a T_GATE/PHASE_ROTATION whose axis does not commute with every generator
// of S, or an INSTRUMENT that takes the Activate branch. Pure query (no
// mutation), so closure and candidate gathering can test many ready ops
// before committing to one. Must keep classifying identically to apply_op's
// own branches below.
bool is_expanding(const HirModule& hir, const HeisenbergOp& op, const DormantSubspace& subspace) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            return !subspace.commutes_with_all(pauli_body(hir, op));
        case OpType::INSTRUMENT: {
            const PauliString body = pauli_body(hir, op);
            if (subspace.commutes_with_all(body)) {
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

// Applies `op`'s effect to `subspace` (rotation, measurement, or the same
// four-branch instrument rule analyze_active_width uses; every other op
// type is inert) and returns whether it was expanding. Safe to call on any
// ready op regardless of its classification: apply_rotation and
// apply_measurement already decide their own branch from the current
// subspace, so calling this on a non-expanding op is simply a no-op (or a
// width-decreasing collapse for a measurement) rather than an error.
bool apply_op(const HirModule& hir, const HeisenbergOp& op, DormantSubspace& subspace) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            return subspace.apply_rotation(pauli_body(hir, op));
        case OpType::MEASURE:
            subspace.apply_measurement(pauli_body(hir, op));
            return false;
        case OpType::INSTRUMENT: {
            const PauliString body = pauli_body(hir, op);
            if (!subspace.commutes_with_all(body)) {
                const InstrumentSite& site =
                    hir.instrument_sites.at(static_cast<uint32_t>(op.instrument_site_idx()));
                const bool traps = hir.neglect_instrument_damping ||
                                   site.probabilities.p_fire[0] == site.probabilities.p_fire[1];
                if (!traps) {
                    [[maybe_unused]] const bool promoted = subspace.apply_rotation(body);
                    assert(promoted && "instrument body must anticommute with S here");
                    return true;
                }
            }
            return false;
        }
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Exact executed-op bitset memo key. See active_width_search.h for why the
// key must be the literal bitset rather than a hash digest treated as an
// identity.
// ---------------------------------------------------------------------------

void bitset_set(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] |= (uint64_t{1} << (index % 64));
}

void bitset_clear(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64] &= ~(uint64_t{1} << (index % 64));
}

// Hash for bucketing only. std::unordered_set's own key equality
// (std::vector<uint64_t>::operator==, exact and element-wise) is what
// decides identity; two bitsets that hash alike but differ are still kept
// apart as distinct keys.
struct BitsetHash {
    size_t operator()(const std::vector<uint64_t>& bits) const noexcept {
        size_t seed = bits.size();
        for (uint64_t word : bits) {
            seed ^= std::hash<uint64_t>{}(word) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using FailedNodeMemo = std::unordered_set<std::vector<uint64_t>, BitsetHash>;

// ---------------------------------------------------------------------------
// Incremental readiness bookkeeping.
// ---------------------------------------------------------------------------

// Tracks which ops are executed and which are ready (every predecessor
// executed) as one mutable structure shared by the whole DFS, updated via
// execute()/undo() pairs bracketing each recursive call. Readiness and the
// executed set are both pure functions of the executed-op set alone (see
// active_width_search.h's confluence note), so undo() only has to reverse
// exactly what its matching execute() call did, never recompute anything
// from scratch.
class SearchFrontier {
  public:
    explicit SearchFrontier(const ScheduleDependence& dependence)
        : dependence_(&dependence),
          executed_((dependence.num_ops() + 63) / 64, 0),
          remaining_preds_(dependence.num_ops()) {
        for (uint32_t op = 0; op < dependence.num_ops(); ++op) {
            remaining_preds_[op] = static_cast<uint32_t>(dependence.predecessors(op).size());
            if (remaining_preds_[op] == 0) {
                ready_.insert(op);
            }
        }
    }

    [[nodiscard]] const std::set<uint32_t>& ready() const { return ready_; }
    [[nodiscard]] const std::vector<uint64_t>& executed_bits() const { return executed_; }
    [[nodiscard]] size_t executed_count() const { return executed_count_; }

    // Marks `op` executed and returns the successors that newly became
    // ready as a result, which undo() needs to reverse this call exactly.
    std::vector<uint32_t> execute(uint32_t op) {
        assert(ready_.contains(op) && "execute() called on a non-ready op");
        ready_.erase(op);
        bitset_set(executed_, op);
        ++executed_count_;
        std::vector<uint32_t> newly_ready;
        for (uint32_t succ : dependence_->successors(op)) {
            if (--remaining_preds_[succ] == 0) {
                ready_.insert(succ);
                newly_ready.push_back(succ);
            }
        }
        return newly_ready;
    }

    // Reverses exactly the execute(op) call that returned `newly_ready`.
    void undo(uint32_t op, const std::vector<uint32_t>& newly_ready) {
        for (uint32_t succ : dependence_->successors(op)) {
            ++remaining_preds_[succ];
        }
        for (uint32_t r : newly_ready) {
            ready_.erase(r);
        }
        bitset_clear(executed_, op);
        --executed_count_;
        ready_.insert(op);
    }

  private:
    const ScheduleDependence* dependence_;
    std::vector<uint64_t> executed_;
    std::vector<uint32_t> remaining_preds_;
    std::set<uint32_t> ready_;
    size_t executed_count_ = 0;
};

// An executed op paired with the successors execute() found newly ready:
// everything undo() needs to reverse that one call.
using UndoStep = std::pair<uint32_t, std::vector<uint32_t>>;

void undo_all(SearchFrontier& frontier, const std::vector<UndoStep>& log) {
    for (auto it = log.rbegin(); it != log.rend(); ++it) {
        frontier.undo(it->first, it->second);
    }
}

// Lowest-index ready op that is not expanding, or nullopt when every
// currently ready op (if any) is expanding.
std::optional<uint32_t> find_ready_non_expanding(const HirModule& hir,
                                                 const SearchFrontier& frontier,
                                                 const DormantSubspace& subspace) {
    for (uint32_t op : frontier.ready()) {
        if (!is_expanding(hir, hir.ops[op], subspace)) {
            return op;
        }
    }
    return std::nullopt;
}

// Executes every ready non-expanding op, lowest index first, until none is
// ready: the closure step the search's correctness depends on (see
// active_width_search.h). Appends each executed op, in execution order, to
// `order` and logs it in `log` for the caller to undo later if needed.
void run_closure(const HirModule& hir, SearchFrontier& frontier, DormantSubspace& subspace,
                 std::vector<uint32_t>& order, std::vector<UndoStep>& log) {
    while (const std::optional<uint32_t> op = find_ready_non_expanding(hir, frontier, subspace)) {
        log.emplace_back(*op, frontier.execute(*op));
        order.push_back(*op);
        [[maybe_unused]] const bool expanding = apply_op(hir, hir.ops[*op], subspace);
        assert(!expanding && "find_ready_non_expanding chose an op apply_op treats as expanding");
    }
}

// ---------------------------------------------------------------------------
// Threshold feasibility DFS.
// ---------------------------------------------------------------------------

enum class SearchOutcome : uint8_t { Success, Infeasible, Unknown };

struct SearchContext {
    const HirModule& hir;
    const ScheduleDependence& dependence;
    uint64_t node_budget;
    uint32_t threshold = 0;
    uint64_t explored = 0;
    FailedNodeMemo memo;
};

// One branch at a search node: the ready expanding op chosen to execute
// next, plus everything closure swept in immediately after it, precomputed
// once so ranking candidates never has to redo the work for whichever one
// is ultimately chosen.
struct Candidate {
    // DormantSubspace has no default constructor, so this struct needs one
    // too: subspace_after always starts as a copy of the parent node's
    // subspace, never a fresh empty one.
    explicit Candidate(const DormantSubspace& initial_subspace)
        : subspace_after(initial_subspace) {}

    uint32_t first_op = 0;
    uint32_t width_after_closure = 0;
    std::vector<uint32_t> ops;
    DormantSubspace subspace_after;
};

// Executes `first_op` (already confirmed ready, expanding, and within
// threshold by the caller) and closes the result, against the shared
// `frontier`, then undoes both so the frontier is back to its caller-visible
// state on return -- this only evaluates what committing to `first_op` would
// look like, for ranking purposes, and does not itself commit to it.
Candidate simulate_candidate(const HirModule& hir, SearchFrontier& frontier,
                             const DormantSubspace& subspace, uint32_t first_op) {
    Candidate candidate(subspace);
    candidate.first_op = first_op;
    std::vector<UndoStep> log;

    log.emplace_back(first_op, frontier.execute(first_op));
    candidate.ops.push_back(first_op);
    [[maybe_unused]] const bool expanding =
        apply_op(hir, hir.ops[first_op], candidate.subspace_after);
    assert(expanding && "simulate_candidate called on an op the caller did not classify expanding");

    run_closure(hir, frontier, candidate.subspace_after, candidate.ops, log);
    candidate.width_after_closure = candidate.subspace_after.active_width();

    undo_all(frontier, log);
    return candidate;
}

// Recursion depth is bounded by the number of expanding ops on the deepest
// path, since every recursive call consumes at least one (its chosen
// candidate); that is small for the corpus this search targets (see
// active_width_search.h), so plain recursion is used rather than an
// explicit stack.
SearchOutcome dfs(SearchContext& ctx, SearchFrontier& frontier, const DormantSubspace& subspace,
                  std::vector<uint32_t>& order) {
    if (frontier.executed_count() == ctx.dependence.num_ops()) {
        return SearchOutcome::Success;
    }
    if (ctx.memo.contains(frontier.executed_bits())) {
        return SearchOutcome::Infeasible;
    }
    if (ctx.explored >= ctx.node_budget) {
        return SearchOutcome::Unknown;
    }
    ++ctx.explored;

    // The frontier is always closed on entry (every path here ran
    // run_closure last), so every ready op is expanding by construction;
    // is_expanding is still checked explicitly below rather than assumed,
    // so this loop stays correct even if that invariant is ever weakened.
    //
    // Snapshot the ready set before iterating: simulate_candidate() below
    // mutates and restores frontier's own ready set via execute()/undo(), so
    // iterating frontier.ready() directly here would iterate a std::set
    // while a nested call erases and reinserts the very element the outer
    // iterator is standing on, which is undefined behavior even though the
    // set's contents end up value-equal afterward.
    const std::vector<uint32_t> ready_ops(frontier.ready().begin(), frontier.ready().end());
    std::vector<Candidate> candidates;
    for (uint32_t op : ready_ops) {
        if (!is_expanding(ctx.hir, ctx.hir.ops[op], subspace)) {
            continue;
        }
        if (subspace.active_width() + 1 > ctx.threshold) {
            continue;
        }
        candidates.push_back(simulate_candidate(ctx.hir, frontier, subspace, op));
    }
    if (candidates.empty()) {
        ctx.memo.insert(frontier.executed_bits());
        return SearchOutcome::Infeasible;
    }

    // Heuristic order: try the candidate whose execution plus closure
    // reaches the lowest width first, ties by lower op index. first_op is
    // unique per candidate, so this fully orders the list regardless of
    // sort stability.
    std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) {
        if (a.width_after_closure != b.width_after_closure) {
            return a.width_after_closure < b.width_after_closure;
        }
        return a.first_op < b.first_op;
    });

    for (Candidate& candidate : candidates) {
        std::vector<UndoStep> log;
        log.reserve(candidate.ops.size());
        for (uint32_t op : candidate.ops) {
            log.emplace_back(op, frontier.execute(op));
        }
        order.insert(order.end(), candidate.ops.begin(), candidate.ops.end());

        const SearchOutcome outcome = dfs(ctx, frontier, candidate.subspace_after, order);
        if (outcome == SearchOutcome::Success) {
            return SearchOutcome::Success;
        }

        order.resize(order.size() - candidate.ops.size());
        undo_all(frontier, log);

        if (outcome == SearchOutcome::Unknown) {
            // Giving up partway through a subtree is not the same as
            // proving every candidate infeasible, so this node is not
            // memoized, and there is no reason to try the remaining
            // (already-budget-starved) candidates either.
            return SearchOutcome::Unknown;
        }
    }

    ctx.memo.insert(frontier.executed_bits());
    return SearchOutcome::Infeasible;
}

// Runs the threshold-ctx.threshold feasibility search from a fresh initial
// closure, sharing ctx's node budget and failed-node memo with whatever
// thresholds the outer loop already tried. On success, `witness_order` holds
// a complete linear extension of ctx.dependence.
SearchOutcome feasible(SearchContext& ctx, std::vector<uint32_t>& witness_order) {
    SearchFrontier frontier(ctx.dependence);
    DormantSubspace subspace(ctx.hir.num_qubits);
    witness_order.clear();

    // The root closure is never undone: it is the fixed starting point for
    // this whole feasible() call, not a candidate that might be backed out
    // of, so its undo log is simply discarded once run_closure returns.
    std::vector<UndoStep> discarded_log;
    run_closure(ctx.hir, frontier, subspace, witness_order, discarded_log);

    return dfs(ctx, frontier, subspace, witness_order);
}

}  // namespace

WidthSearchResult search_width_schedule(const HirModule& hir, const ScheduleDependence& dependence,
                                        WidthSearchOptions options) {
    if (dependence.num_ops() != hir.ops.size()) {
        throw std::invalid_argument(
            "search_width_schedule: dependence relation was not built from this HIR's operation "
            "count");
    }

    const ActiveWidthTrace incumbent = analyze_active_width(hir);

    WidthSearchResult result;
    result.incumbent_peak = incumbent.peak_width;
    result.upper_bound = incumbent.peak_width;
    result.lower_bound = incumbent.final_width;
    result.noise_transparent = dependence.noise_transparent();
    result.best_order.resize(hir.ops.size());
    std::iota(result.best_order.begin(), result.best_order.end(), uint32_t{0});

    SearchContext ctx{hir, dependence, options.node_budget, 0, 0, {}};

    // final_width can never exceed peak_width (it is one of the widths the
    // peak maxes over), so lower_bound <= upper_bound holds from the start
    // and upper_bound - 1 below never underflows.
    while (result.upper_bound > result.lower_bound) {
        ctx.threshold = result.upper_bound - 1;
        std::vector<uint32_t> witness_order;
        const SearchOutcome outcome = feasible(ctx, witness_order);

        if (outcome == SearchOutcome::Success) {
            // The threshold only bounds candidates from above, so the
            // witness's actual peak can undercut ctx.threshold; report that
            // achieved value, not the threshold that produced it, and
            // resume searching one below it.
            HirModule witness_hir = hir;
            apply_schedule(witness_hir, dependence, witness_order);
            result.best_order = std::move(witness_order);
            result.upper_bound = analyze_active_width(witness_hir).peak_width;
            continue;
        }
        if (outcome == SearchOutcome::Infeasible) {
            result.lower_bound = ctx.threshold + 1;
            break;
        }
        result.budget_exhausted = true;
        break;
    }

    result.explored_nodes = ctx.explored;
    return result;
}

}  // namespace clifft
