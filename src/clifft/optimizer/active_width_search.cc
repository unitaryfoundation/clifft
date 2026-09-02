#include "clifft/optimizer/active_width_search.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_closure.h"
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

// Closure and readiness bookkeeping (is_expanding, apply_op, SearchFrontier,
// UndoStep, undo_all, find_ready_non_expanding, run_closure) live in
// active_width_closure.h, shared with active_width_schedule_pass.cc's beam
// search so the two cannot disagree on what counts as ready or expanding.
using detail::apply_op;
using detail::find_ready_non_expanding;
using detail::is_expanding;
using detail::run_closure;
using detail::SearchFrontier;
using detail::undo_all;
using detail::UndoStep;

// ---------------------------------------------------------------------------
// Exact executed-op bitset memo key. See active_width_search.h for why the
// key must be the literal bitset rather than a hash digest treated as an
// identity.
// ---------------------------------------------------------------------------

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
