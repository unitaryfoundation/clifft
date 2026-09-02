#include "active_width_search.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/active_width_closure.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clifft::research {

namespace {

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

// Ready ops in ascending index order, snapshotted before any candidate is
// evaluated: evaluate_candidate below executes and undoes each candidate
// against the shared frontier in turn, so scanning a live view while that
// loop body edits readiness would be unsafe. Mirrors
// active_width_schedule_pass.cc's own ready_ops_snapshot helper; duplicated
// rather than shared because that one is private to a different target and
// this tool intentionally depends on nothing outside clifft_core's public
// headers.
std::vector<uint32_t> ready_ops_snapshot(const detail::SearchFrontier& frontier) {
    std::vector<uint32_t> ready;
    for (uint32_t op = frontier.lowest_ready_hint(); op < frontier.num_ops(); ++op) {
        if (frontier.is_ready(op)) {
            ready.push_back(op);
        }
    }
    return ready;
}

// A scored but not yet committed branch: executing `first_op` (a ready
// expanding op) and closing would append `ops` (first_op followed by its
// closure sweep, in execution order) and reach `width_after_closure`.
// DormantSubspace has no default constructor, so this struct needs one too:
// subspace_after always starts as a copy of the parent node's subspace,
// never a fresh empty one.
struct Candidate {
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
Candidate evaluate_candidate(const HirModule& hir, detail::SearchFrontier& frontier,
                             const DormantSubspace& subspace, uint32_t first_op) {
    Candidate candidate(subspace);
    candidate.first_op = first_op;
    std::vector<detail::UndoStep> log;
    std::vector<uint32_t> newly_ready_log;

    log.push_back(detail::UndoStep{first_op, frontier.execute(first_op, newly_ready_log)});
    candidate.ops.push_back(first_op);
    const WidthTransition first_transition =
        classify_and_apply(hir, hir.ops[first_op], candidate.subspace_after);
    assert(is_expanding_effect(first_transition.effect) &&
           "evaluate_candidate called on an op the caller did not classify expanding");

    detail::run_closure(hir, frontier, candidate.subspace_after, candidate.ops, log,
                        newly_ready_log);
    candidate.width_after_closure = candidate.subspace_after.active_width();

    detail::undo_all(frontier, log, newly_ready_log);
    return candidate;
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

// Recursion depth is bounded by the number of expanding ops on the deepest
// path, since every recursive call consumes at least one (its chosen
// candidate); that stays small for the corpus this tool targets, so plain
// recursion is used rather than an explicit stack.
SearchOutcome dfs(SearchContext& ctx, detail::SearchFrontier& frontier,
                  const DormantSubspace& subspace, std::vector<uint32_t>& order) {
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
    std::vector<Candidate> candidates;
    for (uint32_t op : ready_ops_snapshot(frontier)) {
        if (!detail::is_expanding(ctx.hir, ctx.hir.ops[op], subspace)) {
            continue;
        }
        if (subspace.active_width() + 1 > ctx.threshold) {
            continue;
        }
        candidates.push_back(evaluate_candidate(ctx.hir, frontier, subspace, op));
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
        std::vector<detail::UndoStep> log;
        std::vector<uint32_t> newly_ready_log;
        log.reserve(candidate.ops.size());
        for (uint32_t op : candidate.ops) {
            log.push_back(detail::UndoStep{op, frontier.execute(op, newly_ready_log)});
        }
        order.insert(order.end(), candidate.ops.begin(), candidate.ops.end());

        const SearchOutcome outcome = dfs(ctx, frontier, candidate.subspace_after, order);
        if (outcome == SearchOutcome::Success) {
            return SearchOutcome::Success;
        }

        order.resize(order.size() - candidate.ops.size());
        detail::undo_all(frontier, log, newly_ready_log);

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
    detail::SearchFrontier frontier(ctx.dependence);
    DormantSubspace subspace(ctx.hir.num_qubits);
    witness_order.clear();

    // The root closure is never undone: it is the fixed starting point for
    // this whole feasible() call, not a candidate that might be backed out
    // of, so its undo log is simply discarded once run_closure returns.
    std::vector<detail::UndoStep> discarded_log;
    std::vector<uint32_t> discarded_newly_ready;
    detail::run_closure(ctx.hir, frontier, subspace, witness_order, discarded_log,
                        discarded_newly_ready);

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

}  // namespace clifft::research
