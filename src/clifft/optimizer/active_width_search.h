#pragma once

// Exact, budgeted search for the schedule of an HIR that minimizes peak
// active width, over the trace class ScheduleDependence defines.
//
// Definitions, over the dependence DAG (predecessors/successors, ops 0..N-1
// in original index order):
//
//   ready       every predecessor of the op has already executed.
//   expanding   executing the op would raise the active width k: a T_GATE or
//               PHASE_ROTATION whose axis does not commute with every
//               generator of the current DormantSubspace S, or an
//               INSTRUMENT that takes the Activate branch (see
//               active_width_analysis.h's WidthEffect). Every other ready op
//               is non-expanding: any MEASURE, any rotation whose axis
//               commutes with S, and every other op type.
//   closure     starting from a set of executed ops, repeatedly execute the
//               lowest-index ready non-expanding op until none remains.
//
// Closure theorem: some schedule that minimizes peak active width executes
// every ready non-expanding op as soon as it is ready. Sketch: readiness
// only grows as ops execute, a rotation that currently commutes with S stays
// commuting under any op independent of it (S only shrinks along directions
// that anticommute with something already executed), and no non-expanding
// op's execution can make another already-ready non-expanding op expanding,
// so delaying a ready non-expanding op can only ever leave the peak the same
// or larger, never smaller. Consequently a search only has to branch on
// which ready expanding op to execute next; closure fills in the rest
// deterministically. An INSTRUMENT is a positional barrier under
// ScheduleDependence (commutation.cc's can_swap refuses to reorder anything
// across one), so every op before it in original order is transitively its
// predecessor and every op after it is transitively its successor -- when it
// is ready, it is therefore the only ready op, and this search executes it
// regardless of whether it is expanding (an Activate instrument that would
// exceed the current threshold simply fails that search node, the same as
// any other over-threshold candidate).
//
// Confluence: the subspace S reached by executing a given set of ops is the
// same regardless of the order they executed in (independent GF(2) updates
// commute), so a search state is fully determined by its set of executed
// ops, not by the sequence that reached it. That is what makes the
// executed-op bitset below a sound memoization key: two different branches
// that happen to execute the same set of ops are provably the same state,
// including for feasibility purposes, even though this search only ever
// compares bitsets exactly and never tries to prove that fact directly.
//
// Threshold feasibility: for a fixed bound B, a depth-first search from the
// initial closed state where a node's candidates are its ready expanding ops
// whose execution would not push k above B (tried in the order most likely
// to succeed first: lowest resulting k, ties by lower op index), recursing
// on each until one reaches every op executed. A node with no candidate
// left, none of which is ready and within budget, is infeasible; that
// verdict is memoized by the exact executed-op bitset so any other branch
// that reaches the identical set of executed ops is pruned immediately
// rather than re-explored. The memo key must be exact, not a hash digest
// used as an identity: two distinct executed-op sets that happened to hash
// alike would otherwise be treated as one, and if one of them is genuinely
// feasible while the memoized one is not, the collision would misreport a
// feasible bound as infeasible. A node whose subtree is only partly explored
// when the shared node budget runs out reports unknown instead and is never
// memoized as failed, since giving up is not the same as proving
// infeasibility.
//
// Outer loop: starting from the incumbent HIR's own order (peak from
// analyze_active_width), repeatedly ask whether some schedule reaches one
// less than the best peak proven achievable so far. Each yes lowers the
// achievable bound to that schedule's actual peak (which can undercut the
// threshold that was asked for, since the threshold only bounds candidates
// from above); each exhaustive no proves nothing tighter is possible and
// stops the search with matching bounds; running out of budget stops the
// search with the bounds it had already proven. The node budget and the
// failed-node memo are both shared across every threshold this loop tries,
// not reset per threshold, because a set of executed ops proven to have no
// completion within a looser bound also has none within any tighter one --
// the candidate list at a lower threshold is a subset of the candidate list
// at a higher one -- so carrying both forward only ever prunes more, never
// incorrectly.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/schedule_dependence.h"

#include <cstdint>
#include <vector>

namespace clifft {

struct WidthSearchOptions {
    // Cumulative cap on search-tree nodes expanded across the whole outer
    // loop (every threshold attempted below the incumbent peak shares this
    // one budget; it is not reset per threshold). A memo hit does not count
    // as an expansion, since it does no new work.
    uint64_t node_budget = 200000;
};

// The result of a bounded exact search for the minimum peak active width of
// `hir` over the legal schedules of a given ScheduleDependence.
//
// Certificate scope: `optimal()` certifies a minimum over the trace class of
// this exact HIR under the dependence relation it was searched with --
// i.e., over every linear extension of that relation, scored by the
// sampling planner's own structural width model (analyze_active_width /
// DormantSubspace). It says nothing about schedules reachable only through
// a rewrite that exposes new gate fusion, about amplitude-level stabilizers
// the structural planner does not track, or about any other circuit that
// merely samples the same distribution as this one.
struct WidthSearchResult {
    // A linear extension of the dependence relation the caller passed to
    // search_width_schedule(), applicable via apply_schedule(). Records the
    // full order the winning search node executed ops in: the initial
    // closure, then each chosen expanding op immediately followed by the
    // non-expanding ops its execution made ready. Equals the identity order
    // (0, 1, ..., num_ops() - 1) when no schedule bettered the incumbent.
    std::vector<uint32_t> best_order;

    // Peak active width of `hir` in its order as given (analyze_active_width
    // on the unmodified HIR), before this search looked for anything better.
    uint32_t incumbent_peak = 0;

    // Best proven-achievable peak: incumbent_peak, or a witness schedule's
    // actual peak if the search found and confirmed one lower.
    uint32_t upper_bound = 0;

    // Best proven lower bound on the peak of any legal schedule: starts at
    // the order-invariant final active width (every legal schedule ends
    // with the same DormantSubspace, so its width is a valid floor on the
    // peak) and rises to threshold + 1 whenever threshold is proven
    // exhaustively infeasible.
    uint32_t lower_bound = 0;

    // Total search-tree nodes expanded across every threshold this search
    // tried, sharing WidthSearchOptions::node_budget.
    uint64_t explored_nodes = 0;

    // True when the node budget ran out before the last threshold attempted
    // was resolved either way. lower_bound is then whatever it was already
    // proven to be (possibly still its initial value); upper_bound may still
    // improve with a larger budget.
    bool budget_exhausted = false;

    // Copied from the ScheduleDependence this search ran against, so a
    // caller holding only the result can tell which relation the bounds and
    // certificate above apply to.
    bool noise_transparent = false;

    // True exactly when lower_bound == upper_bound: the achievable peak is
    // proven minimal over the certificate scope documented above.
    [[nodiscard]] bool optimal() const { return lower_bound == upper_bound; }
};

// Searches for a legal schedule of `hir` (per `dependence`, which must have
// been built from an HIR with the same operation count) that minimizes peak
// active width, budgeted by `options.node_budget`. Never mutates `hir`; a
// caller that wants the improvement applies best_order itself via
// apply_schedule(). Throws std::invalid_argument if `dependence` was not
// built from an HIR with hir.ops.size() operations.
[[nodiscard]] WidthSearchResult search_width_schedule(const HirModule& hir,
                                                      const ScheduleDependence& dependence,
                                                      WidthSearchOptions options = {});

}  // namespace clifft
