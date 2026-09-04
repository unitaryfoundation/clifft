#pragma once

// Conservative dependence relation over HIR operations, and the machinery
// to commit one legal reordering back into an HirModule.
//
// The relation is a DAG over op indices 0..N-1 with edges i -> j for i < j.
// It is a chain-reduced subset of the full "must not reorder" pairwise
// relation described below, not a listing of every conflicting pair: build()
// omits an edge (i, j) whenever some other edge (or chain of edges) it
// already recorded forces i before j, since restoring that edge could not
// change which permutations are legal. Two guarantees callers may rely on
// even though edges are missing:
//
//   * Linear extensions. A legal schedule is any topological order of this
//     DAG: a "linear extension" in is_linear_extension()'s sense.
//     apply_schedule() below commits one such order into an HirModule. The
//     reduced DAG and the full pairwise relation share the same transitive
//     closure, so they share the exact same set of linear extensions.
//
//   * Adjacency. If i and j land next to each other in some linear
//     extension and the closure orders one before the other, the edge
//     between them in this DAG is direct, never merely implied by a longer
//     chain. Reason: an omitted edge (i, j) is only omitted because some op
//     k with i < k < j already makes i an ancestor of j, and every linear
//     extension places that same k strictly between i and j -- so whenever
//     i and j really are adjacent, no such k exists, and the direct
//     allowed()-or-not test build() ran for that specific pair is what
//     decided the edge. A caller that only ever asks "can I swap these two
//     ops that are adjacent right now" -- the same one-swap-at-a-time query
//     the squeeze pass already relies on -- can therefore read
//     predecessors()/successors() directly instead of computing the full
//     closure; see sink_neutral_rotations's independent() helper in
//     active_width_schedule_pass.cc for exactly that use.
//
// Because can_swap() (see commutation.h) is symmetric, a pair the closure
// does not order at all (no path either direction) is mutually independent:
// some linear extension puts either one first. Consecutive fixed ops (see
// below) always get a direct chain edge, never merely an implied one, so
// that specific relationship never requires walking a chain to confirm.
//
// The relation is conservative, not exact, because can_swap() itself is
// conservative: it decides purely from each op's own inline Pauli mask (or,
// for NOISE, its noise-site channel masks), and it refuses to cross EXP_VAL
// and INSTRUMENT unconditionally, since both are positional side-table
// references. Two ops that commute for a reason can_swap cannot see (for
// example, two rotations that would act on disjoint qubits after upstream
// cancellation this analysis does not perform) still get an edge here. That
// can only cost a scheduler some legal reorderings, never grant an illegal
// one -- soundness here is inherited entirely from can_swap's own.
//
// Only T_GATE, PHASE_ROTATION, and MEASURE ever move: every other op type
// keeps its original relative order, via edges chaining each consecutive
// pair of fixed ops regardless of what can_swap would say about that pair
// specifically. This mirrors the sampling planner's own requirements
// (sampling/planner.cc): NOISE and INSTRUMENT sites must arrive in circuit
// order or the planner throws, and DETECTOR, OBSERVABLE, READOUT_NOISE, and
// CONDITIONAL_PAULI read classical state whose validity depends on that
// same fixed order.
//
// Noise transparency (ScheduleDependenceOptions::noise_transparent) is an
// optional relaxation on top: with it set, no edge is placed between a
// NOISE op and a movable one solely because their Pauli masks anticommute.
// This is sound because every NOISE channel is presampled -- its symbol's
// value exists before the action stream runs, independent of where in the
// schedule a planner happens to consume it -- and because the sampling
// planner resolves an operation's noise-dependent sign from its logical
// (original-circuit) position rather than its schedule position whenever
// HirModule::logical_noise_prefix records the two as different. For any
// fixed noise realization, a noise-transparent reorder is a legal can_swap
// reorder of the noise-free circuit with that realization's Pauli errors
// absorbed into downstream signs; averaged over the presampled
// distribution it preserves the sampling distribution exactly, not only on
// average.
//
// Cost: build() keeps a running ancestor bitset per op (see its definition
// in the .cc file for the exact scan) so that, while considering op j, it
// can skip every earlier op already known to be j's ancestor and only call
// allowed() on the rest. That makes the scan O(n^2 / 64) word operations to
// find the untested candidates, plus one allowed() call per pair not yet
// implied -- far short of the full O(N^2) pairwise scan an earlier version
// of this file ran, and it records far fewer edges on the same input.
// ScheduleDependenceOptions::ancestor_cache_bytes bounds how much of that
// bitset history build() keeps at once, so memory stays fixed instead of
// growing with N; see that option's comment for what a smaller budget
// costs (never correctness, only some extra allowed() calls and extra, but
// still sound, recorded edges).
//
// This is pass-internal machinery, not part of the public API: it lives in
// namespace clifft::detail and is not exposed to Python.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/commutation.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::detail {

struct ScheduleDependenceOptions {
    // Controls how much freedom the relation grants around NOISE ops. See
    // the file comment for why the relaxation is sound.
    bool noise_transparent = true;

    // Memory budget, in bytes, for the ancestor bitsets build() keeps while
    // scanning (see build()'s comment below). Shrinking this budget can
    // only make build() keep more edges and call allowed() more often --
    // every pair it would otherwise have skipped is still a real ancestor
    // relationship enforced some other way in the DAG, so no budget can
    // add or remove a linear extension. 64 MiB is large enough to hold a
    // full row (one bit per op, rounded up to a 64-bit word) for every op
    // in circuits up to roughly 20000 ops at once, which covers the corpus
    // this pass was built for; past that size the budget bounds memory at
    // a fixed size instead of letting the rows it would need grow with N.
    size_t ancestor_cache_bytes = 64u << 20;
};

// A DAG of "must not reorder" edges over one HIR's op indices, built once
// and queried repeatedly by a scheduler or by apply_schedule() below. See
// the file comment for the relation's exact rules and soundness argument.
class ScheduleDependence {
  public:
    // Builds the relation for `hir`. While scanning op j's candidate
    // predecessors, this skips a pair (i, j) once i is already known to be
    // an ancestor of j through edges this same call already recorded --
    // that ancestor relationship came either from a direct edge to j or
    // from a chain of edges through some op between i and j, so the pair
    // is still enforced even without a direct edge of its own. Every
    // skipped pair is therefore either genuinely unconstrained (allowed()
    // would say so, had it been asked) or already ordered some other way;
    // the DAG's transitive closure, and so its set of linear extensions,
    // comes out identical to the full pairwise relation's. See
    // ScheduleDependenceOptions::ancestor_cache_bytes for the memory/edge
    // tradeoff this scan makes, and the file comment's "Adjacency"
    // paragraph for what callers can assume about pairs that end up next
    // to each other in a schedule.
    [[nodiscard]] static ScheduleDependence build(const HirModule& hir,
                                                  ScheduleDependenceOptions options = {});

    [[nodiscard]] size_t num_ops() const { return movable_.size(); }

    // True for T_GATE, PHASE_ROTATION, and MEASURE; false for every other
    // op type, which the relation pins to its original neighbors.
    [[nodiscard]] bool is_movable(size_t op) const;

    // Ops that must precede `op` in any legal schedule, sorted ascending.
    [[nodiscard]] std::span<const uint32_t> predecessors(size_t op) const;

    // Ops that must follow `op` in any legal schedule, sorted ascending.
    [[nodiscard]] std::span<const uint32_t> successors(size_t op) const;

    [[nodiscard]] bool noise_transparent() const { return noise_transparent_; }

    // True when `order` is a permutation of 0..num_ops()-1 (every op
    // appears exactly once) that places every predecessor before its
    // successor for every edge in the relation. Legal schedules are
    // exactly the orders this accepts.
    [[nodiscard]] bool is_linear_extension(std::span<const uint32_t> order) const;

    // Fingerprint (see commutation.h) of the HIR build() computed this
    // relation from. See apply_schedule()'s comment for how this guards
    // against a mismatched target.
    [[nodiscard]] const CommutationFingerprint& fingerprint() const { return fingerprint_; }

  private:
    ScheduleDependence() = default;

    bool noise_transparent_ = false;
    std::vector<bool> movable_;
    CommutationFingerprint fingerprint_;

    // CSR adjacency: op i's entries occupy indices
    // [offsets[i], offsets[i + 1]) of the matching indices vector, both
    // sized num_ops() + 1.
    std::vector<uint32_t> pred_offsets_;
    std::vector<uint32_t> pred_indices_;
    std::vector<uint32_t> succ_offsets_;
    std::vector<uint32_t> succ_indices_;
};

// Reorders hir.ops (and its parallel side tables) in place to `order`, a
// linear extension of `dependence`. Throws std::invalid_argument if `hir`'s
// fingerprint does not match the HIR `dependence` was built from (which
// subsumes an operation-count mismatch), or if `order` is not a linear
// extension of `dependence`.
//
// ops, source_map (when parallel to ops), and logical_noise_prefix (when
// present) are permuted together so every side-table entry travels with
// the operation it describes -- the same contract StatevectorSqueezePass
// and PeepholeFusionPass already honor for source_map and
// logical_noise_prefix.
//
// When `dependence` is noise-transparent, this calls
// hir.materialize_logical_noise_prefix() before permuting anything. That
// materialization is the mechanism that makes a NOISE-crossing reorder
// sound: it freezes each operation's current (pre-reorder, i.e. logical)
// position as data that then travels with the operation, so the planner
// can still resolve its noise-dependent sign from that original position
// after the move. Without noise transparency, an edge between a NOISE op
// and a movable one is omitted only when plain can_swap allows it, which
// happens only when the movable operation commutes with every channel at
// that site -- a crossing that needs no sign correction at all, since a
// commuting channel contributes nothing to that correction regardless of
// position. Materializing anyway would still leave behind a
// logical_noise_prefix that PeepholeFusionPass reads as inconsistent with
// schedule position (even though the missing correction is semantically
// zero) and so refuses to fuse across, for no benefit. Skipping
// materialization when it is not needed leaves the vector exactly as it
// was -- typically empty -- keeping that pass fully usable afterward.
void apply_schedule(HirModule& hir, const ScheduleDependence& dependence,
                    std::span<const uint32_t> order);

}  // namespace clifft::detail
