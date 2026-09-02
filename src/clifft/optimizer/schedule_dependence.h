#pragma once

// Conservative dependence relation over HIR operations, and the machinery
// to commit one legal reordering back into an HirModule.
//
// The relation is a DAG over op indices 0..N-1 with edges i -> j for i < j.
// A legal schedule is any topological order of this DAG: a "linear
// extension" in is_linear_extension()'s sense. apply_schedule() below
// commits one such order into an HirModule. Because can_swap() (see
// commutation.h) is symmetric, two ops with no edge between them in either
// direction are mutually independent, so the set of linear extensions is
// exactly the trace class reachable from the original order by adjacent
// transpositions of independent ops. That is the same guarantee the
// squeeze pass already leans on pairwise, one swap at a time; this module
// makes it explicit and total so a scheduler can search over it directly.
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
// Cost: build() makes an O(N^2) can_swap call in the worst case, one per
// pair with at least one movable endpoint. That is acceptable for this
// prototype; a real scheduler would want cheap pruning (for example,
// bounding the pairwise scan by qubit overlap) before running on the
// largest circuits in the corpus.

#include "clifft/frontend/hir.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

// Controls how much freedom the relation grants around NOISE ops. See the
// file comment for why the relaxation is sound.
struct ScheduleDependenceOptions {
    bool noise_transparent = true;
};

// A DAG of "must not reorder" edges over one HIR's op indices, built once
// and queried repeatedly by a scheduler or by apply_schedule() below. See
// the file comment for the relation's exact rules and soundness argument.
class ScheduleDependence {
  public:
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

  private:
    ScheduleDependence() = default;

    bool noise_transparent_ = false;
    std::vector<bool> movable_;

    // CSR adjacency: op i's entries occupy indices
    // [offsets[i], offsets[i + 1]) of the matching indices vector, both
    // sized num_ops() + 1.
    std::vector<uint32_t> pred_offsets_;
    std::vector<uint32_t> pred_indices_;
    std::vector<uint32_t> succ_offsets_;
    std::vector<uint32_t> succ_indices_;
};

// Reorders hir.ops (and its parallel side tables) in place to `order`, a
// linear extension of `dependence`. Throws std::invalid_argument if
// `dependence` was not built from an HIR with the same operation count as
// `hir`, or if `order` is not a linear extension of `dependence`.
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

}  // namespace clifft
