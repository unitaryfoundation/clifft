#pragma once

// Conservative scheduling constraints over the original HIR op indices.
// Edges point forward in the original order; any topological order is legal.
//
// Only T_GATE, PHASE_ROTATION, and MEASURE may move. Other ops retain their
// relative order, preserving noise-site order and classical dependencies.
// can_swap supplies the pairwise constraints, including EXP_VAL and INSTRUMENT
// barriers. Noise transparency additionally permits movable ops to cross NOISE:
// logical noise positions let the planner correct signs for each noise realization.
//
// The DAG omits edges already implied by a path, preserving the legal orders.
// For two adjacent ops in a legal order, any ordering constraint must be a
// direct edge: an intermediate path node would have to sit between them.
// Schedulers can therefore check adjacent swaps without walking the graph.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/commutation.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::detail {

struct ScheduleDependenceOptions {
    // Allow noise crossings using the planner's logical-position sign correction.
    bool noise_transparent = true;

    // Target budget for cached ancestor bitsets; at least one row is retained.
    // Smaller caches can leave redundant edges and require more pairwise checks,
    // but do not change the legal orders. This does not bound the DAG's size.
    size_t ancestor_cache_bytes = 64u << 20;
};

class ScheduleDependence {
  public:
    // Build constraints for this HIR, omitting edges implied by known ancestors.
    [[nodiscard]] static ScheduleDependence build(const HirModule& hir,
                                                  ScheduleDependenceOptions options = {});

    [[nodiscard]] size_t num_ops() const { return movable_.size(); }

    // Direct predecessors, sorted by original op index.
    [[nodiscard]] std::span<const uint32_t> predecessors(size_t op) const;

    // Direct successors, sorted by original op index.
    [[nodiscard]] std::span<const uint32_t> successors(size_t op) const;

    [[nodiscard]] bool noise_transparent() const { return noise_transparent_; }

    // True if order is a permutation of all op indices respecting every edge.
    [[nodiscard]] bool is_linear_extension(std::span<const uint32_t> order) const;

  private:
    ScheduleDependence() = default;

    bool noise_transparent_ = false;
    std::vector<bool> movable_;

    // CSR offsets have num_ops() + 1 entries; op i's neighbors occupy
    // [offsets[i], offsets[i + 1]) in the corresponding indices vector.
    std::vector<uint32_t> pred_offsets_;
    std::vector<uint32_t> pred_indices_;
    std::vector<uint32_t> succ_offsets_;
    std::vector<uint32_t> succ_indices_;
};

// Applies a legal order to ops and their parallel metadata. Requires the
// unchanged HIR used to build dependence, or an identical copy. Throws
// std::invalid_argument if order is not a linear extension.
//
// Noise-transparent schedules preserve logical noise positions before moving
// ops, so the planner can correct their signs. Ordinary commuting crossings
// need no new metadata; materializing it unnecessarily can inhibit later fusion.
void apply_schedule(HirModule& hir, const ScheduleDependence& dependence,
                    std::span<const uint32_t> order);

}  // namespace clifft::detail
