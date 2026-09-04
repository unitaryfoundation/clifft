#include "clifft/optimizer/schedule_dependence.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace clifft::detail {

namespace {

bool is_movable_op(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
}

// True when no edge is needed between the ops at i < j: either noise
// transparency specifically waives it for a NOISE-versus-movable pair, or
// the pair passes the ordinary commutation test. See the file comment for
// why the waiver is sound. The waiver is checked first because it is a
// plain op-type comparison, while can_swap on a NOISE op walks every
// channel at that site regardless of outcome -- work worth skipping
// whenever the waiver alone already settles the answer.
bool allowed(const HirModule& hir, const HeisenbergOp& left, const HeisenbergOp& right,
             bool noise_transparent) {
    if (noise_transparent) {
        const bool left_noise = left.op_type() == OpType::NOISE;
        const bool right_noise = right.op_type() == OpType::NOISE;
        const bool left_movable = is_movable_op(left.op_type());
        const bool right_movable = is_movable_op(right.op_type());
        if ((left_noise && right_movable) || (right_noise && left_movable)) {
            return true;
        }
    }
    return can_swap(left, right, hir);
}

// Groups (key, value) pairs into CSR form: offsets[k]..offsets[k + 1]
// indexes into `indices` for every value recorded under key k. std::pair's
// default order is lexicographic, so sorting by (key, value) both groups
// equal keys together and, within each group, leaves values ascending.
void group_into_csr(size_t num_keys, std::vector<std::pair<uint32_t, uint32_t>> pairs,
                    std::vector<uint32_t>& offsets, std::vector<uint32_t>& indices) {
    std::ranges::sort(pairs);
    offsets.assign(num_keys + 1, 0);
    for (const auto& entry : pairs) {
        ++offsets[entry.first + 1];
    }
    for (size_t i = 0; i < num_keys; ++i) {
        offsets[i + 1] += offsets[i];
    }
    indices.clear();
    indices.reserve(pairs.size());
    for (const auto& entry : pairs) {
        indices.push_back(entry.second);
    }
}

}  // namespace

ScheduleDependence ScheduleDependence::build(const HirModule& hir,
                                             ScheduleDependenceOptions options) {
    const size_t n = hir.ops.size();
    ScheduleDependence dep;
    dep.noise_transparent_ = options.noise_transparent;
    dep.fingerprint_ = commutation_fingerprint(hir);
    dep.movable_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        dep.movable_[i] = is_movable_op(hir.ops[i].op_type());
    }

    // (from, to) edges, gathered here and only grouped into per-op
    // adjacency (by group_into_csr) once collection finishes, since the
    // scan below produces them grouped by "to" (ascending "from" within
    // each group) while successors needs the opposite grouping too.
    std::vector<std::pair<uint32_t, uint32_t>> edges;

    if (n > 0) {
        // One ancestor bitset per op, `words` 64-bit words wide (n bits,
        // rounded up), kept in a ring buffer of `ring_rows` rows: op k's
        // row lives at slot k % ring_rows, and stays valid to read only
        // while j - k < ring_rows for the op j currently being processed.
        // See ScheduleDependenceOptions::ancestor_cache_bytes for how
        // ring_rows is chosen and why a small ring only costs extra work,
        // never a wrong answer.
        const size_t words = (n + 63) / 64;
        const size_t row_bytes = words * sizeof(uint64_t);
        const size_t rows_in_budget = options.ancestor_cache_bytes / row_bytes;
        const size_t ring_rows = std::max<size_t>(1, std::min(n, rows_in_budget));

        std::vector<uint64_t> ancestor_rows(ring_rows * words, 0);
        auto row = [&](size_t k) {
            return std::span<uint64_t>(ancestor_rows).subspan((k % ring_rows) * words, words);
        };

        std::optional<uint32_t> previous_fixed;

        for (size_t j = 0; j < n; ++j) {
            const std::span<uint64_t> rj = row(j);
            std::ranges::fill(rj, uint64_t{0});

            // Records edge i -> j, and absorbs i's own known ancestors into
            // rj when i's row has not been evicted from the ring: from
            // then on, every ancestor of i also reads as an ancestor of j,
            // so the scan below never re-tests it.
            auto link = [&](uint32_t i) {
                if (j - i < ring_rows) {
                    const std::span<const uint64_t> ri = row(i);
                    for (size_t w = 0; w < words; ++w) {
                        rj[w] |= ri[w];
                    }
                }
                rj[i / 64] |= (uint64_t{1} << (i % 64));
                edges.emplace_back(i, static_cast<uint32_t>(j));
            };

            // Chains consecutive fixed ops so every fixed op keeps its
            // original relative order, regardless of what allowed() would
            // say about any one pair of them. This edge is always direct,
            // never merely implied, which is what lets callers rely on
            // finding it via a single predecessors()/successors() lookup.
            if (!dep.movable_[j]) {
                if (previous_fixed.has_value()) {
                    link(*previous_fixed);
                }
                previous_fixed = static_cast<uint32_t>(j);
            }

            if (j == 0) {
                continue;
            }

            // Tests the ops below j not yet known to be its ancestors,
            // nearest first: linking a close predecessor tends to absorb a
            // large ancestor set in one step, ruling out many farther-away
            // candidates before they are ever tested.
            const size_t top_word = (j - 1) / 64;
            for (size_t w = top_word + 1; w-- > 0;) {
                uint64_t mask = ~rj[w];
                if (w == top_word) {
                    const size_t valid_bits = j - w * 64;
                    if (valid_bits < 64) {
                        mask &= (uint64_t{1} << valid_bits) - 1;
                    }
                }
                while (mask != 0) {
                    const int bit = 63 - std::countl_zero(mask);
                    mask &= ~(uint64_t{1} << bit);
                    const auto i = static_cast<uint32_t>(w * 64 + static_cast<size_t>(bit));
                    if (!dep.movable_[i] && !dep.movable_[j]) {
                        continue;  // both fixed: already ordered by the chain above
                    }
                    if (!allowed(hir, hir.ops[i], hir.ops[j], dep.noise_transparent_)) {
                        link(i);
                        // Whatever link() just absorbed into rj needs no
                        // further testing in this word.
                        mask &= ~rj[w];
                    }
                }
            }
        }
    }

    // Successors group by "from", ascending "to": exactly (from, to)
    // lexicographic order, which is what `edges` already sorts to.
    group_into_csr(n, edges, dep.succ_offsets_, dep.succ_indices_);

    // Predecessors group by "to", ascending "from": swap each pair first so
    // the same helper's (key, value) = (first, second) grouping applies.
    std::vector<std::pair<uint32_t, uint32_t>> reversed_edges;
    reversed_edges.reserve(edges.size());
    for (const auto& [from, to] : edges) {
        reversed_edges.emplace_back(to, from);
    }
    group_into_csr(n, std::move(reversed_edges), dep.pred_offsets_, dep.pred_indices_);

    return dep;
}

bool ScheduleDependence::is_movable(size_t op) const {
    assert(op < movable_.size());
    return movable_[op];
}

std::span<const uint32_t> ScheduleDependence::predecessors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(pred_indices_)
        .subspan(pred_offsets_[op], pred_offsets_[op + 1] - pred_offsets_[op]);
}

std::span<const uint32_t> ScheduleDependence::successors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(succ_indices_)
        .subspan(succ_offsets_[op], succ_offsets_[op + 1] - succ_offsets_[op]);
}

bool ScheduleDependence::is_linear_extension(std::span<const uint32_t> order) const {
    const size_t n = num_ops();
    if (order.size() != n) {
        return false;
    }
    // position[op] == n means "not seen yet"; every valid position is < n,
    // so this sentinel also catches an out-of-range entry sharing the slot
    // an in-range duplicate would otherwise collide on.
    std::vector<uint32_t> position(n, static_cast<uint32_t>(n));
    for (size_t pos = 0; pos < order.size(); ++pos) {
        const uint32_t op = order[pos];
        if (op >= n || position[op] != n) {
            return false;
        }
        position[op] = static_cast<uint32_t>(pos);
    }
    for (size_t op = 0; op < n; ++op) {
        for (uint32_t succ : successors(op)) {
            if (position[op] >= position[succ]) {
                return false;
            }
        }
    }
    return true;
}

void apply_schedule(HirModule& hir, const ScheduleDependence& dependence,
                    std::span<const uint32_t> order) {
    if (commutation_fingerprint(hir) != dependence.fingerprint()) {
        throw std::invalid_argument(
            "apply_schedule: target HIR's fingerprint does not match the HIR the dependence "
            "relation was built from");
    }
    if (!dependence.is_linear_extension(order)) {
        throw std::invalid_argument(
            "apply_schedule: order is not a linear extension of the dependence relation");
    }

    if (dependence.noise_transparent()) {
        hir.materialize_logical_noise_prefix();
    }

    const bool has_source_map = hir.source_map.size() == hir.ops.size();
    const bool has_lnp = hir.has_logical_noise_prefix();

    std::vector<HeisenbergOp> new_ops;
    new_ops.reserve(hir.ops.size());
    std::vector<std::vector<uint32_t>> new_source_map;
    std::vector<uint32_t> new_lnp;
    if (has_source_map) {
        new_source_map.reserve(hir.ops.size());
    }
    if (has_lnp) {
        new_lnp.reserve(hir.ops.size());
    }

    for (uint32_t idx : order) {
        new_ops.push_back(hir.ops[idx]);
        if (has_source_map) {
            new_source_map.push_back(std::move(hir.source_map[idx]));
        }
        if (has_lnp) {
            new_lnp.push_back(hir.logical_noise_prefix[idx]);
        }
    }

    hir.ops = std::move(new_ops);
    if (has_source_map) {
        hir.source_map = std::move(new_source_map);
    }
    if (has_lnp) {
        hir.logical_noise_prefix = std::move(new_lnp);
    }
}

}  // namespace clifft::detail
