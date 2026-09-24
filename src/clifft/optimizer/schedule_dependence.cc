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

// Check noise transparency first: a type check avoids walking noise-channel
// masks when sign correction already makes the crossing legal.
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

// Sort by key and value so each CSR adjacency list is ordered.
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
    dep.num_ops_ = n;
    std::vector<bool> movable(n);
    for (size_t i = 0; i < n; ++i) {
        movable[i] = is_movable_op(hir.ops[i].op_type());
    }

    std::vector<std::pair<uint32_t, uint32_t>> edges;

    if (n > 0) {
        // Cache ancestor sets in a ring. Evicted rows only cost extra checks
        // and redundant edges; the constraints remain the same.
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

            // Known ancestors of i also precede j, so they need no separate test.
            // Only read cached rows that have not been evicted.
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

            // Fixed ops retain their relative order even when they commute.
            // Keep consecutive fixed ops directly linked for adjacency queries.
            if (!movable[j]) {
                if (previous_fixed.has_value()) {
                    link(*previous_fixed);
                }
                previous_fixed = static_cast<uint32_t>(j);
            }

            if (j == 0) {
                continue;
            }

            // Try nearby predecessors first: their ancestor sets can rule out
            // many earlier candidates at once.
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
                    if (!movable[i] && !movable[j]) {
                        continue;  // both fixed: already ordered by the chain above
                    }
                    if (!allowed(hir, hir.ops[i], hir.ops[j], dep.noise_transparent_)) {
                        link(i);
                        // Skip ancestors just absorbed by link().
                        mask &= ~rj[w];
                    }
                }
            }
        }
    }

    group_into_csr(n, edges, dep.succ_offsets_, dep.succ_indices_);

    std::vector<std::pair<uint32_t, uint32_t>> reversed_edges;
    reversed_edges.reserve(edges.size());
    for (const auto& [from, to] : edges) {
        reversed_edges.emplace_back(to, from);
    }
    group_into_csr(n, std::move(reversed_edges), dep.pred_offsets_, dep.pred_indices_);

    return dep;
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
    assert(hir.ops.size() == dependence.num_ops());
    if (!dependence.is_linear_extension(order)) {
        throw std::invalid_argument(
            "apply_schedule: order is not a linear extension of the dependence relation");
    }

    if (dependence.noise_transparent()) {
        hir.materialize_logical_noise_prefix();
    }

    hir.permute_ops(order);
}

}  // namespace clifft::detail
