#include "clifft/optimizer/t_gate_block_collection_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>

namespace clifft {

namespace {

[[nodiscard]] bool is_t_gate(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE;
}

[[nodiscard]] bool t_gate_commutes_with_block(const HirModule& hir, size_t candidate, size_t begin,
                                              size_t end) {
    for (size_t i = begin; i < end; ++i) {
        if (!is_t_gate(hir.ops[i])) {
            continue;
        }
        if (anti_commute(hir.destab_mask(hir.ops[candidate]), hir.stab_mask(hir.ops[candidate]),
                         hir.destab_mask(hir.ops[i]), hir.stab_mask(hir.ops[i]))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool can_bubble_left_to(const HirModule& hir, size_t from, size_t to) {
    const HeisenbergOp& candidate = hir.ops[from];
    for (size_t curr = from; curr > to; --curr) {
        if (!can_swap(hir.ops[curr - 1], candidate, hir)) {
            return false;
        }
    }
    return true;
}

void bubble_left_to(HirModule& hir, size_t from, size_t to, bool has_source_map,
                    size_t& adjacent_swaps) {
    for (size_t curr = from; curr > to; --curr) {
        std::swap(hir.ops[curr - 1], hir.ops[curr]);
        if (has_source_map) {
            std::swap(hir.source_map[curr - 1], hir.source_map[curr]);
        }
        ++adjacent_swaps;
    }
}

}  // namespace

void TGateBlockCollectionPass::run(HirModule& hir) {
    blocks_collected_ = 0;
    t_gates_moved_ = 0;
    adjacent_swaps_ = 0;

    const bool has_source_map = hir.source_map.size() == hir.ops.size();

    size_t i = 0;
    while (i < hir.ops.size()) {
        if (!is_t_gate(hir.ops[i])) {
            ++i;
            continue;
        }

        const size_t begin = i;
        size_t end = i + 1;
        while (end < hir.ops.size() && is_t_gate(hir.ops[end]) &&
               t_gate_commutes_with_block(hir, end, begin, end)) {
            ++end;
        }

        bool moved_into_block = false;
        size_t scanned = 0;
        size_t scan = end;
        while (scan < hir.ops.size() && scanned < max_scan_) {
            if (!is_t_gate(hir.ops[scan])) {
                ++scan;
                ++scanned;
                continue;
            }

            if (!t_gate_commutes_with_block(hir, scan, begin, end) ||
                !can_bubble_left_to(hir, scan, end)) {
                break;
            }

            bubble_left_to(hir, scan, end, has_source_map, adjacent_swaps_);
            ++end;
            ++t_gates_moved_;
            moved_into_block = true;
            scan = end;
            scanned = 0;
        }

        if (moved_into_block) {
            ++blocks_collected_;
        }
        i = end;
    }
}

}  // namespace clifft
