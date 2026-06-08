#include "clifft/optimizer/t_fusion.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/util/constants.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace clifft {

void normalize_t_sign(HirModule& hir, HeisenbergOp& op) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        hir.global_weight *= op.is_dagger() ? kExpMinusIPiOver4 : kExpIPiOver4;
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

bool blocks_scan_path(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), hir.destab_mask(op_j),
                                hir.stab_mask(op_j));

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            for (const auto& ch : hir.noise_sites[site_idx].channels) {
                auto cv = hir.noise_channel_masks.at(ch.mask);
                if (anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), cv.x(), cv.z()))
                    return true;
            }
            return false;
        }

        case OpType::EXP_VAL:
            return true;

        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::READOUT_NOISE:
            return false;

        default:
            return true;
    }
}

void fuse_same_axis_t_in_range(HirModule& hir, size_t range_start, size_t& range_end,
                               SameAxisTStats& stats) {
    bool changed = true;
    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<uint8_t> deleted(n, 0);

        for (size_t i = range_start; i < range_end; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::T_GATE)
                continue;

            normalize_t_sign(hir, hir.ops[i]);
            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < range_end; ++j) {
                if (deleted[j])
                    continue;
                if (hir.ops[j].op_type() == OpType::T_GATE)
                    normalize_t_sign(hir, hir.ops[j]);

                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                if (op_j.op_type() == OpType::T_GATE && hir.destab_mask(op_j) == destab_i &&
                    hir.stab_mask(op_j) == stab_i) {
                    int dir_i = op_i.is_dagger() ? -1 : 1;
                    int dir_j = op_j.is_dagger() ? -1 : 1;
                    int total = dir_i + dir_j;

                    deleted[i] = true;
                    deleted[j] = true;
                    stats.t_removed += 2;
                    ++stats.merges;

                    if (total != 0) {
                        bool s_is_dagger = (total == -2);
                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, s_is_dagger,
                                                   deleted);
                    }

                    changed = true;
                    break;
                }

                if (blocks_scan_path(op_i, op_j, hir))
                    break;
            }
        }

        if (changed) {
            compact_deleted_ops(hir, deleted);
            range_end = std::min(range_end, hir.ops.size());
        }
    }
}

void compact_deleted_ops(HirModule& hir, const std::vector<uint8_t>& deleted) {
    size_t idx = 0;
    auto is_deleted = [&](const HeisenbergOp&) { return deleted[idx++] != 0; };

    auto new_end = std::remove_if(hir.ops.begin(), hir.ops.end(), is_deleted);
    hir.ops.erase(new_end, hir.ops.end());

    if (hir.source_map.size() == deleted.size()) {
        idx = 0;
        auto sm_end = std::remove_if(hir.source_map.begin(), hir.source_map.end(),
                                     [&](const auto&) { return deleted[idx++] != 0; });
        hir.source_map.erase(sm_end, hir.source_map.end());
    }
}

}  // namespace clifft
