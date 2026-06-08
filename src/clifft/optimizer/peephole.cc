#include "clifft/optimizer/peephole.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/t_fusion.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

void PeepholeFusionPass::run(HirModule& hir) {
    cancellations_ = 0;
    fusions_ = 0;

    bool has_source_map = hir.source_map.size() == hir.ops.size();

    bool changed = true;
    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<uint8_t> deleted(n, 0);

        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::T_GATE)
                continue;

            normalize_t_sign(hir, hir.ops[i]);
            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < n; ++j) {
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

                    if (total == 0) {
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        bool s_is_dagger = (total == -2);
                        deleted[i] = true;
                        deleted[j] = true;

                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, s_is_dagger,
                                                   deleted);
                        ++fusions_;
                    }

                    changed = true;
                    break;
                }

                if (blocks_scan_path(op_i, op_j, hir))
                    break;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                if (op_j.op_type() == OpType::PHASE_ROTATION && hir.destab_mask(op_j) == destab_i &&
                    hir.stab_mask(op_j) == stab_i) {
                    double alpha_i = op_i.alpha() * (hir.sign(op_i) ? -1.0 : 1.0);
                    double alpha_j = op_j.alpha() * (hir.sign(op_j) ? -1.0 : 1.0);
                    double fused = alpha_i + alpha_j;

                    fused = fused - 2.0 * std::floor(fused / 2.0);

                    constexpr double kDemoteEps = 1e-12;
                    if (std::abs(fused) < kDemoteEps || std::abs(fused - 2.0) < kDemoteEps) {
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else if (std::abs(fused - 0.5) < kDemoteEps) {
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, false,
                                                   deleted);
                        ++fusions_;
                    } else if (std::abs(fused - 1.5) < kDemoteEps) {
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, destab_i, stab_i, false, true,
                                                   deleted);
                        ++fusions_;
                    } else if (std::abs(fused - 0.25) < kDemoteEps) {
                        hir.demote_to_tgate(hir.ops[i], false);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    } else if (std::abs(fused - 1.75) < kDemoteEps) {
                        hir.demote_to_tgate(hir.ops[i], /*dagger=*/true);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    } else {
                        hir.demote_to_phase_rotation(hir.ops[i], fused);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    }

                    changed = true;
                    break;
                }

                if (blocks_scan_path(op_i, op_j, hir))
                    break;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            double alpha = hir.ops[i].alpha() * (hir.sign(hir.ops[i]) ? -1.0 : 1.0);
            double a_mod2 = alpha - 2.0 * std::floor(alpha / 2.0);

            constexpr double kDemoteEps = 1e-12;
            if (std::abs(a_mod2) < kDemoteEps || std::abs(a_mod2 - 2.0) < kDemoteEps) {
                deleted[i] = true;
                ++cancellations_;
                changed = true;
            } else if (std::abs(a_mod2 - 0.5) < kDemoteEps) {
                apply_virtual_s_downstream(hir, i + 1, hir.destab_mask(hir.ops[i]),
                                           hir.stab_mask(hir.ops[i]), false, false, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 1.5) < kDemoteEps) {
                apply_virtual_s_downstream(hir, i + 1, hir.destab_mask(hir.ops[i]),
                                           hir.stab_mask(hir.ops[i]), false, true, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 0.25) < kDemoteEps) {
                hir.demote_to_tgate(hir.ops[i], false);
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 1.75) < kDemoteEps) {
                hir.demote_to_tgate(hir.ops[i], /*dagger=*/true);
                ++fusions_;
                changed = true;
            }
        }

        if (changed)
            compact_deleted_ops(hir, deleted);
    }
}

}  // namespace clifft
