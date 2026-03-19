#include "ucc/optimizer/peephole.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ucc {

namespace {

// Try to demote a PHASE_ROTATION to a zero-cost Clifford or T gate.
// Returns true if demotion was applied (op is replaced in-place).
// Alpha should already have sign folded in; the relative phase is periodic
// every 2.0 half-turns since the Front-End already extracted the global phase.
inline bool try_demote_rotation(HeisenbergOp& op, double alpha) {
    // Normalize to [0, 2) half-turns (the period of the relative phase)
    double a = alpha - 2.0 * std::floor(alpha / 2.0);

    auto destab = op.destab_mask();
    auto stab = op.stab_mask();

    constexpr double kDemoteEps = 1e-12;
    // 0.5 half-turns = S gate (relative phase i)
    if (std::abs(a - 0.5) < kDemoteEps) {
        op = HeisenbergOp::make_clifford_phase(destab, stab, false, /*dagger=*/false);
        return true;
    }
    // 1.5 half-turns = S_dag gate (relative phase -i)
    if (std::abs(a - 1.5) < kDemoteEps) {
        op = HeisenbergOp::make_clifford_phase(destab, stab, false, /*dagger=*/true);
        return true;
    }
    // 0.25 half-turns = T gate (relative phase e^{i pi/4})
    if (std::abs(a - 0.25) < kDemoteEps) {
        op = HeisenbergOp::make_tgate(destab, stab, false, /*dagger=*/false);
        return true;
    }
    // 1.75 half-turns = T_dag gate (relative phase e^{-i pi/4})
    if (std::abs(a - 1.75) < kDemoteEps) {
        op = HeisenbergOp::make_tgate(destab, stab, false, /*dagger=*/true);
        return true;
    }
    return false;
}

/// Symplectic inner product over BitMask masks.
/// Returns true if the two Pauli strings anti-commute.
inline bool anti_commute(const PauliBitMask& x1, const PauliBitMask& z1, const PauliBitMask& x2,
                         const PauliBitMask& z2) {
    return (((x1 & z2) ^ (z1 & x2)).popcount() & 1) != 0;
}

/// Compute the effective rotation direction for a T gate.
/// T -> +1, T_dag -> -1, negated if the Pauli sign is negative.
inline int effective_angle(const HeisenbergOp& op) {
    return (op.is_dagger() ? -1 : 1) * (op.sign() ? -1 : 1);
}

/// Check whether op_j blocks op_i from commuting past it.
/// Returns true if op_i is blocked (anti-commutes with op_j).
inline bool is_blocked(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::CLIFFORD_PHASE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(op_i.destab_mask(), op_i.stab_mask(), op_j.destab_mask(),
                                op_j.stab_mask());

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            const auto& channels = hir.noise_sites[site_idx].channels;
            for (const auto& ch : channels) {
                if (anti_commute(op_i.destab_mask(), op_i.stab_mask(), ch.destab_mask,
                                 ch.stab_mask)) {
                    return true;
                }
            }
            return false;
        }

        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::READOUT_NOISE:
            return false;

        default:
            return true;
    }
}

}  // namespace

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

            const auto& op_i = hir.ops[i];
            auto destab_i = op_i.destab_mask();
            auto stab_i = op_i.stab_mask();

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_j = hir.ops[j];

                // Check if op_j is a matching T gate on the same axis
                if (op_j.op_type() == OpType::T_GATE && op_j.destab_mask() == destab_i &&
                    op_j.stab_mask() == stab_i) {
                    int total = effective_angle(op_i) + effective_angle(op_j);

                    if (total == 0) {
                        // T + T_dag = identity cancellation
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        // |total| == 2: fuse into S or S_dag
                        hir.ops[i] = HeisenbergOp::make_clifford_phase(
                            destab_i, stab_i, /*sign=*/false, /*dagger=*/(total == -2));
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

                // Not a match -- check if op_j blocks commutation
                if (is_blocked(op_i, op_j, hir)) {
                    break;
                }
            }
        }

        // PHASE_ROTATION fusion: merge same-axis rotations by adding angles.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i])
                continue;
            if (hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            const auto& op_i = hir.ops[i];
            auto destab_i = op_i.destab_mask();
            auto stab_i = op_i.stab_mask();

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_j = hir.ops[j];

                if (op_j.op_type() == OpType::PHASE_ROTATION && op_j.destab_mask() == destab_i &&
                    op_j.stab_mask() == stab_i) {
                    // Compute fused angle accounting for sign bits.
                    double alpha_i = op_i.alpha() * (op_i.sign() ? -1.0 : 1.0);
                    double alpha_j = op_j.alpha() * (op_j.sign() ? -1.0 : 1.0);
                    double fused = alpha_i + alpha_j;

                    // Normalize to [0, 2) relative phase range
                    fused = fused - 2.0 * std::floor(fused / 2.0);

                    constexpr double kDemoteEps = 1e-12;
                    // Check for identity relative phase (angle ~ 0 or 2)
                    if (std::abs(fused) < kDemoteEps || std::abs(fused - 2.0) < kDemoteEps) {
                        // The Front-End already extracted the absolute global phase
                        // unconditionally. We safely delete without touching global_weight.
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        hir.ops[i] = HeisenbergOp::make_phase_rotation(destab_i, stab_i,
                                                                       /*sign=*/false, fused);
                        try_demote_rotation(hir.ops[i], fused);
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

                if (is_blocked(op_i, op_j, hir)) {
                    break;
                }
            }
        }

        // Demote standalone PHASE_ROTATIONs at Clifford/T angles to
        // cheaper gate types (S, S_dag, T, T_dag).
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            double alpha = hir.ops[i].alpha() * (hir.ops[i].sign() ? -1.0 : 1.0);
            double a_mod2 = alpha - 2.0 * std::floor(alpha / 2.0);

            constexpr double kDemoteEps = 1e-12;
            if (std::abs(a_mod2) < kDemoteEps || std::abs(a_mod2 - 2.0) < kDemoteEps) {
                deleted[i] = true;
                ++cancellations_;
                changed = true;
            } else if (try_demote_rotation(hir.ops[i], a_mod2)) {
                changed = true;
                ++fusions_;
            }
        }

        // Compact: remove deleted ops via erase-remove idiom
        if (changed) {
            size_t write = 0;
            for (size_t read = 0; read < n; ++read) {
                if (!deleted[read]) {
                    if (write != read) {
                        hir.ops[write] = hir.ops[read];
                        if (has_source_map) {
                            hir.source_map[write] = std::move(hir.source_map[read]);
                        }
                    }
                    ++write;
                }
            }
            hir.ops.erase(hir.ops.begin() + static_cast<ptrdiff_t>(write), hir.ops.end());
            if (has_source_map) {
                hir.source_map.resize(write);
            }
        }
    }
}

}  // namespace ucc
