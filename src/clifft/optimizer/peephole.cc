#include "clifft/optimizer/peephole.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// =========================================================================
// Symplectic conjugation: absorb a virtual S gate into downstream ops
// =========================================================================

/// Conjugate the stored Pauli Q by S_P in place.
///
/// For S gate (is_dagger=false), c_phase=3 computes S^dag Q S.
/// For S_dag gate (is_dagger=true), c_phase=1 computes S Q S^dag.
///
/// The symplectic product of two single-qubit Paulis A_i, B_i contributes
/// a phase i^{+1} when A*B advances cyclically (X->Y->Z->X) and i^{-1}
/// when it retreats. The total phase from the commutator [P, Q] is
/// i^{sum of per-qubit contributions}.
///
/// `p` must not view the mask being conjugated. `product` is scratch storage.
void conjugate_pauli_by_S(PauliMaskView p, PauliMaskArena& masks, PauliMaskHandle q_handle,
                          bool is_dagger, std::vector<PauliWord>& product) {
    const PauliMaskView q = masks.at(q_handle);
    if (!anti_commute(p, q))
        return;

    // Compute per-qubit phase contributions from the Pauli product P*Q.
    // Each qubit pair (A_i, B_i) contributes +1 or -1 to the total phase
    // exponent based on the cyclic ordering X->Y->Z->X. Only words stored
    // by both Paulis can contribute.
    int phase = 0;
    for_each_common_word(p, q, [&](const PauliWord& p_word, const PauliWord& q_word) {
        uint64_t X1 = p_word.x;
        uint64_t Z1 = p_word.z;
        uint64_t X2 = q_word.x;
        uint64_t Z2 = q_word.z;

        // +1 phase contributions: XY->Z, YZ->X, ZX->Y
        uint64_t mask_plus = (X1 & ~Z1 & X2 & Z2) | (X1 & Z1 & ~X2 & Z2) | (~X1 & Z1 & X2 & ~Z2);
        // -1 phase contributions: YX->Z, ZY->X, XZ->Y
        uint64_t mask_minus = (X1 & ~Z1 & ~X2 & Z2) | (X1 & Z1 & X2 & ~Z2) | (~X1 & Z1 & X2 & Z2);

        phase += std::popcount(mask_plus);
        phase -= std::popcount(mask_minus);
    });

    int p_mod4 = ((phase % 4) + 4) % 4;
    // S (is_dagger=false): downstream we want S^dag Q S = +i P Q -> c_phase=1
    // S_dag (is_dagger=true): downstream we want S Q S^dag = -i P Q -> c_phase=3
    int c_phase = is_dagger ? 3 : 1;

    int total_phase = (p_mod4 + c_phase) % 4;
    if (p.sign())
        total_phase = (total_phase + 2) % 4;
    if (q.sign())
        total_phase = (total_phase + 2) % 4;

    product.clear();
    append_body_product(p, q, product);
    masks.assign(q_handle, PauliMaskView{product, total_phase == 2});
}

/// Absorb a virtual S gate on the Pauli `axis` into all downstream HIR
/// operations and the final tableau.
void apply_virtual_s_downstream(HirModule& hir, size_t start_idx, PauliMaskView axis,
                                bool is_dagger, const std::vector<uint8_t>& deleted) {
    std::vector<PauliWord> product;

    // 1. Conjugate all downstream ops
    for (size_t k = start_idx; k < hir.ops.size(); ++k) {
        if (deleted[k])
            continue;
        auto& op = hir.ops[k];

        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL:
                conjugate_pauli_by_S(axis, hir.pauli_masks, op.mask_handle(), is_dagger, product);
                break;

            case OpType::INSTRUMENT: {
                conjugate_pauli_by_S(axis, hir.pauli_masks, op.mask_handle(), is_dagger, product);
                // The site's destination flip rides in the arena behind a
                // side-table handle; it must track the same frame change
                // as the op's own mask.
                const auto& site =
                    hir.instrument_sites[static_cast<uint32_t>(op.instrument_site_idx())];
                conjugate_pauli_by_S(axis, hir.pauli_masks, site.destination_flip_mask, is_dagger,
                                     product);
                break;
            }

            case OpType::NOISE: {
                auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                for (auto& ch : hir.noise_sites[site_idx].channels) {
                    conjugate_pauli_by_S(axis, hir.noise_channel_masks, ch.mask, is_dagger,
                                         product);
                    // Channel signs are irrelevant and stay cleared.
                    hir.noise_channel_masks.set_sign(ch.mask, false);
                }
                break;
            }

            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    // 2. Final Tableau: U_C' = U_C S (requires inverted dagger flag)
    if (hir.final_tableau.has_value()) {
        hir.final_tableau->prepend_pauli_rotation(
            optimizer_detail::copy_axis(axis, hir.num_qubits).view(), is_dagger);
    }
}

/// Absorb a virtual Pauli into downstream signs and the final Clifford frame.
void apply_virtual_pauli_downstream(HirModule& hir, size_t start_idx, PauliMaskView axis,
                                    const std::vector<uint8_t>& deleted) {
    auto conjugate_mask = [&](PauliMaskHandle handle) {
        const PauliMaskView mask = hir.pauli_masks.at(handle);
        if (anti_commute(axis, mask)) {
            hir.pauli_masks.set_sign(handle, !mask.sign());
        }
    };

    for (size_t k = start_idx; k < hir.ops.size(); ++k) {
        if (deleted[k]) {
            continue;
        }
        auto& op = hir.ops[k];
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL:
                conjugate_mask(op.mask_handle());
                break;

            case OpType::INSTRUMENT: {
                conjugate_mask(op.mask_handle());
                const auto& site =
                    hir.instrument_sites[static_cast<uint32_t>(op.instrument_site_idx())];
                conjugate_mask(site.destination_flip_mask);
                break;
            }

            // Pauli-channel signs are physically irrelevant, and these ops
            // carry no coherent Pauli axis to conjugate.
            case OpType::NOISE:
            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }

    if (hir.final_tableau.has_value()) {
        PauliString unsigned_axis = optimizer_detail::copy_axis(axis, hir.num_qubits);
        unsigned_axis.set_sign(false);
        hir.final_tableau->prepend_pauli(unsigned_axis.view());
    }
}

// =========================================================================
// Peephole helpers
// =========================================================================

/// Normalize a T gate to a positive Pauli sign. Reversing its direction can
/// change only the omitted global phase, so no scalar correction is needed.
inline void normalize_t_sign(HirModule& hir, HeisenbergOp& op) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

/// Check whether op_j blocks op_i from commuting past it.
/// Returns true if op_i is blocked (anti-commutes with op_j).
inline bool is_blocked(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    const PauliMaskView axis = hir.mask_view(op_i);
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(axis, hir.mask_view(op_j));

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            const auto& channels = hir.noise_sites[site_idx].channels;
            return std::ranges::any_of(channels, [&](const NoiseChannel& ch) {
                return anti_commute(axis, hir.noise_channel_masks.at(ch.mask));
            });
        }

        case OpType::EXP_VAL:
            return true;

        // Positional barrier: nothing commutes past an instrument site
        // regardless of mask commutation (see can_swap).
        case OpType::INSTRUMENT:
            return true;

        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::READOUT_NOISE:
            return false;

        default:
            return true;
    }
}

/// Delete phase operations whose only remaining quantum effect is a phase in
/// the eigenspaces of a later measurement on the same unsigned Pauli axis.
///
/// Pauli noise is transparent for this purpose even when it anti-commutes
/// with the phase axis: branch by branch it only reverses the rotation
/// direction, and the matching measurement discards either direction. Other
/// phase operations and measurements may be crossed only when they commute
/// with the candidate. A conditional Pauli is transparent only when its
/// quantum support is disjoint and the ordinary quantum/classical dataflow
/// proof permits a swap. Classical record consumers are inspected but never
/// moved. All other operations are conservative barriers.
size_t eliminate_terminal_measurement_phases(HirModule& hir, std::vector<uint8_t>& deleted) {
    size_t eliminated = 0;

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (deleted[i])
            continue;
        const OpType candidate_type = hir.ops[i].op_type();
        if (candidate_type != OpType::T_GATE && candidate_type != OpType::PHASE_ROTATION)
            continue;

        const PauliMaskView candidate = hir.mask_view(hir.ops[i]);

        for (size_t j = i + 1; j < hir.ops.size(); ++j) {
            if (deleted[j])
                continue;
            const HeisenbergOp& next = hir.ops[j];

            // Terminal phases may cross every Pauli-noise branch. Conditional
            // Paulis require a stronger proof: ordinary commutation/dataflow
            // safety plus disjoint quantum support. The disjointness keeps
            // overlapping feedback conservative even when its total Pauli
            // happens to commute symplectically with the candidate.
            if (next.op_type() == OpType::NOISE) {
                continue;
            }

            if (next.op_type() == OpType::CONDITIONAL_PAULI) {
                if (!supports_overlap(candidate, hir.mask_view(next)) &&
                    can_swap(hir.ops[i], next, hir)) {
                    continue;
                }
                break;
            }

            if (next.op_type() == OpType::MEASURE && same_body(hir.mask_view(next), candidate)) {
                deleted[i] = true;
                ++eliminated;
                break;
            }

            if (!is_blocked(hir.ops[i], next, hir)) {
                continue;
            }
            break;
        }
    }

    return eliminated;
}

}  // namespace

void PeepholeFusionPass::run(HirModule& hir) {
    cancellations_ = 0;
    fusions_ = 0;

    // Fusion changes the Pauli frame. After a noise crossing, that would
    // invalidate the masks used to correct the operation's noise sign.
    if (!hir.logical_noise_prefix_matches_schedule()) {
        return;
    }

    bool has_source_map = hir.source_map.size() == hir.ops.size();
    // The current order already contains the same noise-position information.
    hir.logical_noise_prefix.clear();

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
            // Normalization left the axis unsigned and no later step in this
            // scan reassigns this op's mask, so the view stays valid.
            const PauliMaskView axis_i = hir.mask_view(hir.ops[i]);

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                // Normalize candidate T gates before comparison so that
                // sign-induced dagger flips are accounted for in effective_angle.
                if (hir.ops[j].op_type() == OpType::T_GATE)
                    normalize_t_sign(hir, hir.ops[j]);

                // Re-fetch op_i in case anything mutated it (it didn't, but
                // be explicit since views must remain valid).
                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                // Check if op_j is a matching T gate on the same axis.
                // After normalization both ops have sign=false, so
                // effective_angle depends only on the dagger flag.
                if (op_j.op_type() == OpType::T_GATE && same_body(hir.mask_view(op_j), axis_i)) {
                    int dir_i = op_i.is_dagger() ? -1 : 1;
                    int dir_j = op_j.is_dagger() ? -1 : 1;
                    int total = dir_i + dir_j;

                    if (total == 0) {
                        // T + T_dag = identity cancellation
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else {
                        // |total| == 2: fuse into S, absorb downstream
                        bool s_is_dagger = (total == -2);
                        deleted[i] = true;
                        deleted[j] = true;

                        apply_virtual_s_downstream(hir, j + 1, axis_i, s_is_dagger, deleted);
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

            // The unsigned axis; no step in this scan reassigns this op's mask.
            const PauliMaskView axis_i{hir.mask_view(hir.ops[i]).words(), false};

            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j])
                    continue;

                const auto& op_i = hir.ops[i];
                const auto& op_j = hir.ops[j];

                if (op_j.op_type() == OpType::PHASE_ROTATION &&
                    same_body(hir.mask_view(op_j), axis_i)) {
                    // Compute fused angle accounting for sign bits.
                    double alpha_i = op_i.alpha() * (hir.sign(op_i) ? -1.0 : 1.0);
                    double alpha_j = op_j.alpha() * (hir.sign(op_j) ? -1.0 : 1.0);
                    double fused = alpha_i + alpha_j;

                    // Normalize to [0, 2) relative phase range
                    fused = fused - 2.0 * std::floor(fused / 2.0);

                    const auto clifford = classify_clifford_rotation(fused);
                    if (clifford == CliffordRotation::IDENTITY) {
                        deleted[i] = true;
                        deleted[j] = true;
                        ++cancellations_;
                    } else if (clifford == CliffordRotation::SQRT) {
                        // S gate: absorb downstream.
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, axis_i, false, deleted);
                        ++fusions_;
                    } else if (clifford == CliffordRotation::SQRT_DAG) {
                        // S_dag gate: absorb downstream.
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_s_downstream(hir, j + 1, axis_i, true, deleted);
                        ++fusions_;
                    } else if (clifford == CliffordRotation::PAULI) {
                        // A Pauli rotation is Clifford up to global phase.
                        deleted[i] = true;
                        deleted[j] = true;
                        apply_virtual_pauli_downstream(hir, j + 1, axis_i, deleted);
                        ++fusions_;
                    } else if (std::abs(fused - 0.25) < kRotationCanonicalizationTolerance) {
                        hir.demote_to_tgate(hir.ops[i], false);
                        if (has_source_map) {
                            auto& dst = hir.source_map[i];
                            auto& src = hir.source_map[j];
                            dst.insert(dst.end(), src.begin(), src.end());
                        }
                        deleted[j] = true;
                        ++fusions_;
                    } else if (std::abs(fused - 1.75) < kRotationCanonicalizationTolerance) {
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

                if (is_blocked(op_i, op_j, hir)) {
                    break;
                }
            }
        }

        // Demote standalone PHASE_ROTATIONs at Clifford/T angles.
        // S and S_dag angles are absorbed downstream; T and T_dag are
        // demoted to cheaper T_GATE ops; identities are deleted.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::PHASE_ROTATION)
                continue;

            double alpha = hir.ops[i].alpha() * (hir.sign(hir.ops[i]) ? -1.0 : 1.0);
            double a_mod2 = alpha - 2.0 * std::floor(alpha / 2.0);
            // The signed angle already accounts for the axis sign.
            const PauliMaskView axis_i{hir.mask_view(hir.ops[i]).words(), false};

            const auto clifford = classify_clifford_rotation(a_mod2);
            if (clifford == CliffordRotation::IDENTITY) {
                deleted[i] = true;
                ++cancellations_;
                changed = true;
            } else if (clifford == CliffordRotation::SQRT) {
                // S: absorb downstream.
                apply_virtual_s_downstream(hir, i + 1, axis_i, false, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (clifford == CliffordRotation::SQRT_DAG) {
                // S_dag: absorb downstream.
                apply_virtual_s_downstream(hir, i + 1, axis_i, true, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (clifford == CliffordRotation::PAULI) {
                apply_virtual_pauli_downstream(hir, i + 1, axis_i, deleted);
                deleted[i] = true;
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 0.25) < kRotationCanonicalizationTolerance) {
                hir.demote_to_tgate(hir.ops[i], false);
                ++fusions_;
                changed = true;
            } else if (std::abs(a_mod2 - 1.75) < kRotationCanonicalizationTolerance) {
                hir.demote_to_tgate(hir.ops[i], /*dagger=*/true);
                ++fusions_;
                changed = true;
            }
        }

        const size_t terminal_eliminations = eliminate_terminal_measurement_phases(hir, deleted);
        if (terminal_eliminations != 0) {
            cancellations_ += terminal_eliminations;
            changed = true;
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

}  // namespace clifft
