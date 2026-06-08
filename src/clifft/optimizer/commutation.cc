#include "clifft/optimizer/commutation.h"

#include "clifft/util/constants.h"

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <span>

namespace clifft {

namespace {

/// Conjugate Pauli Q by S_P in place (see peephole.cc for derivation).
/// c_phase=1 for S (is_dagger=false), c_phase=3 for S_dag (is_dagger=true).
inline void conjugate_pauli_by_S(MaskView x_p, MaskView z_p, bool sign_p, MutableMaskView x_q,
                                 MutableMaskView z_q, bool& sign_q, bool is_dagger) {
    if (!anti_commute(x_p, z_p, x_q, z_q))
        return;

    int p_mod4 = pauli_product_phase_mod4(x_p, z_p, x_q, z_q);
    int c_phase = is_dagger ? 3 : 1;
    int total_phase = (p_mod4 + c_phase) % 4;
    if (sign_p)
        total_phase = (total_phase + 2) % 4;
    if (sign_q)
        total_phase = (total_phase + 2) % 4;

    sign_q = (total_phase == 2);
    x_q.xor_with(x_p);
    z_q.xor_with(z_p);
}

}  // namespace

void apply_virtual_s_downstream(HirModule& hir, size_t start_idx, MaskView x_v, MaskView z_v,
                                 bool sign_v, bool is_dagger,
                                 const std::vector<uint8_t>& deleted) {
    for (size_t k = start_idx; k < hir.ops.size(); ++k) {
        if (deleted[k])
            continue;
        auto& op = hir.ops[k];

        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::EXP_VAL: {
                auto m = hir.mask_at(op);
                bool sign_i = m.sign();
                conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), sign_i, is_dagger);
                m.set_sign(sign_i);
                break;
            }

            case OpType::PHASE_ROTATION: {
                auto m = hir.mask_at(op);
                bool sign_before = m.sign();
                bool sign_i = sign_before;
                conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), sign_i, is_dagger);

                if (sign_i != sign_before) {
                    double corr = op.alpha() * std::numbers::pi * (sign_before ? -1.0 : 1.0);
                    hir.global_weight *= std::complex<double>(std::cos(corr), std::sin(corr));
                }

                m.set_sign(sign_i);
                break;
            }

            case OpType::NOISE: {
                auto site_idx = static_cast<uint32_t>(op.noise_site_idx());
                for (auto& ch : hir.noise_sites[site_idx].channels) {
                    auto m = hir.noise_channel_masks.mut_at(ch.mask);
                    bool dummy_sign = false;
                    conjugate_pauli_by_S(x_v, z_v, sign_v, m.x(), m.z(), dummy_sign, is_dagger);
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

    if (hir.final_tableau.has_value()) {
        stim::Tableau<kStimWidth>& tab = *hir.final_tableau;
        const size_t words = std::min<size_t>((tab.num_qubits + 63) / 64, x_v.num_words());

        stim::PauliString<kStimWidth> p_virt(tab.num_qubits);
        for (size_t w = 0; w < words; ++w) {
            p_virt.xs.u64[w] = x_v.words[w];
            p_virt.zs.u64[w] = z_v.words[w];
        }
        p_virt.sign = sign_v;

        stim::PauliString<kStimWidth> p_phys = tab(p_virt);

        std::vector<uint64_t> px_phys(words, 0);
        std::vector<uint64_t> pz_phys(words, 0);
        for (size_t w = 0; w < words; ++w) {
            px_phys[w] = p_phys.xs.u64[w];
            pz_phys[w] = p_phys.zs.u64[w];
        }
        bool psign_phys = p_phys.sign;
        MaskView px_view{std::span<const uint64_t>(px_phys)};
        MaskView pz_view{std::span<const uint64_t>(pz_phys)};

        std::vector<uint64_t> q_x(words, 0);
        std::vector<uint64_t> q_z(words, 0);
        MutableMaskView qx_view{std::span<uint64_t>(q_x)};
        MutableMaskView qz_view{std::span<uint64_t>(q_z)};

        for (size_t q = 0; q < tab.num_qubits; ++q) {
            auto apply_to_ps = [&](stim::PauliStringRef<kStimWidth> row) {
                for (size_t w = 0; w < words; ++w) {
                    q_x[w] = row.xs.u64[w];
                    q_z[w] = row.zs.u64[w];
                }
                bool q_sign = row.sign;

                conjugate_pauli_by_S(px_view, pz_view, psign_phys, qx_view, qz_view, q_sign,
                                     !is_dagger);

                for (size_t w = 0; w < words; ++w) {
                    row.xs.u64[w] = q_x[w];
                    row.zs.u64[w] = q_z[w];
                }
                row.sign = q_sign;
            };

            apply_to_ps(tab.xs[q]);
            apply_to_ps(tab.zs[q]);
        }
    }
}

namespace {

/// Returns the classical measurement index written by this operation, if any.
std::optional<uint32_t> get_written_meas_idx(const HeisenbergOp& op, const HirModule& hir) {
    if (op.op_type() == OpType::MEASURE) {
        return static_cast<uint32_t>(op.meas_record_idx());
    }
    if (op.op_type() == OpType::READOUT_NOISE) {
        return hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())].meas_idx;
    }
    return std::nullopt;
}

/// Returns true if the operation accesses (reads or writes) the given
/// classical measurement index.
bool accesses_classical_index(const HeisenbergOp& op, uint32_t target_idx, const HirModule& hir) {
    switch (op.op_type()) {
        case OpType::MEASURE:
            return static_cast<uint32_t>(op.meas_record_idx()) == target_idx;
        case OpType::CONDITIONAL_PAULI:
            return static_cast<uint32_t>(op.controlling_meas()) == target_idx;
        case OpType::READOUT_NOISE:
            return hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())].meas_idx ==
                   target_idx;
        case OpType::DETECTOR:
            for (uint32_t idx : hir.detector_targets[static_cast<uint32_t>(op.detector_idx())]) {
                if (idx == target_idx)
                    return true;
            }
            return false;
        case OpType::OBSERVABLE:
            for (uint32_t idx : hir.observable_targets[op.observable_target_list_idx()]) {
                if (idx == target_idx)
                    return true;
            }
            return false;
        default:
            return false;
    }
}

/// Check Pauli commutativity between an op's masks and a noise site's channels.
bool anti_commutes_with_noise(const HeisenbergOp& op, const NoiseSite& site, const HirModule& hir) {
    for (const auto& ch : site.channels) {
        auto ch_view = hir.noise_channel_masks.at(ch.mask);
        if (anti_commute(hir.destab_mask(op), hir.stab_mask(op), ch_view.x(), ch_view.z())) {
            return true;
        }
    }
    return false;
}

/// Check Pauli anti-commutativity between any channel pair of two noise sites.
bool noise_sites_anti_commute(const NoiseSite& a, const NoiseSite& b, const HirModule& hir) {
    for (const auto& ch_a : a.channels) {
        auto va = hir.noise_channel_masks.at(ch_a.mask);
        for (const auto& ch_b : b.channels) {
            auto vb = hir.noise_channel_masks.at(ch_b.mask);
            if (anti_commute(va.x(), va.z(), vb.x(), vb.z())) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir) {
    auto lt = left.op_type();
    auto rt = right.op_type();

    // Precise classical dataflow barrier: prevent swapping if one op writes
    // to a classical measurement index that the other accesses.
    auto left_write = get_written_meas_idx(left, hir);
    if (left_write.has_value() && accesses_classical_index(right, *left_write, hir)) {
        return false;
    }
    auto right_write = get_written_meas_idx(right, hir);
    if (right_write.has_value() && accesses_classical_index(left, *right_write, hir)) {
        return false;
    }

    // EXP_VAL is a positional probe: the user expects the expectation value
    // at an exact circuit point. Never reorder anything across it.
    if (lt == OpType::EXP_VAL || rt == OpType::EXP_VAL) {
        return false;
    }

    // Quantum commutativity via symplectic inner product.
    // Both ops carry inline Pauli masks:
    bool left_is_noise = (lt == OpType::NOISE);
    bool right_is_noise = (rt == OpType::NOISE);

    // NOISE ops carry zero inline Pauli masks; the actual channel content
    // lives in the NoiseSite side-table. Two NOISE ops must be checked
    // via noise_sites_anti_commute (channel-vs-channel), not via inline masks.
    if (left_is_noise && right_is_noise) {
        auto li = static_cast<uint32_t>(left.noise_site_idx());
        auto ri = static_cast<uint32_t>(right.noise_site_idx());
        return !noise_sites_anti_commute(hir.noise_sites[li], hir.noise_sites[ri], hir);
    }

    if (left_is_noise) {
        auto li = static_cast<uint32_t>(left.noise_site_idx());
        return !anti_commutes_with_noise(right, hir.noise_sites[li], hir);
    }

    if (right_is_noise) {
        auto ri = static_cast<uint32_t>(right.noise_site_idx());
        return !anti_commutes_with_noise(left, hir.noise_sites[ri], hir);
    }

    // DETECTOR, OBSERVABLE, READOUT_NOISE have no quantum Pauli footprint
    // (they only read classical data), so they commute with everything
    // that passes the classical/PRNG checks above.
    bool left_classical =
        (lt == OpType::DETECTOR || lt == OpType::OBSERVABLE || lt == OpType::READOUT_NOISE);
    bool right_classical =
        (rt == OpType::DETECTOR || rt == OpType::OBSERVABLE || rt == OpType::READOUT_NOISE);
    if (left_classical || right_classical) {
        return true;
    }

    // Standard Pauli anti-commutation check
    return !anti_commute(hir.destab_mask(left), hir.stab_mask(left), hir.destab_mask(right),
                         hir.stab_mask(right));
}

}  // namespace clifft
