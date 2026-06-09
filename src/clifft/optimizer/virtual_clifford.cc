#include "clifft/optimizer/virtual_clifford.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/util/config.h"
#include "clifft/util/constants.h"

#include "stim.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

namespace {

inline void conjugate_pauli_by_S(MaskView x_p, MaskView z_p, bool sign_p, MutableMaskView x_q,
                                 MutableMaskView z_q, bool& sign_q, bool is_dagger) {
    if (!anti_commute(x_p, z_p, x_q, z_q))
        return;

    int phase = 0;
    for (uint32_t w = 0; w < x_p.num_words(); ++w) {
        uint64_t X1 = x_p.words[w];
        uint64_t Z1 = z_p.words[w];
        uint64_t X2 = x_q.words[w];
        uint64_t Z2 = z_q.words[w];

        uint64_t mask_plus = (X1 & ~Z1 & X2 & Z2) | (X1 & Z1 & ~X2 & Z2) | (~X1 & Z1 & X2 & ~Z2);
        uint64_t mask_minus = (X1 & ~Z1 & ~X2 & Z2) | (X1 & Z1 & X2 & ~Z2) | (~X1 & Z1 & X2 & Z2);

        phase += std::popcount(mask_plus);
        phase -= std::popcount(mask_minus);
    }

    int p_mod4 = ((phase % 4) + 4) % 4;
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
                                bool sign_v, bool is_dagger, const std::vector<uint8_t>& deleted) {
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

}  // namespace clifft
