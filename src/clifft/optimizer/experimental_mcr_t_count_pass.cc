#include "clifft/optimizer/experimental_mcr_t_count_pass.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/util/constants.h"

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clifft {

namespace {

// =========================================================================
// Windowed bounded MCR search
// =========================================================================

/// Conjugate Pauli Q by S_P in place.
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

void normalize_t_sign(HirModule& hir, HeisenbergOp& op, std::complex<double>& global_weight) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        global_weight *= op.is_dagger() ? kExpMinusIPiOver4 : kExpIPiOver4;
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

bool is_t_gate_blocked(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir) {
    switch (op_j.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
            return anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), hir.destab_mask(op_j),
                                hir.stab_mask(op_j));

        case OpType::NOISE: {
            auto site_idx = static_cast<uint32_t>(op_j.noise_site_idx());
            const auto& channels = hir.noise_sites[site_idx].channels;
            for (const auto& ch : channels) {
                auto cv = hir.noise_channel_masks.at(ch.mask);
                if (anti_commute(hir.destab_mask(op_i), hir.stab_mask(op_i), cv.x(), cv.z())) {
                    return true;
                }
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

struct AxisKey {
    std::vector<uint64_t> x_words;
    std::vector<uint64_t> z_words;

    bool operator==(const AxisKey& other) const {
        return x_words == other.x_words && z_words == other.z_words;
    }
};

struct AxisKeyHash {
    size_t operator()(const AxisKey& key) const {
        auto mix = [](size_t seed, uint64_t word) {
            seed ^= std::hash<uint64_t>{}(word) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        };

        size_t seed = key.x_words.size();
        for (uint64_t word : key.x_words)
            seed = mix(seed, word);
        for (uint64_t word : key.z_words)
            seed = mix(seed, word);
        return seed;
    }
};

struct McrCandidate {
    size_t a;
    size_t b;
    size_t c;
    size_t d;
    size_t anchor_t_idx;
    size_t b_t_idx;
    size_t c_t_idx;
    size_t d_t_idx;
    size_t window_start;
    size_t window_end;
};

struct WindowInfo {
    size_t start;
    size_t end;
    std::vector<size_t> t_positions;
};

struct LocalFusionStats {
    size_t merges = 0;
    size_t t_removed = 0;
};

struct TGateInfo {
    MaskView x;
    MaskView z;
    bool effective_dagger = false;
};

using SymbolicUnitary = std::unordered_map<AxisKey, std::complex<double>, AxisKeyHash>;
using OriginalSpanSignatureCache = std::unordered_map<uint64_t, SymbolicUnitary>;

constexpr size_t kWindowSpanCap = 64;

struct MergePairHash {
    size_t operator()(const std::pair<size_t, size_t>& pair) const {
        size_t seed = std::hash<size_t>{}(pair.first);
        seed ^=
            std::hash<size_t>{}(pair.second) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};

bool is_window_barrier(const HeisenbergOp& op) {
    switch (op.op_type()) {
        case OpType::T_GATE:
            return false;
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::PHASE_ROTATION:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::EXP_VAL:
        case OpType::NUM_OP_TYPES:
            return true;
    }
    return true;
}

int mul_phase_mod4(MaskView x1, MaskView z1, MaskView x2, MaskView z2) {
    int phase = 0;
    for (uint32_t w = 0; w < x1.num_words(); ++w) {
        uint64_t X1 = x1.words[w];
        uint64_t Z1 = z1.words[w];
        uint64_t X2 = x2.words[w];
        uint64_t Z2 = z2.words[w];

        uint64_t mask_plus = (X1 & ~Z1 & X2 & Z2) | (X1 & Z1 & ~X2 & Z2) | (~X1 & Z1 & X2 & ~Z2);
        uint64_t mask_minus = (X1 & ~Z1 & ~X2 & Z2) | (X1 & Z1 & X2 & ~Z2) | (~X1 & Z1 & X2 & Z2);

        phase += std::popcount(mask_plus);
        phase -= std::popcount(mask_minus);
    }

    return ((phase % 4) + 4) % 4;
}

AxisKey identity_axis_key(size_t words) {
    AxisKey key;
    key.x_words.assign(words, 0);
    key.z_words.assign(words, 0);
    return key;
}

MaskView axis_x_view(const AxisKey& key) {
    return MaskView{std::span<const uint64_t>(key.x_words)};
}

MaskView axis_z_view(const AxisKey& key) {
    return MaskView{std::span<const uint64_t>(key.z_words)};
}

AxisKey xor_axis_key(const AxisKey& lhs, MaskView rhs_x, MaskView rhs_z) {
    AxisKey key;
    key.x_words.resize(lhs.x_words.size());
    key.z_words.resize(lhs.z_words.size());

    for (size_t i = 0; i < lhs.x_words.size(); ++i) {
        key.x_words[i] = lhs.x_words[i] ^ rhs_x.words[i];
        key.z_words[i] = lhs.z_words[i] ^ rhs_z.words[i];
    }
    return key;
}

std::complex<double> phase_factor_mod4(int phase_mod4) {
    switch (phase_mod4 & 3) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
    }
}

bool distinct_axes(const HirModule& hir, const std::array<size_t, 4>& idxs) {
    for (size_t i = 0; i < idxs.size(); ++i) {
        for (size_t j = i + 1; j < idxs.size(); ++j) {
            if (hir.destab_mask(hir.ops[idxs[i]]) == hir.destab_mask(hir.ops[idxs[j]]) &&
                hir.stab_mask(hir.ops[idxs[i]]) == hir.stab_mask(hir.ops[idxs[j]])) {
                return false;
            }
        }
    }
    return true;
}

bool matches_mcr_commutation_pattern(const TGateInfo& a, const TGateInfo& b, const TGateInfo& c,
                                     const TGateInfo& d) {
    bool ab = anti_commute(a.x, a.z, b.x, b.z);
    bool ac = anti_commute(a.x, a.z, c.x, c.z);
    bool ad = anti_commute(a.x, a.z, d.x, d.z);
    bool bc = anti_commute(b.x, b.z, c.x, c.z);
    bool bd = anti_commute(b.x, b.z, d.x, d.z);
    bool cd = anti_commute(c.x, c.z, d.x, d.z);

    return (!ab && !cd && ac && ad && bc && bd) || (!ac && !bd && ab && ad && bc && cd) ||
           (!ad && !bc && ab && ac && bd && cd);
}

template <typename GetGateFn>
SymbolicUnitary expand_t_product_signature(size_t gate_count, GetGateFn&& get_gate) {
    constexpr double kQuarterTurn = std::numbers::pi / 8.0;
    const std::complex<double> identity_coeff{std::cos(kQuarterTurn), 0.0};
    const std::complex<double> t_coeff{0.0, -std::sin(kQuarterTurn)};
    const std::complex<double> t_dag_coeff{0.0, std::sin(kQuarterTurn)};

    const TGateInfo& first_gate = get_gate(0);
    size_t words = first_gate.x.num_words();
    SymbolicUnitary terms;
    terms.emplace(identity_axis_key(words), std::complex<double>{1.0, 0.0});

    for (size_t gate_idx = 0; gate_idx < gate_count; ++gate_idx) {
        const TGateInfo& gate = get_gate(gate_idx);
        std::complex<double> axis_coeff = gate.effective_dagger ? t_dag_coeff : t_coeff;

        SymbolicUnitary next_terms;
        next_terms.reserve(terms.size() * 2);
        for (const auto& [axis, coeff] : terms) {
            next_terms[axis] += coeff * identity_coeff;

            int phase_mod4 = mul_phase_mod4(axis_x_view(axis), axis_z_view(axis), gate.x, gate.z);
            AxisKey next_axis = xor_axis_key(axis, gate.x, gate.z);
            next_terms[next_axis] += coeff * axis_coeff * phase_factor_mod4(phase_mod4);
        }

        terms = std::move(next_terms);
    }

    return terms;
}

bool equal_up_to_global_phase(const SymbolicUnitary& lhs, const SymbolicUnitary& rhs) {
    constexpr double kEps = 1e-9;
    std::complex<double> phase = {0.0, 0.0};
    double best_mag = 0.0;

    for (const auto& [axis, coeff_lhs] : lhs) {
        auto it = rhs.find(axis);
        if (it == rhs.end())
            continue;
        double mag_lhs = std::abs(coeff_lhs);
        double mag_rhs = std::abs(it->second);
        double mag = std::min(mag_lhs, mag_rhs);
        if (mag <= std::max(best_mag, kEps))
            continue;
        phase = coeff_lhs / it->second;
        best_mag = mag;
    }

    if (best_mag <= kEps)
        return false;
    phase /= std::abs(phase);

    auto compare_side = [&](const SymbolicUnitary& a, const SymbolicUnitary& b) {
        for (const auto& [axis, coeff_a] : a) {
            auto it = b.find(axis);
            std::complex<double> coeff_b =
                it == b.end() ? std::complex<double>{0.0, 0.0} : it->second;

            if (std::abs(coeff_a) < kEps && std::abs(coeff_b) < kEps)
                continue;
            if (std::abs(coeff_a) < kEps || std::abs(coeff_b) < kEps)
                return false;

            if (std::abs(coeff_a - phase * coeff_b) > kEps)
                return false;
        }
        return true;
    };

    return compare_side(lhs, rhs) && compare_side(rhs, lhs);
}

size_t swapped_rel_to_t_idx(size_t rel_idx, size_t anchor_t_idx, const McrCandidate& cand) {
    size_t b_rel = cand.b_t_idx - cand.anchor_t_idx;
    size_t c_rel = cand.c_t_idx - cand.anchor_t_idx;
    size_t d_rel = cand.d_t_idx - cand.anchor_t_idx;

    if (rel_idx == 0)
        return cand.c_t_idx;
    if (rel_idx == b_rel)
        return cand.d_t_idx;
    if (rel_idx == c_rel)
        return anchor_t_idx;
    if (rel_idx == d_rel)
        return cand.b_t_idx;
    return anchor_t_idx + rel_idx;
}

uint64_t pack_span_key(size_t anchor_t_idx, size_t d_t_idx) {
    return (static_cast<uint64_t>(anchor_t_idx) << 32) | static_cast<uint64_t>(d_t_idx);
}

std::vector<TGateInfo> build_window_t_gate_infos(const HirModule& hir, const WindowInfo& window) {
    std::vector<TGateInfo> infos;
    infos.reserve(window.t_positions.size());
    for (size_t op_idx : window.t_positions) {
        const auto& op = hir.ops[op_idx];
        infos.push_back(TGateInfo{
            .x = hir.destab_mask(op),
            .z = hir.stab_mask(op),
            .effective_dagger = op.is_dagger() != hir.sign(op),
        });
    }
    return infos;
}

bool exact_span_swap_rewrite_is_valid(std::span<const TGateInfo> gate_infos, size_t anchor_t_idx,
                                      size_t d_t_idx, const McrCandidate& cand,
                                      OriginalSpanSignatureCache& original_span_cache) {
    size_t span_len = d_t_idx - anchor_t_idx + 1;
    uint64_t cache_key = pack_span_key(anchor_t_idx, d_t_idx);
    auto [original_it, inserted] = original_span_cache.try_emplace(cache_key);
    if (inserted) {
        original_it->second = expand_t_product_signature(
            span_len, [&](size_t rel_idx) { return gate_infos[anchor_t_idx + rel_idx]; });
    }

    SymbolicUnitary swapped = expand_t_product_signature(span_len, [&](size_t rel_idx) {
        return gate_infos[swapped_rel_to_t_idx(rel_idx, anchor_t_idx, cand)];
    });
    return equal_up_to_global_phase(original_it->second, swapped);
}

bool is_moved_op(size_t op_idx, const std::array<size_t, 4>& moved_ops) {
    return std::find(moved_ops.begin(), moved_ops.end(), op_idx) != moved_ops.end();
}

std::unordered_set<std::pair<size_t, size_t>, MergePairHash> collect_reachable_merge_pairs(
    const HirModule& hir, const std::vector<size_t>& order,
    const std::array<size_t, 4>& moved_ops) {
    std::unordered_set<std::pair<size_t, size_t>, MergePairHash> pairs;

    for (size_t i = 0; i < order.size(); ++i) {
        const auto& op_i = hir.ops[order[i]];
        MaskView x_i = hir.destab_mask(op_i);
        MaskView z_i = hir.stab_mask(op_i);

        for (size_t j = i + 1; j < order.size(); ++j) {
            const auto& op_j = hir.ops[order[j]];
            if (hir.destab_mask(op_j) == x_i && hir.stab_mask(op_j) == z_i) {
                if (is_moved_op(order[i], moved_ops) || is_moved_op(order[j], moved_ops))
                    pairs.emplace(order[i], order[j]);
                break;
            }
            if (anti_commute(x_i, z_i, hir.destab_mask(op_j), hir.stab_mask(op_j)))
                break;
        }
    }

    return pairs;
}

bool window_swap_has_merge_potential(const HirModule& hir, const WindowInfo& window,
                                     const McrCandidate& cand) {
    std::array<size_t, 4> orig{
        window.t_positions[cand.anchor_t_idx],
        window.t_positions[cand.b_t_idx],
        window.t_positions[cand.c_t_idx],
        window.t_positions[cand.d_t_idx],
    };
    auto original_pairs = collect_reachable_merge_pairs(hir, window.t_positions, orig);

    std::vector<size_t> order = window.t_positions;
    order[cand.anchor_t_idx] = orig[2];
    order[cand.b_t_idx] = orig[3];
    order[cand.c_t_idx] = orig[0];
    order[cand.d_t_idx] = orig[1];

    auto swapped_pairs = collect_reachable_merge_pairs(hir, order, orig);
    for (const auto& pair : swapped_pairs) {
        if (!original_pairs.contains(pair))
            return true;
    }
    return false;
}

std::vector<WindowInfo> collect_windows(HirModule& hir) {
    std::vector<WindowInfo> windows;
    size_t i = 0;
    while (i < hir.ops.size()) {
        while (i < hir.ops.size() && is_window_barrier(hir.ops[i])) {
            ++i;
        }
        if (i >= hir.ops.size())
            break;

        size_t start = i;
        WindowInfo window{.start = start, .end = start};

        while (i < hir.ops.size() && !is_window_barrier(hir.ops[i])) {
            if (hir.ops[i].op_type() == OpType::T_GATE) {
                normalize_t_sign(hir, hir.ops[i], hir.global_weight);
                window.t_positions.push_back(i);
            }
            ++i;
        }
        window.end = i;

        if (window.t_positions.size() >= 4)
            windows.push_back(std::move(window));
    }
    return windows;
}

/// Bound each anchor by both T-count lookahead and raw span in ops.
size_t anchor_horizon_end(const WindowInfo& window, size_t anchor_t_idx, size_t lookahead_cap) {
    size_t end = std::min(window.t_positions.size(), anchor_t_idx + lookahead_cap);
    size_t anchor_pos = window.t_positions[anchor_t_idx];

    while (end > anchor_t_idx + 1 && window.t_positions[end - 1] - anchor_pos > kWindowSpanCap) {
        --end;
    }
    return end;
}

std::optional<McrCandidate> find_candidate_from_anchor(
    const HirModule& hir, const WindowInfo& window, size_t anchor_t_idx, size_t lookahead_cap,
    size_t& candidates_considered, size_t& merge_potential_rejects, size_t& equivalence_checks,
    std::span<const TGateInfo> gate_infos, OriginalSpanSignatureCache& original_span_cache) {
    size_t horizon_end = anchor_horizon_end(window, anchor_t_idx, lookahead_cap);
    if (horizon_end - anchor_t_idx < 4)
        return std::nullopt;

    size_t a = window.t_positions[anchor_t_idx];

    for (size_t b_t = anchor_t_idx + 1; b_t + 2 < horizon_end; ++b_t) {
        size_t b = window.t_positions[b_t];
        for (size_t c_t = b_t + 1; c_t + 1 < horizon_end; ++c_t) {
            size_t c = window.t_positions[c_t];
            for (size_t d_t = c_t + 1; d_t < horizon_end; ++d_t) {
                size_t d = window.t_positions[d_t];
                std::array<size_t, 4> idxs{a, b, c, d};
                if (!distinct_axes(hir, idxs))
                    continue;
                if (!matches_mcr_commutation_pattern(gate_infos[anchor_t_idx], gate_infos[b_t],
                                                     gate_infos[c_t], gate_infos[d_t])) {
                    continue;
                }
                ++candidates_considered;
                McrCandidate cand{
                    .a = a,
                    .b = b,
                    .c = c,
                    .d = d,
                    .anchor_t_idx = anchor_t_idx,
                    .b_t_idx = b_t,
                    .c_t_idx = c_t,
                    .d_t_idx = d_t,
                    .window_start = window.start,
                    .window_end = window.end,
                };
                if (!window_swap_has_merge_potential(hir, window, cand)) {
                    ++merge_potential_rejects;
                    continue;
                }
                ++equivalence_checks;
                bool is_valid = exact_span_swap_rewrite_is_valid(gate_infos, anchor_t_idx, d_t,
                                                                 cand, original_span_cache);
                if (!is_valid)
                    continue;

                return cand;
            }
        }
    }

    return std::nullopt;
}

/// Apply the MCR swap pattern locally, then reuse same-axis T fusion to
/// decide whether the rewrite is worthwhile.
bool apply_candidate(HirModule& hir, const McrCandidate& cand, LocalFusionStats& stats) {
    bool has_source_map = hir.source_map.size() == hir.ops.size();
    size_t before_t = hir.num_t_gates();

    auto op_a = hir.ops[cand.a];
    auto op_b = hir.ops[cand.b];
    auto op_c = hir.ops[cand.c];
    auto op_d = hir.ops[cand.d];
    hir.ops[cand.a] = op_c;
    hir.ops[cand.b] = op_d;
    hir.ops[cand.c] = op_a;
    hir.ops[cand.d] = op_b;
    if (has_source_map) {
        auto src_a = hir.source_map[cand.a];
        auto src_b = hir.source_map[cand.b];
        auto src_c = hir.source_map[cand.c];
        auto src_d = hir.source_map[cand.d];
        hir.source_map[cand.a] = std::move(src_c);
        hir.source_map[cand.b] = std::move(src_d);
        hir.source_map[cand.c] = std::move(src_a);
        hir.source_map[cand.d] = std::move(src_b);
    }

    PeepholeFusionPass peephole;
    peephole.run(hir);
    size_t after_t = hir.num_t_gates();
    stats.merges = peephole.cancellations() + peephole.fusions();
    stats.t_removed = before_t - after_t;
    return after_t < before_t;
}

}  // namespace

void ExperimentalMcrTCountPass::run(HirModule& hir) {
    window_scans_ = 0;
    window_scans_over_lookahead_cap_ = 0;
    candidates_considered_ = 0;
    merge_potential_rejects_ = 0;
    equivalence_checks_ = 0;
    quadruples_found_ = 0;
    swaps_applied_ = 0;
    merges_ = 0;
    t_removed_ = 0;

    bool changed = true;
    while (changed) {
        changed = false;

        for (const auto& window : collect_windows(hir)) {
            std::vector<TGateInfo> gate_infos = build_window_t_gate_infos(hir, window);
            OriginalSpanSignatureCache original_span_cache;
            ++window_scans_;
            if (window.t_positions.size() > kLookaheadCap)
                ++window_scans_over_lookahead_cap_;

            for (size_t anchor_t_idx = 0; anchor_t_idx < window.t_positions.size();
                 ++anchor_t_idx) {
                auto cand = find_candidate_from_anchor(
                    hir, window, anchor_t_idx, kLookaheadCap, candidates_considered_,
                    merge_potential_rejects_, equivalence_checks_, gate_infos, original_span_cache);
                if (!cand.has_value())
                    continue;

                ++quadruples_found_;

                HirModule trial = hir;
                LocalFusionStats stats;
                if (!apply_candidate(trial, *cand, stats))
                    continue;

                size_t before_t = hir.num_t_gates();
                size_t after_t = trial.num_t_gates();
                if (after_t >= before_t)
                    continue;

                hir = std::move(trial);
                ++swaps_applied_;
                merges_ += stats.merges;
                t_removed_ += before_t - after_t;
                changed = true;
                break;
            }

            if (changed)
                break;
        }
    }
}

}  // namespace clifft
