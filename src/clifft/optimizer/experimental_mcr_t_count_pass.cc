#include "clifft/optimizer/experimental_mcr_t_count_pass.h"

#include "clifft/optimizer/commutation.h"
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
    size_t window_start;
    size_t window_end;
};

struct WindowInfo {
    size_t start;
    size_t end;
    std::vector<size_t> t_positions;
    std::unordered_map<AxisKey, std::vector<size_t>, AxisKeyHash> t_indices_by_axis;
};

struct LocalFusionStats {
    size_t merges = 0;
    size_t t_removed = 0;
};

constexpr size_t kWindowSpanCap = 64;

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

AxisKey make_axis_key(MaskView x, MaskView z) {
    AxisKey key;
    key.x_words.assign(x.words.begin(), x.words.end());
    key.z_words.assign(z.words.begin(), z.words.end());
    return key;
}

AxisKey xor_axis_key(const HirModule& hir, const HeisenbergOp& a, const HeisenbergOp& b,
                     const HeisenbergOp& c) {
    AxisKey key;
    size_t words = hir.destab_mask(a).num_words();
    key.x_words.resize(words);
    key.z_words.resize(words);

    for (size_t i = 0; i < words; ++i) {
        key.x_words[i] =
            hir.destab_mask(a).words[i] ^ hir.destab_mask(b).words[i] ^ hir.destab_mask(c).words[i];
        key.z_words[i] =
            hir.stab_mask(a).words[i] ^ hir.stab_mask(b).words[i] ^ hir.stab_mask(c).words[i];
    }
    return key;
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

bool commute(const HirModule& hir, const HeisenbergOp& lhs, const HeisenbergOp& rhs) {
    return !anti_commute(hir.destab_mask(lhs), hir.stab_mask(lhs), hir.destab_mask(rhs),
                         hir.stab_mask(rhs));
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

/// This prototype only rewrites uniform-direction quarter turns.
bool same_t_direction(const HirModule& hir, const std::array<size_t, 4>& idxs) {
    bool is_dagger = hir.ops[idxs[0]].is_dagger();
    for (size_t i = 1; i < idxs.size(); ++i) {
        if (hir.ops[idxs[i]].is_dagger() != is_dagger)
            return false;
    }
    return true;
}

bool exact_mcr_product_relation(const HirModule& hir, const std::array<size_t, 4>& idxs) {
    const auto& first_op = hir.ops[idxs[0]];
    MaskView first_x = hir.destab_mask(first_op);
    MaskView first_z = hir.stab_mask(first_op);

    std::vector<uint64_t> acc_x(first_x.words.begin(), first_x.words.end());
    std::vector<uint64_t> acc_z(first_z.words.begin(), first_z.words.end());
    MaskView acc_x_view{std::span<const uint64_t>(acc_x)};
    MaskView acc_z_view{std::span<const uint64_t>(acc_z)};

    int phase_mod4 = hir.sign(first_op) ? 2 : 0;
    for (size_t i = 1; i < idxs.size(); ++i) {
        const auto& op = hir.ops[idxs[i]];
        MaskView rhs_x = hir.destab_mask(op);
        MaskView rhs_z = hir.stab_mask(op);
        phase_mod4 = (phase_mod4 + (hir.sign(op) ? 2 : 0) +
                      mul_phase_mod4(acc_x_view, acc_z_view, rhs_x, rhs_z)) %
                     4;
        for (size_t w = 0; w < acc_x.size(); ++w) {
            acc_x[w] ^= rhs_x.words[w];
            acc_z[w] ^= rhs_z.words[w];
        }
    }

    for (size_t w = 0; w < acc_x.size(); ++w) {
        if (acc_x[w] != 0 || acc_z[w] != 0)
            return false;
    }
    return phase_mod4 == 2;
}

/// Bubble a candidate op left through commuting neighbors only.
bool bubble_left(HirModule& hir, size_t from, size_t to, bool has_source_map) {
    if (from < to)
        return false;
    for (size_t k = from; k > to; --k) {
        if (!can_swap(hir.ops[k - 1], hir.ops[k], hir))
            return false;
        std::swap(hir.ops[k - 1], hir.ops[k]);
        if (has_source_map)
            std::swap(hir.source_map[k - 1], hir.source_map[k]);
    }
    return true;
}

void fuse_same_axis_t_window(HirModule& hir, size_t window_start, size_t window_end,
                             LocalFusionStats& stats) {
    bool has_source_map = hir.source_map.size() == hir.ops.size();
    bool changed = true;

    while (changed) {
        changed = false;
        size_t n = hir.ops.size();
        std::vector<uint8_t> deleted(n, 0);

        for (size_t i = window_start; i < window_end; ++i) {
            if (deleted[i] || hir.ops[i].op_type() != OpType::T_GATE)
                continue;

            normalize_t_sign(hir, hir.ops[i], hir.global_weight);
            auto destab_i = hir.destab_mask(hir.ops[i]);
            auto stab_i = hir.stab_mask(hir.ops[i]);

            for (size_t j = i + 1; j < window_end; ++j) {
                if (deleted[j])
                    continue;
                if (hir.ops[j].op_type() == OpType::T_GATE)
                    normalize_t_sign(hir, hir.ops[j], hir.global_weight);

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

                if (is_t_gate_blocked(op_i, op_j, hir))
                    break;
            }
        }

        if (changed) {
            size_t write = 0;
            for (size_t read = 0; read < n; ++read) {
                if (!deleted[read]) {
                    if (write != read) {
                        hir.ops[write] = hir.ops[read];
                        if (has_source_map)
                            hir.source_map[write] = std::move(hir.source_map[read]);
                    }
                    ++write;
                }
            }
            hir.ops.erase(hir.ops.begin() + static_cast<std::ptrdiff_t>(write), hir.ops.end());
            if (has_source_map)
                hir.source_map.resize(write);
            window_end = std::min(window_end, write);
        }
    }
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
                size_t t_index = window.t_positions.size();
                window.t_positions.push_back(i);
                window
                    .t_indices_by_axis[make_axis_key(hir.destab_mask(hir.ops[i]),
                                                     hir.stab_mask(hir.ops[i]))]
                    .push_back(t_index);
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

std::optional<McrCandidate> find_candidate_from_anchor(const HirModule& hir,
                                                       const WindowInfo& window,
                                                       size_t anchor_t_idx, size_t lookahead_cap) {
    size_t horizon_end = anchor_horizon_end(window, anchor_t_idx, lookahead_cap);
    if (horizon_end - anchor_t_idx < 4)
        return std::nullopt;

    size_t a = window.t_positions[anchor_t_idx];

    for (size_t b_t = anchor_t_idx + 1; b_t + 2 < horizon_end; ++b_t) {
        size_t b = window.t_positions[b_t];
        if (!commute(hir, hir.ops[a], hir.ops[b]))
            continue;

        for (size_t c_t = b_t + 1; c_t + 1 < horizon_end; ++c_t) {
            size_t c = window.t_positions[c_t];
            if (commute(hir, hir.ops[a], hir.ops[c]) || commute(hir, hir.ops[b], hir.ops[c]))
                continue;

            AxisKey target_d = xor_axis_key(hir, hir.ops[a], hir.ops[b], hir.ops[c]);
            auto it = window.t_indices_by_axis.find(target_d);
            if (it == window.t_indices_by_axis.end())
                continue;

            auto d_begin = std::lower_bound(it->second.begin(), it->second.end(), c_t + 1);
            for (auto d_it = d_begin; d_it != it->second.end() && *d_it < horizon_end; ++d_it) {
                size_t d_t = *d_it;

                size_t d = window.t_positions[d_t];
                if (!commute(hir, hir.ops[c], hir.ops[d]))
                    continue;
                if (commute(hir, hir.ops[a], hir.ops[d]) || commute(hir, hir.ops[b], hir.ops[d]))
                    continue;
                std::array<size_t, 4> idxs{a, b, c, d};
                if (!distinct_axes(hir, idxs))
                    continue;
                if (!same_t_direction(hir, idxs))
                    continue;
                if (!exact_mcr_product_relation(hir, idxs))
                    continue;

                return McrCandidate{
                    .a = a,
                    .b = b,
                    .c = c,
                    .d = d,
                    .window_start = window.start,
                    .window_end = window.end,
                };
            }
        }
    }

    return std::nullopt;
}

/// Apply the MCR swap pattern locally, then reuse same-axis T fusion to
/// decide whether the rewrite is worthwhile.
bool apply_candidate(HirModule& hir, const McrCandidate& cand, LocalFusionStats& stats) {
    bool has_source_map = hir.source_map.size() == hir.ops.size();

    size_t a = cand.a;
    size_t b = cand.b;
    size_t c = cand.c;
    size_t d = cand.d;

    if (!bubble_left(hir, b, a + 1, has_source_map))
        return false;

    if (!bubble_left(hir, c, a + 2, has_source_map))
        return false;

    if (!bubble_left(hir, d, a + 3, has_source_map))
        return false;

    std::rotate(hir.ops.begin() + static_cast<std::ptrdiff_t>(a),
                hir.ops.begin() + static_cast<std::ptrdiff_t>(a + 2),
                hir.ops.begin() + static_cast<std::ptrdiff_t>(a + 4));
    if (has_source_map) {
        std::rotate(hir.source_map.begin() + static_cast<std::ptrdiff_t>(a),
                    hir.source_map.begin() + static_cast<std::ptrdiff_t>(a + 2),
                    hir.source_map.begin() + static_cast<std::ptrdiff_t>(a + 4));
    }

    fuse_same_axis_t_window(hir, cand.window_start, cand.window_end, stats);
    return stats.t_removed > 0;
}

}  // namespace

void ExperimentalMcrTCountPass::run(HirModule& hir) {
    window_scans_ = 0;
    window_scans_over_lookahead_cap_ = 0;
    quadruples_found_ = 0;
    swaps_applied_ = 0;
    merges_ = 0;
    t_removed_ = 0;

    bool changed = true;
    while (changed) {
        changed = false;

        for (const auto& window : collect_windows(hir)) {
            ++window_scans_;
            if (window.t_positions.size() > kLookaheadCap)
                ++window_scans_over_lookahead_cap_;

            for (size_t anchor_t_idx = 0; anchor_t_idx < window.t_positions.size();
                 ++anchor_t_idx) {
                auto cand = find_candidate_from_anchor(hir, window, anchor_t_idx, kLookaheadCap);
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
