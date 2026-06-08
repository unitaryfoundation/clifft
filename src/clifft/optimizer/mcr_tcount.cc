#include "clifft/optimizer/mcr_tcount.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/t_fusion.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace clifft {

namespace {

constexpr size_t kLookaheadCap = 16;
constexpr size_t kWindowSpanCap = 64;

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
    size_t start = 0;
    size_t end = 0;
    std::vector<size_t> t_positions;
    std::unordered_map<AxisKey, std::vector<size_t>, AxisKeyHash> t_indices_by_axis;
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
                      pauli_product_phase_mod4(acc_x_view, acc_z_view, rhs_x, rhs_z)) %
                     4;
        for (size_t w = 0; w < acc_x.size(); ++w) {
            acc_x[w] ^= rhs_x.words[w];
            acc_z[w] ^= rhs_z.words[w];
        }
        acc_x_view = MaskView{std::span<const uint64_t>(acc_x)};
        acc_z_view = MaskView{std::span<const uint64_t>(acc_z)};
    }

    for (size_t w = 0; w < acc_x.size(); ++w) {
        if (acc_x[w] != 0 || acc_z[w] != 0)
            return false;
    }
    return phase_mod4 == 2;
}

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

std::vector<WindowInfo> collect_windows(HirModule& hir) {
    std::vector<WindowInfo> windows;
    size_t i = 0;
    while (i < hir.ops.size()) {
        while (i < hir.ops.size() && is_window_barrier(hir.ops[i])) {
            ++i;
        }
        if (i >= hir.ops.size())
            break;

        WindowInfo window;
        window.start = i;

        while (i < hir.ops.size() && !is_window_barrier(hir.ops[i])) {
            if (hir.ops[i].op_type() == OpType::T_GATE) {
                normalize_t_sign(hir, hir.ops[i]);
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

size_t anchor_horizon_end(const WindowInfo& window, size_t anchor_t_idx) {
    size_t end = std::min(window.t_positions.size(), anchor_t_idx + kLookaheadCap);
    size_t anchor_pos = window.t_positions[anchor_t_idx];
    while (end > anchor_t_idx + 1 && window.t_positions[end - 1] - anchor_pos > kWindowSpanCap) {
        --end;
    }
    return end;
}

std::optional<McrCandidate> find_candidate_from_anchor(const HirModule& hir,
                                                       const WindowInfo& window,
                                                       size_t anchor_t_idx) {
    size_t horizon_end = anchor_horizon_end(window, anchor_t_idx);
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

                return McrCandidate{a, b, c, d, window.start, window.end};
            }
        }
    }
    return std::nullopt;
}

bool apply_candidate(HirModule& hir, const McrCandidate& cand, SameAxisTStats& stats) {
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

    std::rotate(hir.ops.begin() + static_cast<ptrdiff_t>(a),
                hir.ops.begin() + static_cast<ptrdiff_t>(a + 2),
                hir.ops.begin() + static_cast<ptrdiff_t>(a + 4));
    if (has_source_map) {
        std::rotate(hir.source_map.begin() + static_cast<ptrdiff_t>(a),
                    hir.source_map.begin() + static_cast<ptrdiff_t>(a + 2),
                    hir.source_map.begin() + static_cast<ptrdiff_t>(a + 4));
    }

    size_t window_end = cand.window_end;
    SameAxisTStats before = stats;
    fuse_same_axis_t_in_range(hir, cand.window_start, window_end, stats);
    return stats.t_removed > before.t_removed;
}

}  // namespace

void run_mcr_tcount(HirModule& hir, McrTcountStats& stats) {
    stats = McrTcountStats{};

    bool changed = true;
    while (changed) {
        changed = false;

        for (const auto& window : collect_windows(hir)) {
            ++stats.window_scans;
            if (window.t_positions.size() > kLookaheadCap)
                ++stats.window_scans_over_lookahead_cap;

            for (size_t anchor_t_idx = 0; anchor_t_idx < window.t_positions.size(); ++anchor_t_idx) {
                auto cand = find_candidate_from_anchor(hir, window, anchor_t_idx);
                if (!cand.has_value())
                    continue;

                ++stats.quadruples_found;

                HirModule trial = hir;
                SameAxisTStats fusion;
                if (!apply_candidate(trial, *cand, fusion))
                    continue;

                size_t before_t = hir.num_t_gates();
                size_t after_t = trial.num_t_gates();
                if (after_t >= before_t)
                    continue;

                hir = std::move(trial);
                ++stats.swaps_applied;
                stats.merges += fusion.merges;
                stats.t_removed += before_t - after_t;
                changed = true;
                break;
            }

            if (changed)
                break;
        }
    }
}

}  // namespace clifft
