#include "clifft/optimizer/todd_phase_pass.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/gf2_phase.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/virtual_clifford.h"
#include "clifft/util/constants.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

namespace {

constexpr uint32_t kMaxToddQubits = 12;
constexpr size_t kMaxToddColumns = 64;
constexpr size_t kMaxToddRounds = 32;

inline void normalize_t_sign(HirModule& hir, HeisenbergOp& op) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        hir.global_weight *= op.is_dagger() ? kExpMinusIPiOver4 : kExpIPiOver4;
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

void extract_parity_column(const HirModule& hir, const HeisenbergOp& op, uint32_t num_words,
                           uint64_t* out) {
    for (uint32_t w = 0; w < num_words; ++w) {
        out[w] = 0;
    }
    auto xd = hir.destab_mask(op);
    auto zs = hir.stab_mask(op);
    if (xd.is_zero()) {
        for (uint32_t w = 0; w < num_words; ++w) {
            out[w] = zs.words[w];
        }
    } else {
        for (uint32_t w = 0; w < num_words; ++w) {
            out[w] = xd.words[w];
        }
    }
}

bool t_gates_commute(const HirModule& hir, const HeisenbergOp& a, const HeisenbergOp& b) {
    return !anti_commute(hir.destab_mask(a), hir.stab_mask(a), hir.destab_mask(b),
                         hir.stab_mask(b));
}

int tgate_coeff_mod8(const HeisenbergOp& op) {
    return op.is_dagger() ? 7 : 1;
}

bool is_t_window_barrier(const HeisenbergOp& op) {
    return op.op_type() != OpType::T_GATE;
}

struct TWindow {
    size_t start;
    size_t end;
    std::vector<size_t> t_positions;
};

std::vector<TWindow> collect_t_windows(const HirModule& hir) {
    std::vector<TWindow> windows;
    size_t i = 0;
    while (i < hir.ops.size()) {
        while (i < hir.ops.size() && is_t_window_barrier(hir.ops[i])) {
            ++i;
        }
        if (i >= hir.ops.size()) {
            break;
        }
        TWindow window{.start = i, .end = i};
        while (i < hir.ops.size() && !is_t_window_barrier(hir.ops[i])) {
            if (hir.ops[i].op_type() == OpType::T_GATE) {
                window.t_positions.push_back(i);
            }
            ++i;
        }
        window.end = i;
        if (window.t_positions.size() >= 3) {
            windows.push_back(std::move(window));
        }
    }
    return windows;
}

void absorb_even_coeff_clifford(HirModule& hir, size_t after_idx, const PauliAxis& axis,
                                int coeff_mod8, std::vector<uint8_t>& deleted) {
    coeff_mod8 %= 8;
    if (coeff_mod8 < 0) {
        coeff_mod8 += 8;
    }
    if (coeff_mod8 == 0 || (coeff_mod8 & 1) != 0) {
        return;
    }
    MaskView xv{std::span<const uint64_t>(axis.x)};
    MaskView zv{std::span<const uint64_t>(axis.z)};
    if (coeff_mod8 == 2) {
        apply_virtual_s_downstream(hir, after_idx, xv, zv, false, false, deleted);
    } else if (coeff_mod8 == 6) {
        apply_virtual_s_downstream(hir, after_idx, xv, zv, false, true, deleted);
    } else if (coeff_mod8 == 4) {
        hir.global_weight *= -1.0;
    }
}

HeisenbergOp& append_t_from_axis(HirModule& hir, const PauliAxis& axis, bool dagger) {
    return hir.append_tgate(dagger, [&axis](MutablePauliMaskView slot) {
        for (uint32_t w = 0; w < axis.x.size(); ++w) {
            slot.x().words[w] = axis.x[w];
            slot.z().words[w] = axis.z[w];
        }
        slot.set_sign(false);
    });
}

bool optimize_commuting_cluster(HirModule& hir, const std::vector<size_t>& cluster, size_t& saved) {
    if (cluster.size() < 3) {
        return false;
    }

    const uint32_t n = hir.num_qubits;
    if (n == 0 || n > kMaxToddQubits) {
        return false;
    }

    const uint32_t num_words = (n + 63) / 64;
    Gf2Matrix mat;
    mat.n = n;
    mat.num_words = num_words;
    std::vector<int> coeffs;
    std::vector<PauliAxis> axes;
    std::vector<uint64_t> col_buf(num_words);

    for (size_t idx : cluster) {
        auto& op = hir.ops[idx];
        normalize_t_sign(hir, op);
        extract_parity_column(hir, op, num_words, col_buf.data());
        mat.append_col(col_buf.data());
        coeffs.push_back(tgate_coeff_mod8(op));
        PauliAxis axis;
        axis.resize(num_words);
        axis.set_from(hir.destab_mask(op), hir.stab_mask(op));
        axes.push_back(std::move(axis));
    }

    const size_t t_before = cluster.size();
    if (!todd_optimize(mat, coeffs, kMaxToddQubits, kMaxToddColumns, kMaxToddRounds, &axes)) {
        return false;
    }

    size_t odd_terms = 0;
    for (int c : coeffs) {
        if ((c & 1) != 0) {
            ++odd_terms;
        }
    }
    if (odd_terms >= t_before) {
        return false;
    }

    std::vector<uint8_t> deleted(hir.ops.size(), 0);
    const size_t new_start = hir.ops.size();

    for (size_t j = 0; j < mat.num_cols(); ++j) {
        int c = coeffs[j] % 8;
        if (c < 0) {
            c += 8;
        }
        if ((c & 1) == 0) {
            absorb_even_coeff_clifford(hir, new_start, axes[j], c, deleted);
            continue;
        }
        bool dagger = (c == 7 || c == 3);
        append_t_from_axis(hir, axes[j], dagger);
    }
    const size_t new_end = hir.ops.size();

    std::vector<HeisenbergOp> new_ts(hir.ops.begin() + static_cast<ptrdiff_t>(new_start),
                                     hir.ops.begin() + static_cast<ptrdiff_t>(new_end));

    std::vector<HeisenbergOp> rebuilt;
    rebuilt.reserve(hir.ops.size() - cluster.size() + odd_terms);

    std::vector<uint8_t> in_cluster(new_start, 0);
    for (size_t idx : cluster) {
        if (idx < new_start) {
            in_cluster[idx] = 1;
        }
    }

    const size_t insert_at = cluster.front();
    bool inserted = false;
    for (size_t i = 0; i < new_start; ++i) {
        if (i == insert_at && !inserted) {
            rebuilt.insert(rebuilt.end(), new_ts.begin(), new_ts.end());
            inserted = true;
        }
        if (!in_cluster[i]) {
            rebuilt.push_back(hir.ops[i]);
        }
    }
    if (!inserted) {
        rebuilt.insert(rebuilt.end(), new_ts.begin(), new_ts.end());
    }

    saved = t_before - odd_terms;
    hir.ops = std::move(rebuilt);
    return true;
}

bool run_todd_on_windows(HirModule& hir, size_t& blocks, size_t& t_removed) {
    bool any = false;
    bool changed = true;
    while (changed) {
        changed = false;
        auto windows = collect_t_windows(hir);
        for (const auto& window : windows) {
            std::vector<uint8_t> used(window.t_positions.size(), 0);
            for (size_t ti = 0; ti < window.t_positions.size(); ++ti) {
                if (used[ti]) {
                    continue;
                }
                std::vector<size_t> cluster;
                cluster.push_back(window.t_positions[ti]);
                for (size_t tj = ti + 1; tj < window.t_positions.size(); ++tj) {
                    size_t cand = window.t_positions[tj];
                    bool ok = true;
                    for (size_t member : cluster) {
                        if (!t_gates_commute(hir, hir.ops[member], hir.ops[cand])) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        cluster.push_back(cand);
                    }
                }
                if (cluster.size() < 3) {
                    continue;
                }
                for (size_t tj = ti; tj < window.t_positions.size(); ++tj) {
                    if (std::find(cluster.begin(), cluster.end(), window.t_positions[tj]) !=
                        cluster.end()) {
                        used[tj] = 1;
                    }
                }
                size_t saved = 0;
                if (optimize_commuting_cluster(hir, cluster, saved)) {
                    ++blocks;
                    t_removed += saved;
                    changed = true;
                    any = true;
                    break;
                }
            }
            if (changed) {
                break;
            }
        }
    }
    return any;
}

}  // namespace

void ToddPhasePass::run(HirModule& hir) {
    blocks_ = 0;
    t_removed_ = 0;
    run_todd_on_windows(hir, blocks_, t_removed_);
}

}  // namespace clifft
