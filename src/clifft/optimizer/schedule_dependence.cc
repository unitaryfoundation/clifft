#include "clifft/optimizer/schedule_dependence.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace clifft::detail {

namespace {

constexpr size_t kDenseScanCandidatesPerWord = 8;
constexpr size_t kMaxIndexedSupport = 256;

bool is_movable_op(OpType type) {
    return type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
}

// Check noise transparency first: a type check avoids walking noise-channel
// masks when sign correction already makes the crossing legal.
bool allowed(const HirModule& hir, const HeisenbergOp& left, const HeisenbergOp& right,
             bool noise_transparent, const PauliCommutation* right_axis = nullptr) {
    if (noise_transparent) {
        const bool left_noise = left.op_type() == OpType::NOISE;
        const bool right_noise = right.op_type() == OpType::NOISE;
        const bool left_movable = is_movable_op(left.op_type());
        const bool right_movable = is_movable_op(right.op_type());
        if ((left_noise && right_movable) || (right_noise && left_movable)) {
            return true;
        }
    }
    return can_swap(right, left, hir, right_axis);
}

void append_unique_back(std::vector<uint32_t>& vec, uint32_t val) {
    if (vec.empty() || vec.back() != val) {
        vec.push_back(val);
    }
}

// Per-qubit and per-record lists of earlier ops for one movability class.
// High-weight ops and barriers go in wide_ops so each op adds a bounded
// number of index entries even on dense multi-thousand-qubit Paulis.
struct ConflictIndex {
    std::vector<std::vector<uint32_t>> with_x;
    std::vector<std::vector<uint32_t>> with_z;
    std::vector<std::vector<uint32_t>> with_slot;
    std::vector<uint32_t> wide_ops;
};

// Sort by key and value so each CSR adjacency list is ordered.
void group_into_csr(size_t num_keys, std::vector<std::pair<uint32_t, uint32_t>> pairs,
                    std::vector<uint32_t>& offsets, std::vector<uint32_t>& indices) {
    std::ranges::sort(pairs);
    offsets.assign(num_keys + 1, 0);
    for (const auto& entry : pairs) {
        ++offsets[entry.first + 1];
    }
    for (size_t i = 0; i < num_keys; ++i) {
        offsets[i + 1] += offsets[i];
    }
    indices.clear();
    indices.reserve(pairs.size());
    for (const auto& entry : pairs) {
        indices.push_back(entry.second);
    }
}

}  // namespace

ScheduleDependence ScheduleDependence::build(const HirModule& hir,
                                             ScheduleDependenceOptions options) {
    const size_t n = hir.ops.size();
    ScheduleDependence dep;
    dep.noise_transparent_ = options.noise_transparent;
    dep.num_ops_ = n;

    std::vector<std::pair<uint32_t, uint32_t>> edges;

    if (n > 0) {
        // Cache ancestor sets in a ring. Evicted rows only cost extra checks
        // and redundant edges; the constraints remain the same.
        const size_t words = (n + 63) / 64;
        const size_t row_bytes = words * sizeof(uint64_t);
        const size_t rows_in_budget = options.ancestor_cache_bytes / row_bytes;
        const size_t ring_rows = std::max<size_t>(1, std::min(n, rows_in_budget));

        std::vector<bool> movable(n);
        std::vector<uint64_t> movable_bits(words, 0);
        for (size_t i = 0; i < n; ++i) {
            if (is_movable_op(hir.ops[i].op_type())) {
                movable[i] = true;
                movable_bits[i / 64] |= (uint64_t{1} << (i % 64));
            }
        }

        // Index earlier ops by movability (0 = fixed, 1 = movable) and by the
        // qubits and classical slots they touch so a sparse op on a wide
        // register inspects only potential conflicts.
        const size_t num_record_slots =
            static_cast<size_t>(hir.num_measurements) + hir.num_hidden_measurements;
        const uint32_t noise_words =
            dep.noise_transparent_ ? 0u : hir.noise_channel_masks.num_words();
        const uint32_t mask_words = std::max(hir.pauli_masks.num_words(), noise_words);
        const size_t num_bit_slots = static_cast<size_t>(mask_words) * 64;

        std::array<ConflictIndex, 2> index;
        for (ConflictIndex& idx : index) {
            idx.with_x.resize(num_bit_slots);
            idx.with_z.resize(num_bit_slots);
            idx.with_slot.resize(num_record_slots);
        }

        std::vector<uint64_t> noise_ux(noise_words, 0);
        std::vector<uint64_t> noise_uz(noise_words, 0);
        std::vector<uint32_t> candidates;

        std::vector<uint64_t> ancestor_rows(ring_rows * words, 0);
        auto row = [&](size_t k) {
            return std::span<uint64_t>(ancestor_rows).subspan((k % ring_rows) * words, words);
        };

        std::optional<uint32_t> previous_fixed;

        for (size_t j = 0; j < n; ++j) {
            const std::span<uint64_t> rj = row(j);
            const size_t j_words = j / 64 + 1;

            const HeisenbergOp& op_j = hir.ops[j];
            const OpType type_j = op_j.op_type();
            const bool j_movable = movable[j];

            // Known ancestors of i have index < i, so only words [0, i / 64] can
            // be nonzero. Only read cached rows that have not been evicted.
            auto link = [&](uint32_t i) {
                if (j - i < ring_rows) {
                    const uint64_t* ri = row(i).data();
                    const size_t i_words = i / 64 + 1;
                    for (size_t w = 0; w < i_words; ++w) {
                        rj[w] |= ri[w];
                    }
                }
                rj[i / 64] |= (uint64_t{1} << (i % 64));
                edges.emplace_back(i, static_cast<uint32_t>(j));
            };

            // Fixed ops retain their relative order even when they commute.
            // Keep consecutive fixed ops directly linked for adjacency queries.
            if (!j_movable && previous_fixed.has_value()) {
                const uint32_t pf = *previous_fixed;
                if (j - pf < ring_rows) {
                    const size_t pf_words = pf / 64 + 1;
                    std::copy_n(row(pf).data(), pf_words, rj.data());
                    std::fill(rj.data() + pf_words, rj.data() + j_words, uint64_t{0});
                } else {
                    std::fill_n(rj.data(), j_words, uint64_t{0});
                }
                rj[pf / 64] |= (uint64_t{1} << (pf % 64));
                edges.emplace_back(pf, static_cast<uint32_t>(j));
                previous_fixed = static_cast<uint32_t>(j);
            } else {
                std::fill_n(rj.data(), j_words, uint64_t{0});
                if (!j_movable) {
                    previous_fixed = static_cast<uint32_t>(j);
                }
            }

            if (type_j == OpType::NOISE && dep.noise_transparent_) {
                continue;
            }

            std::optional<PauliCommutation> j_axis;
            if (type_j == OpType::T_GATE || type_j == OpType::PHASE_ROTATION ||
                type_j == OpType::MEASURE || type_j == OpType::CONDITIONAL_PAULI) {
                j_axis.emplace(hir.destab_mask(op_j), hir.stab_mask(op_j));
            }
            const PauliCommutation* j_axis_ptr = j_axis ? &*j_axis : nullptr;

            uint32_t noise_begin = 0;
            uint32_t noise_end = 0;
            if (type_j == OpType::NOISE) {
                std::ranges::fill(noise_ux, uint64_t{0});
                std::ranges::fill(noise_uz, uint64_t{0});
                const NoiseSite& site =
                    hir.noise_sites[static_cast<uint32_t>(op_j.noise_site_idx())];
                for (const NoiseChannel& ch : site.channels) {
                    const PauliMaskView cv = hir.noise_channel_masks.at(ch.mask);
                    for (uint32_t w = 0; w < noise_words; ++w) {
                        noise_ux[w] |= cv.x().words[w];
                        noise_uz[w] |= cv.z().words[w];
                    }
                }
                noise_end = noise_words;
                while (noise_end > 0 && (noise_ux[noise_end - 1] | noise_uz[noise_end - 1]) == 0) {
                    --noise_end;
                }
                while (noise_begin < noise_end &&
                       (noise_ux[noise_begin] | noise_uz[noise_begin]) == 0) {
                    ++noise_begin;
                }
            }

            auto for_each_support = [&](auto&& on_x, auto&& on_z, auto&& on_slot) {
                switch (type_j) {
                    case OpType::T_GATE:
                    case OpType::PHASE_ROTATION:
                        for_each_set_bit(hir.destab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_x);
                        for_each_set_bit(hir.stab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_z);
                        break;
                    case OpType::MEASURE:
                        on_slot(static_cast<uint32_t>(op_j.meas_record_idx()));
                        for_each_set_bit(hir.destab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_x);
                        for_each_set_bit(hir.stab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_z);
                        break;
                    case OpType::CONDITIONAL_PAULI:
                        on_slot(static_cast<uint32_t>(op_j.controlling_meas()));
                        for_each_set_bit(hir.destab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_x);
                        for_each_set_bit(hir.stab_mask(op_j), j_axis->begin_word(),
                                         j_axis->end_word(), on_z);
                        break;
                    case OpType::NOISE:
                        for_each_set_bit(MaskView{noise_ux}, noise_begin, noise_end, on_x);
                        for_each_set_bit(MaskView{noise_uz}, noise_begin, noise_end, on_z);
                        break;
                    case OpType::READOUT_NOISE:
                        on_slot(hir.readout_noise[static_cast<uint32_t>(op_j.readout_noise_idx())]
                                    .meas_idx);
                        break;
                    case OpType::DETECTOR:
                        for (uint32_t m :
                             hir.detector_targets[static_cast<uint32_t>(op_j.detector_idx())]) {
                            on_slot(m);
                        }
                        break;
                    case OpType::OBSERVABLE:
                        for (uint32_t m :
                             hir.observable_targets[op_j.observable_target_list_idx()]) {
                            on_slot(m);
                        }
                        break;
                    case OpType::EXP_VAL:
                    case OpType::INSTRUMENT:
                    case OpType::NUM_OP_TYPES:
                        break;
                }
            };

            // Try nearby predecessors first: their ancestor sets can rule out
            // many earlier candidates at once.
            auto scan_dense = [&]() {
                const size_t top_word = (j - 1) / 64;
                for (size_t w = top_word + 1; w-- > 0;) {
                    uint64_t mask = ~rj[w];
                    if (!j_movable) {
                        mask &= movable_bits[w];
                    }
                    if (w == top_word) {
                        const size_t valid_bits = j - w * 64;
                        if (valid_bits < 64) {
                            mask &= (uint64_t{1} << valid_bits) - 1;
                        }
                    }
                    while (mask != 0) {
                        const int bit = 63 - std::countl_zero(mask);
                        mask &= ~(uint64_t{1} << bit);
                        const auto i = static_cast<uint32_t>(w * 64 + static_cast<size_t>(bit));
                        if (!allowed(hir, hir.ops[i], op_j, dep.noise_transparent_, j_axis_ptr)) {
                            link(i);
                            // Skip ancestors just absorbed by link().
                            mask &= ~rj[w];
                        }
                    }
                }
            };

            const bool is_barrier = type_j == OpType::EXP_VAL || type_j == OpType::INSTRUMENT;
            size_t support_count = 0;
            size_t total_candidates =
                index[1].wide_ops.size() + (j_movable ? index[0].wide_ops.size() : 0);
            if (!is_barrier) {
                for_each_support(
                    [&](uint32_t q) {
                        ++support_count;
                        total_candidates += index[1].with_z[q].size();
                        if (j_movable) {
                            total_candidates += index[0].with_z[q].size();
                        }
                    },
                    [&](uint32_t q) {
                        ++support_count;
                        total_candidates += index[1].with_x[q].size();
                        if (j_movable) {
                            total_candidates += index[0].with_x[q].size();
                        }
                    },
                    [&](uint32_t m) {
                        assert(m < num_record_slots);
                        ++support_count;
                        total_candidates += index[1].with_slot[m].size();
                        if (j_movable) {
                            total_candidates += index[0].with_slot[m].size();
                        }
                    });
            }

            if (j > 0) {
                if (is_barrier) {
                    scan_dense();
                } else {
                    // Fall back to the ancestor bitset scan on narrow dense
                    // circuits where candidate lists grow longer than the row.
                    const size_t top_word = (j - 1) / 64;
                    if (total_candidates > (top_word + 1) * kDenseScanCandidatesPerWord) {
                        scan_dense();
                    } else if (total_candidates > 0) {
                        candidates.clear();
                        auto collect = [&](std::span<const uint32_t> ops) {
                            for (uint32_t i : ops) {
                                if ((rj[i / 64] & (uint64_t{1} << (i % 64))) == 0) {
                                    candidates.push_back(i);
                                }
                            }
                        };
                        collect(index[1].wide_ops);
                        if (j_movable) {
                            collect(index[0].wide_ops);
                        }
                        for_each_support(
                            [&](uint32_t q) {
                                collect(index[1].with_z[q]);
                                if (j_movable) {
                                    collect(index[0].with_z[q]);
                                }
                            },
                            [&](uint32_t q) {
                                collect(index[1].with_x[q]);
                                if (j_movable) {
                                    collect(index[0].with_x[q]);
                                }
                            },
                            [&](uint32_t m) {
                                collect(index[1].with_slot[m]);
                                if (j_movable) {
                                    collect(index[0].with_slot[m]);
                                }
                            });
                        if (candidates.size() > 1) {
                            std::ranges::sort(candidates, std::greater<uint32_t>{});
                        }
                        uint32_t prev_i = static_cast<uint32_t>(n);
                        for (uint32_t i : candidates) {
                            if (i == prev_i) {
                                continue;
                            }
                            prev_i = i;
                            if ((rj[i / 64] & (uint64_t{1} << (i % 64))) != 0) {
                                continue;
                            }
                            if (!allowed(hir, hir.ops[i], op_j, dep.noise_transparent_,
                                         j_axis_ptr)) {
                                link(i);
                            }
                        }
                    }
                }
            }

            if (j + 1 < n) {
                const auto j_idx = static_cast<uint32_t>(j);
                ConflictIndex& idx = index[j_movable ? 1 : 0];
                if (is_barrier || support_count > kMaxIndexedSupport) {
                    idx.wide_ops.push_back(j_idx);
                } else {
                    for_each_support(
                        [&](uint32_t q) { idx.with_x[q].push_back(j_idx); },
                        [&](uint32_t q) { idx.with_z[q].push_back(j_idx); },
                        [&](uint32_t m) { append_unique_back(idx.with_slot[m], j_idx); });
                }
            }
        }
    }

    group_into_csr(n, edges, dep.succ_offsets_, dep.succ_indices_);

    std::vector<std::pair<uint32_t, uint32_t>> reversed_edges;
    reversed_edges.reserve(edges.size());
    for (const auto& [from, to] : edges) {
        reversed_edges.emplace_back(to, from);
    }
    group_into_csr(n, std::move(reversed_edges), dep.pred_offsets_, dep.pred_indices_);

    return dep;
}

std::span<const uint32_t> ScheduleDependence::predecessors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(pred_indices_)
        .subspan(pred_offsets_[op], pred_offsets_[op + 1] - pred_offsets_[op]);
}

std::span<const uint32_t> ScheduleDependence::successors(size_t op) const {
    assert(op < num_ops());
    return std::span<const uint32_t>(succ_indices_)
        .subspan(succ_offsets_[op], succ_offsets_[op + 1] - succ_offsets_[op]);
}

bool ScheduleDependence::is_linear_extension(std::span<const uint32_t> order) const {
    const size_t n = num_ops();
    if (order.size() != n) {
        return false;
    }
    std::vector<uint32_t> position(n, static_cast<uint32_t>(n));
    for (size_t pos = 0; pos < order.size(); ++pos) {
        const uint32_t op = order[pos];
        if (op >= n || position[op] != n) {
            return false;
        }
        position[op] = static_cast<uint32_t>(pos);
    }
    for (size_t op = 0; op < n; ++op) {
        for (uint32_t succ : successors(op)) {
            if (position[op] >= position[succ]) {
                return false;
            }
        }
    }
    return true;
}

void apply_schedule(HirModule& hir, const ScheduleDependence& dependence,
                    std::span<const uint32_t> order) {
    assert(hir.ops.size() == dependence.num_ops());
    if (!dependence.is_linear_extension(order)) {
        throw std::invalid_argument(
            "apply_schedule: order is not a linear extension of the dependence relation");
    }

    if (dependence.noise_transparent()) {
        hir.materialize_logical_noise_prefix();
    }

    const bool has_source_map = hir.source_map.size() == hir.ops.size();
    const bool has_lnp = hir.has_logical_noise_prefix();

    std::vector<HeisenbergOp> new_ops;
    new_ops.reserve(hir.ops.size());
    std::vector<std::vector<uint32_t>> new_source_map;
    std::vector<uint32_t> new_lnp;
    if (has_source_map) {
        new_source_map.reserve(hir.ops.size());
    }
    if (has_lnp) {
        new_lnp.reserve(hir.ops.size());
    }

    for (uint32_t idx : order) {
        new_ops.push_back(hir.ops[idx]);
        if (has_source_map) {
            new_source_map.push_back(std::move(hir.source_map[idx]));
        }
        if (has_lnp) {
            new_lnp.push_back(hir.logical_noise_prefix[idx]);
        }
    }

    hir.ops = std::move(new_ops);
    if (has_source_map) {
        hir.source_map = std::move(new_source_map);
    }
    if (has_lnp) {
        hir.logical_noise_prefix = std::move(new_lnp);
    }
}

}  // namespace clifft::detail
