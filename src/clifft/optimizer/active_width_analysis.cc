#include "clifft/optimizer/active_width_analysis.h"

#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace clifft {

namespace detail {

double dense_work_contribution(WidthEffect effect, uint32_t before, uint32_t after) {
    switch (effect) {
        case WidthEffect::RotationNeutral:
        case WidthEffect::RotationPromote:
        case WidthEffect::InstrumentActive:
        case WidthEffect::InstrumentActivate:
            return std::ldexp(1.0, static_cast<int>(after));
        case WidthEffect::MeasureActive:
            return std::ldexp(1.0, static_cast<int>(before));
        default:
            return 0.0;
    }
}

}  // namespace detail

namespace {

// Combined vectors place X bits before Z bits, with each half word-aligned.
uint32_t combined_domain(uint32_t words_per_row) {
    return words_per_row * 64;
}

bool combined_bit_get(MaskView x, MaskView z, uint32_t domain, uint32_t bit) {
    return bit < domain ? x.bit_get(bit) : z.bit_get(bit - domain);
}

uint32_t combined_lowest_bit_in_range(const std::vector<uint64_t>& x,
                                      const std::vector<uint64_t>& z, uint32_t begin_w,
                                      uint32_t end_w, uint32_t domain) {
    for (uint32_t w = begin_w; w < end_w; ++w) {
        if (x[w] != 0) {
            return w * 64 + static_cast<uint32_t>(std::countr_zero(x[w]));
        }
    }
    for (uint32_t w = begin_w; w < end_w; ++w) {
        if (z[w] != 0) {
            return domain + w * 64 + static_cast<uint32_t>(std::countr_zero(z[w]));
        }
    }
    return 2 * domain;
}

bool can_increase_active_width(const HirModule& hir) {
    for (const HeisenbergOp& op : hir.ops) {
        const OpType type = op.op_type();
        if (type == OpType::T_GATE || type == OpType::PHASE_ROTATION ||
            (type == OpType::INSTRUMENT && !detail::instrument_damping_neglected(hir, op))) {
            return true;
        }
    }
    return false;
}

}  // namespace

DormantSubspace::DormantSubspace(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      words_per_row_((num_qubits + 63) / 64),
      dimension_(num_qubits),
      rows_x_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      rows_z_(static_cast<size_t>(num_qubits) * words_per_row_, 0),
      col_x_(static_cast<size_t>(combined_domain(words_per_row_)) * words_per_row_, 0),
      col_z_(static_cast<size_t>(combined_domain(words_per_row_)) * words_per_row_, 0),
      row_begin_(num_qubits, 0),
      row_end_(num_qubits, 0),
      pivot_(num_qubits, 0),
      pivot_owner_(static_cast<size_t>(2) * combined_domain(words_per_row_), kNoOwner),
      row_mask_scratch_(words_per_row_, 0),
      anti_scratch_(words_per_row_, 0),
      scratch_x_(words_per_row_, 0),
      scratch_z_(words_per_row_, 0) {
    // Z on each qubit gives the initial reduced basis for the all-zero state.
    const uint32_t domain = combined_domain(words_per_row_);
    for (uint32_t q = 0; q < num_qubits; ++q) {
        const uint32_t w = q / 64;
        const uint64_t bit = uint64_t{1} << (q % 64);
        rows_z_[static_cast<size_t>(q) * words_per_row_ + w] = bit;
        col_z_[static_cast<size_t>(q) * words_per_row_ + w] = bit;
        row_begin_[q] = w;
        row_end_[q] = w + 1;
        pivot_[q] = domain + q;
        pivot_owner_[domain + q] = q;
    }
}

void DormantSubspace::compute_anticommuting_rows(MaskView x, MaskView z,
                                                 std::span<uint64_t> anti_out) const {
    if (words_per_row_ == 1) {
        uint64_t anti = 0;
        for (uint64_t wx = x.words[0]; wx != 0; wx &= wx - 1) {
            anti ^= col_z_[std::countr_zero(wx)];
        }
        for (uint64_t wz = z.words[0]; wz != 0; wz &= wz - 1) {
            anti ^= col_x_[std::countr_zero(wz)];
        }
        anti_out[0] = anti;
        return;
    }

    std::ranges::fill(anti_out, uint64_t{0});
    const size_t active_rw = anti_out.size();
    for (uint32_t w = 0; w < words_per_row_; ++w) {
        for (uint64_t wx = x.words[w]; wx != 0; wx &= wx - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(wx));
            const uint64_t* cz = col_z_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                anti_out[rw] ^= cz[rw];
            }
        }
        for (uint64_t wz = z.words[w]; wz != 0; wz &= wz - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(wz));
            const uint64_t* cx = col_x_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                anti_out[rw] ^= cx[rw];
            }
        }
    }
}

bool DormantSubspace::commutes_with_all(MaskView x, MaskView z, uint32_t* anticommuting_row) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    if (dimension_ == 0) {
        return true;
    }

    if (words_per_row_ == 1) {
        if (anticommuting_row && *anticommuting_row < dimension_ &&
            (std::popcount((rows_x_[*anticommuting_row] & z.words[0]) ^
                           (rows_z_[*anticommuting_row] & x.words[0])) &
             1) != 0) {
            return false;
        }
        uint64_t anti = 0;
        for (uint64_t wx = x.words[0]; wx != 0; wx &= wx - 1) {
            anti ^= col_z_[std::countr_zero(wx)];
        }
        for (uint64_t wz = z.words[0]; wz != 0; wz &= wz - 1) {
            anti ^= col_x_[std::countr_zero(wz)];
        }
        if (anti == 0) {
            return true;
        }
        if (anticommuting_row) {
            *anticommuting_row = static_cast<uint32_t>(std::countr_zero(anti));
        }
        return false;
    }

    if (anticommuting_row && *anticommuting_row < dimension_) {
        const uint32_t r = *anticommuting_row;
        const uint32_t rb = row_begin_[r];
        const uint32_t re = row_end_[r];
        if (anti_commute(MaskView{row_x(r).words.subspan(rb, re - rb)},
                         MaskView{row_z(r).words.subspan(rb, re - rb)},
                         MaskView{x.words.subspan(rb, re - rb)},
                         MaskView{z.words.subspan(rb, re - rb)})) {
            return false;
        }
    }

    const uint32_t active_rw = (dimension_ + 63) / 64;
    const std::span<uint64_t> anti(anti_scratch_.data(), active_rw);
    compute_anticommuting_rows(x, z, anti);
    for (uint32_t rw = 0; rw < active_rw; ++rw) {
        if (anti[rw] != 0) {
            if (anticommuting_row) {
                *anticommuting_row = rw * 64 + static_cast<uint32_t>(std::countr_zero(anti[rw]));
            }
            return false;
        }
    }
    return true;
}

void DormantSubspace::toggle_row_in_cols(uint32_t row_idx, const uint64_t* vx, const uint64_t* vz,
                                         uint32_t begin_w, uint32_t end_w) {
    const uint32_t rw = row_idx / 64;
    const uint64_t rbit = uint64_t{1} << (row_idx % 64);
    for (uint32_t w = begin_w; w < end_w; ++w) {
        for (uint64_t xw = vx[w]; xw != 0; xw &= xw - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(xw));
            col_x_[static_cast<size_t>(q) * words_per_row_ + rw] ^= rbit;
        }
        for (uint64_t zw = vz[w]; zw != 0; zw &= zw - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(zw));
            col_z_[static_cast<size_t>(q) * words_per_row_ + rw] ^= rbit;
        }
    }
}

void DormantSubspace::xor_rows(const uint64_t* vx, const uint64_t* vz, uint32_t v_begin,
                               uint32_t v_end, std::span<const uint64_t> target_rows) {
    if (v_begin >= v_end) {
        return;
    }
    bool any_target = false;
    for (size_t rw = 0; rw < target_rows.size(); ++rw) {
        uint64_t mask = target_rows[rw];
        if (mask == 0) {
            continue;
        }
        any_target = true;
        while (mask != 0) {
            const auto i =
                static_cast<uint32_t>(rw * 64 + static_cast<size_t>(std::countr_zero(mask)));
            mask &= mask - 1;
            uint64_t* rx = rows_x_.data() + static_cast<size_t>(i) * words_per_row_;
            uint64_t* rz = rows_z_.data() + static_cast<size_t>(i) * words_per_row_;
            for (uint32_t w = v_begin; w < v_end; ++w) {
                rx[w] ^= vx[w];
                rz[w] ^= vz[w];
            }
            uint32_t rb = std::min(row_begin_[i], v_begin);
            uint32_t re = std::max(row_end_[i], v_end);
            while (re > rb && (rx[re - 1] | rz[re - 1]) == 0) {
                --re;
            }
            while (rb < re && (rx[rb] | rz[rb]) == 0) {
                ++rb;
            }
            assert(rb < re && "basis row became zero during elimination");
            row_begin_[i] = rb;
            row_end_[i] = re;
        }
    }
    if (!any_target) {
        return;
    }

    const size_t active_rw = target_rows.size();
    for (uint32_t w = v_begin; w < v_end; ++w) {
        for (uint64_t xw = vx[w]; xw != 0; xw &= xw - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(xw));
            uint64_t* col = col_x_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                col[rw] ^= target_rows[rw];
            }
        }
        for (uint64_t zw = vz[w]; zw != 0; zw &= zw - 1) {
            const uint32_t q = w * 64 + static_cast<uint32_t>(std::countr_zero(zw));
            uint64_t* col = col_z_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                col[rw] ^= target_rows[rw];
            }
        }
    }
}

void DormantSubspace::erase_row(uint32_t k) {
    assert(k < dimension_);
    uint64_t* kx = rows_x_.data() + static_cast<size_t>(k) * words_per_row_;
    uint64_t* kz = rows_z_.data() + static_cast<size_t>(k) * words_per_row_;
    const uint32_t kb = row_begin_[k];
    const uint32_t ke = row_end_[k];

    pivot_owner_[pivot_[k]] = kNoOwner;
    toggle_row_in_cols(k, kx, kz, kb, ke);
    std::fill(kx + kb, kx + ke, uint64_t{0});
    std::fill(kz + kb, kz + ke, uint64_t{0});
    row_begin_[k] = 0;
    row_end_[k] = 0;

    const uint32_t last = dimension_ - 1;
    if (k != last) {
        uint64_t* lx = rows_x_.data() + static_cast<size_t>(last) * words_per_row_;
        uint64_t* lz = rows_z_.data() + static_cast<size_t>(last) * words_per_row_;
        const uint32_t lb = row_begin_[last];
        const uint32_t le = row_end_[last];
        toggle_row_in_cols(last, lx, lz, lb, le);
        std::copy(lx + lb, lx + le, kx + lb);
        std::copy(lz + lb, lz + le, kz + lb);
        std::fill(lx + lb, lx + le, uint64_t{0});
        std::fill(lz + lb, lz + le, uint64_t{0});
        toggle_row_in_cols(k, kx, kz, lb, le);
        row_begin_[k] = lb;
        row_end_[k] = le;
        row_begin_[last] = 0;
        row_end_[last] = 0;
        pivot_[k] = pivot_[last];
        pivot_owner_[pivot_[k]] = k;
    }
    --dimension_;
}

void DormantSubspace::restore_erased_row(uint32_t k, uint32_t saved_pivot, uint32_t saved_begin,
                                         uint32_t saved_end, const uint64_t* saved_x,
                                         const uint64_t* saved_z) {
    const uint32_t last = dimension_;
    ++dimension_;
    uint64_t* kx = rows_x_.data() + static_cast<size_t>(k) * words_per_row_;
    uint64_t* kz = rows_z_.data() + static_cast<size_t>(k) * words_per_row_;

    if (k != last) {
        uint64_t* lx = rows_x_.data() + static_cast<size_t>(last) * words_per_row_;
        uint64_t* lz = rows_z_.data() + static_cast<size_t>(last) * words_per_row_;
        const uint32_t kb = row_begin_[k];
        const uint32_t ke = row_end_[k];
        toggle_row_in_cols(k, kx, kz, kb, ke);
        std::copy(kx + kb, kx + ke, lx + kb);
        std::copy(kz + kb, kz + ke, lz + kb);
        std::fill(kx + kb, kx + ke, uint64_t{0});
        std::fill(kz + kb, kz + ke, uint64_t{0});
        toggle_row_in_cols(last, lx, lz, kb, ke);
        row_begin_[last] = kb;
        row_end_[last] = ke;
        pivot_[last] = pivot_[k];
        pivot_owner_[pivot_[last]] = last;
    }

    const uint32_t len = saved_end - saved_begin;
    std::copy_n(saved_x, len, kx + saved_begin);
    std::copy_n(saved_z, len, kz + saved_begin);
    toggle_row_in_cols(k, kx, kz, saved_begin, saved_end);
    row_begin_[k] = saved_begin;
    row_end_[k] = saved_end;
    pivot_[k] = saved_pivot;
    pivot_owner_[saved_pivot] = k;
}

void DormantSubspace::reduce_into_scratch(MaskView x, MaskView z) const {
    if (scratch_begin_ < scratch_end_) {
        std::fill(scratch_x_.begin() + scratch_begin_, scratch_x_.begin() + scratch_end_,
                  uint64_t{0});
        std::fill(scratch_z_.begin() + scratch_begin_, scratch_z_.begin() + scratch_end_,
                  uint64_t{0});
        scratch_begin_ = 0;
        scratch_end_ = 0;
    }

    if (words_per_row_ == 0) {
        return;
    }

    // Each basis row is zero at every other row's pivot, so only pivots set in
    // the input (x, z) participate in its reduction.
    if (words_per_row_ == 1) {
        uint64_t rx = x.words[0];
        uint64_t rz = z.words[0];
        if ((rx | rz) == 0) {
            return;
        }
        for (uint64_t m = x.words[0]; m != 0; m &= m - 1) {
            const uint32_t owner = pivot_owner_[std::countr_zero(m)];
            if (owner != kNoOwner) {
                rx ^= rows_x_[owner];
                rz ^= rows_z_[owner];
            }
        }
        for (uint64_t m = z.words[0]; m != 0; m &= m - 1) {
            const uint32_t owner = pivot_owner_[64 + std::countr_zero(m)];
            if (owner != kNoOwner) {
                rx ^= rows_x_[owner];
                rz ^= rows_z_[owner];
            }
        }
        if ((rx | rz) != 0) {
            scratch_x_[0] = rx;
            scratch_z_[0] = rz;
            scratch_begin_ = 0;
            scratch_end_ = 1;
        }
        return;
    }

    uint32_t in_end = words_per_row_;
    while (in_end > 0 && (x.words[in_end - 1] | z.words[in_end - 1]) == 0) {
        --in_end;
    }
    uint32_t in_begin = 0;
    while (in_begin < in_end && (x.words[in_begin] | z.words[in_begin]) == 0) {
        ++in_begin;
    }
    if (in_begin == in_end) {
        return;
    }

    std::copy(x.words.begin() + in_begin, x.words.begin() + in_end, scratch_x_.begin() + in_begin);
    std::copy(z.words.begin() + in_begin, z.words.begin() + in_end, scratch_z_.begin() + in_begin);
    uint32_t sb = in_begin;
    uint32_t se = in_end;
    const uint32_t domain = combined_domain(words_per_row_);

    auto xor_owner = [&](uint32_t owner) {
        const uint32_t rb = row_begin_[owner];
        const uint32_t re = row_end_[owner];
        const uint64_t* ox = rows_x_.data() + static_cast<size_t>(owner) * words_per_row_;
        const uint64_t* oz = rows_z_.data() + static_cast<size_t>(owner) * words_per_row_;
        for (uint32_t w = rb; w < re; ++w) {
            scratch_x_[w] ^= ox[w];
            scratch_z_[w] ^= oz[w];
        }
        sb = std::min(sb, rb);
        se = std::max(se, re);
    };

    for (uint32_t w = in_begin; w < in_end; ++w) {
        for (uint64_t m = x.words[w]; m != 0; m &= m - 1) {
            const uint32_t owner =
                pivot_owner_[w * 64 + static_cast<uint32_t>(std::countr_zero(m))];
            if (owner != kNoOwner) {
                xor_owner(owner);
            }
        }
        for (uint64_t m = z.words[w]; m != 0; m &= m - 1) {
            const uint32_t owner =
                pivot_owner_[domain + w * 64 + static_cast<uint32_t>(std::countr_zero(m))];
            if (owner != kNoOwner) {
                xor_owner(owner);
            }
        }
    }

    while (se > sb && (scratch_x_[se - 1] | scratch_z_[se - 1]) == 0) {
        --se;
    }
    while (sb < se && (scratch_x_[sb] | scratch_z_[sb]) == 0) {
        ++sb;
    }
    if (sb < se) {
        scratch_begin_ = sb;
        scratch_end_ = se;
    }
}

bool DormantSubspace::intersect(MaskView x, MaskView z, UndoLog* undo) {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    if (dimension_ == 0) {
        return false;
    }

    const uint32_t active_rw = (dimension_ + 63) / 64;
    compute_anticommuting_rows(x, z, std::span<uint64_t>(anti_scratch_.data(), active_rw));

    std::optional<uint32_t> pivot_row;
    for (uint32_t rw = 0; rw < active_rw; ++rw) {
        for (uint64_t m = anti_scratch_[rw]; m != 0; m &= m - 1) {
            const uint32_t i = rw * 64 + static_cast<uint32_t>(std::countr_zero(m));
            if (!pivot_row.has_value() || pivot_[i] > pivot_[*pivot_row]) {
                pivot_row = i;
            }
        }
    }
    if (!pivot_row.has_value()) {
        return false;
    }

    // Using the largest pivot preserves the leading bits of all remaining
    // rows when we cancel their anticommutation with (x, z).
    const uint32_t k = *pivot_row;
    anti_scratch_[k / 64] &= ~(uint64_t{1} << (k % 64));
    const std::span<const uint64_t> anti_without_k(anti_scratch_.data(), active_rw);

    const uint64_t* kx = rows_x_.data() + static_cast<size_t>(k) * words_per_row_;
    const uint64_t* kz = rows_z_.data() + static_cast<size_t>(k) * words_per_row_;
    const uint32_t kb = row_begin_[k];
    const uint32_t ke = row_end_[k];

    if (undo != nullptr) {
        const size_t offset = undo->words_.size();
        undo->entries_.push_back(UndoLog::Entry{
            .kind = UndoLog::Kind::Intersect,
            .row = k,
            .saved_pivot = pivot_[k],
            .saved_begin = kb,
            .saved_end = ke,
            .active_row_words = active_rw,
            .word_offset = offset,
        });
        undo->words_.insert(undo->words_.end(), kx + kb, kx + ke);
        undo->words_.insert(undo->words_.end(), kz + kb, kz + ke);
        undo->words_.insert(undo->words_.end(), anti_without_k.begin(), anti_without_k.end());
    }

    xor_rows(kx, kz, kb, ke, anti_without_k);
    erase_row(k);
    return true;
}

void DormantSubspace::insert_reduced(UndoLog* undo) {
    assert(dimension_ < num_qubits_ && "S is already Lagrangian; nothing can extend it");
    assert(scratch_begin_ < scratch_end_ && "insert_reduced requires a nonzero vector");

    const uint32_t domain = combined_domain(words_per_row_);
    const uint32_t b =
        combined_lowest_bit_in_range(scratch_x_, scratch_z_, scratch_begin_, scratch_end_, domain);
    assert(b < 2 * domain && "insert_reduced requires a nonzero vector");

    const uint32_t active_rw = (dimension_ + 63) / 64;
    if (active_rw > 0) {
        const uint64_t* src_col =
            (b < domain) ? (col_x_.data() + static_cast<size_t>(b) * words_per_row_)
                         : (col_z_.data() + static_cast<size_t>(b - domain) * words_per_row_);
        std::copy_n(src_col, active_rw, row_mask_scratch_.data());
    }
    const std::span<const uint64_t> target_rows(row_mask_scratch_.data(), active_rw);

    if (undo != nullptr) {
        const size_t offset = undo->words_.size();
        undo->entries_.push_back(UndoLog::Entry{
            .kind = UndoLog::Kind::Insert,
            .active_row_words = active_rw,
            .word_offset = offset,
        });
        undo->words_.insert(undo->words_.end(), target_rows.begin(), target_rows.end());
    }

    xor_rows(scratch_x_.data(), scratch_z_.data(), scratch_begin_, scratch_end_, target_rows);

    const uint32_t idx = dimension_;
    uint64_t* dst_x = rows_x_.data() + static_cast<size_t>(idx) * words_per_row_;
    uint64_t* dst_z = rows_z_.data() + static_cast<size_t>(idx) * words_per_row_;
    std::copy(scratch_x_.begin() + scratch_begin_, scratch_x_.begin() + scratch_end_,
              dst_x + scratch_begin_);
    std::copy(scratch_z_.begin() + scratch_begin_, scratch_z_.begin() + scratch_end_,
              dst_z + scratch_begin_);
    toggle_row_in_cols(idx, dst_x, dst_z, scratch_begin_, scratch_end_);
    row_begin_[idx] = scratch_begin_;
    row_end_[idx] = scratch_end_;
    pivot_[idx] = b;
    pivot_owner_[b] = idx;
    ++dimension_;
}

void DormantSubspace::rollback(UndoLog& undo) {
    for (auto it = undo.entries_.rbegin(); it != undo.entries_.rend(); ++it) {
        const UndoLog::Entry& e = *it;
        if (e.kind == UndoLog::Kind::Intersect) {
            const uint32_t span_len = e.saved_end - e.saved_begin;
            const uint64_t* saved_x = undo.words_.data() + e.word_offset;
            const uint64_t* saved_z = saved_x + span_len;
            const std::span<const uint64_t> anti_without_k(saved_z + span_len, e.active_row_words);
            restore_erased_row(e.row, e.saved_pivot, e.saved_begin, e.saved_end, saved_x, saved_z);
            const uint64_t* kx = rows_x_.data() + static_cast<size_t>(e.row) * words_per_row_;
            const uint64_t* kz = rows_z_.data() + static_cast<size_t>(e.row) * words_per_row_;
            xor_rows(kx, kz, e.saved_begin, e.saved_end, anti_without_k);
        } else {
            const uint32_t idx = dimension_ - 1;
            uint64_t* rx = rows_x_.data() + static_cast<size_t>(idx) * words_per_row_;
            uint64_t* rz = rows_z_.data() + static_cast<size_t>(idx) * words_per_row_;
            const uint32_t rb = row_begin_[idx];
            const uint32_t re = row_end_[idx];
            const std::span<const uint64_t> target_rows(undo.words_.data() + e.word_offset,
                                                        e.active_row_words);
            xor_rows(rx, rz, rb, re, target_rows);
            pivot_owner_[pivot_[idx]] = kNoOwner;
            toggle_row_in_cols(idx, rx, rz, rb, re);
            std::fill(rx + rb, rx + re, uint64_t{0});
            std::fill(rz + rb, rz + re, uint64_t{0});
            row_begin_[idx] = 0;
            row_end_[idx] = 0;
            --dimension_;
        }
    }
    undo.clear();
}

bool DormantSubspace::apply_rotation(MaskView x, MaskView z, UndoLog* undo) {
    return intersect(x, z, undo);
}

DormantSubspace::MeasurementEffect DormantSubspace::apply_measurement(MaskView x, MaskView z,
                                                                      UndoLog* undo) {
    if (intersect(x, z, undo)) {
        // The body anticommuted with the old S, so it is independent of the
        // remaining generators and its reduced remainder is nonzero.
        reduce_into_scratch(x, z);
        insert_reduced(undo);
        return MeasurementEffect::DormantRandom;
    }
    reduce_into_scratch(x, z);
    if (scratch_begin_ == scratch_end_) {
        return MeasurementEffect::Classical;
    }
    insert_reduced(undo);
    return MeasurementEffect::Active;
}

std::vector<PauliString> DormantSubspace::generators() const {
    std::vector<PauliString> result;
    result.reserve(dimension_);
    for (uint32_t i = 0; i < dimension_; ++i) {
        PauliString generator(num_qubits_);
        generator.mut_x().xor_with(row_x(i));
        generator.mut_z().xor_with(row_z(i));
        result.push_back(std::move(generator));
    }
    return result;
}

bool DormantSubspace::contains(MaskView x, MaskView z) const {
    assert(x.num_words() == words_per_row_ && z.num_words() == words_per_row_ &&
           "Pauli body must share the subspace's word width");
    reduce_into_scratch(x, z);
    return scratch_begin_ == scratch_end_;
}

void DormantSubspace::verify_invariants() const {
    if (dimension_ > num_qubits_) {
        throw std::logic_error("DormantSubspace: dimension exceeds num_qubits");
    }
    const uint32_t domain = combined_domain(words_per_row_);
    for (uint32_t i = 0; i < num_qubits_; ++i) {
        const MaskView rx = row_x(i);
        const MaskView rz = row_z(i);
        if (i >= dimension_) {
            if (row_begin_[i] != 0 || row_end_[i] != 0 || !rx.is_zero() || !rz.is_zero()) {
                throw std::logic_error("DormantSubspace: inactive row slot is nonzero");
            }
            continue;
        }
        if (row_begin_[i] >= row_end_[i] || row_end_[i] > words_per_row_) {
            throw std::logic_error("DormantSubspace: invalid active row word bounds");
        }
        if ((rx.words[row_begin_[i]] | rz.words[row_begin_[i]]) == 0 ||
            (rx.words[row_end_[i] - 1] | rz.words[row_end_[i] - 1]) == 0) {
            throw std::logic_error("DormantSubspace: active row word bounds are not tight");
        }
        for (uint32_t w = 0; w < words_per_row_; ++w) {
            if ((w < row_begin_[i] || w >= row_end_[i]) && (rx.words[w] | rz.words[w]) != 0) {
                throw std::logic_error("DormantSubspace: nonzero word outside row bounds");
            }
        }
        const uint32_t lowest =
            rx.lowest_bit() < domain ? rx.lowest_bit() : domain + rz.lowest_bit();
        if (pivot_[i] != lowest || pivot_owner_[pivot_[i]] != i) {
            throw std::logic_error("DormantSubspace: pivot or pivot_owner mismatch");
        }
        for (uint32_t j = 0; j < dimension_; ++j) {
            if (j != i && combined_bit_get(row_x(j), row_z(j), domain, pivot_[i])) {
                throw std::logic_error("DormantSubspace: pivot bit not cleared in another row");
            }
        }
    }

    uint32_t owned_pivots = 0;
    for (uint32_t o : pivot_owner_) {
        if (o != kNoOwner) {
            ++owned_pivots;
        }
    }
    if (owned_pivots != dimension_) {
        throw std::logic_error("DormantSubspace: stale entry in pivot_owner");
    }

    for (uint32_t q = 0; q < domain; ++q) {
        const MaskView cx{std::span<const uint64_t>(
            col_x_.data() + static_cast<size_t>(q) * words_per_row_, words_per_row_)};
        const MaskView cz{std::span<const uint64_t>(
            col_z_.data() + static_cast<size_t>(q) * words_per_row_, words_per_row_)};
        for (uint32_t r = 0; r < domain; ++r) {
            const bool want_x = r < dimension_ && row_x(r).bit_get(q);
            const bool want_z = r < dimension_ && row_z(r).bit_get(q);
            if (cx.bit_get(r) != want_x || cz.bit_get(r) != want_z) {
                throw std::logic_error("DormantSubspace: column transpose mismatch");
            }
        }
    }

    for (uint32_t w = 0; w < words_per_row_; ++w) {
        if ((w < scratch_begin_ || w >= scratch_end_) && (scratch_x_[w] | scratch_z_[w]) != 0) {
            throw std::logic_error("DormantSubspace: dirty scratch word outside range");
        }
    }
}

WidthTransition classify_and_apply(const HirModule& hir, const HeisenbergOp& op,
                                   DormantSubspace& subspace, DormantSubspace::UndoLog* undo) {
    const uint32_t before = subspace.active_width();
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (subspace.apply_rotation(x, z, undo)) {
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::RotationPromote};
            }
            if (subspace.contains(x, z)) {
                return WidthTransition{before, before, WidthEffect::RotationStabilizer};
            }
            return WidthTransition{before, before, WidthEffect::RotationNeutral};
        }
        case OpType::MEASURE: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            switch (subspace.apply_measurement(x, z, undo)) {
                case DormantSubspace::MeasurementEffect::DormantRandom:
                    return WidthTransition{before, before, WidthEffect::MeasureDormantRandom};
                case DormantSubspace::MeasurementEffect::Classical:
                    return WidthTransition{before, before, WidthEffect::MeasureClassical};
                case DormantSubspace::MeasurementEffect::Active:
                    return WidthTransition{before, subspace.active_width(),
                                           WidthEffect::MeasureActive};
            }
            assert(false && "unreachable DormantSubspace::MeasurementEffect");
            return WidthTransition{before, before, WidthEffect::None};
        }
        case OpType::INSTRUMENT: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (detail::instrument_damping_neglected(hir, op)) {
                if (!subspace.commutes_with_all(x, z)) {
                    return WidthTransition{before, before, WidthEffect::InstrumentDormantTrap};
                }
            } else if (subspace.apply_rotation(x, z, undo)) {
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::InstrumentActivate};
            }
            if (subspace.contains(x, z)) {
                return WidthTransition{before, before, WidthEffect::InstrumentClassical};
            }
            return WidthTransition{before, before, WidthEffect::InstrumentActive};
        }
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::EXP_VAL:
        case OpType::NUM_OP_TYPES:
            return WidthTransition{before, before, WidthEffect::None};
    }
    assert(false && "unreachable OpType");
    return WidthTransition{before, before, WidthEffect::None};
}

ActiveWidthCost summarize_active_width(const HirModule& hir) {
    if (!can_increase_active_width(hir)) {
        return {};
    }
    DormantSubspace subspace(hir.num_qubits);
    ActiveWidthCost cost;
    cost.peak_width = subspace.active_width();
    for (const HeisenbergOp& op : hir.ops) {
        const WidthTransition transition = classify_and_apply(hir, op, subspace);
        cost.peak_width = std::max(cost.peak_width, transition.after);
        cost.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
    return cost;
}

ActiveWidthCost summarize_active_width(const HirModule& hir, std::span<const uint32_t> order) {
    assert(order.size() == hir.ops.size());
    if (!can_increase_active_width(hir)) {
        return {};
    }
    DormantSubspace subspace(hir.num_qubits);
    ActiveWidthCost cost;
    cost.peak_width = subspace.active_width();
    for (uint32_t idx : order) {
        const WidthTransition transition = classify_and_apply(hir, hir.ops[idx], subspace);
        cost.peak_width = std::max(cost.peak_width, transition.after);
        cost.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
    return cost;
}

ActiveWidthTrace analyze_active_width(const HirModule& hir) {
    ActiveWidthTrace trace;
    DormantSubspace subspace(hir.num_qubits);
    trace.initial_width = subspace.active_width();
    trace.peak_width = trace.initial_width;
    trace.transitions.reserve(hir.ops.size());

    for (const HeisenbergOp& op : hir.ops) {
        const WidthTransition transition = classify_and_apply(hir, op, subspace);
        trace.transitions.push_back(transition);
        trace.peak_width = std::max(trace.peak_width, transition.after);
    }

    trace.final_width =
        trace.transitions.empty() ? trace.initial_width : trace.transitions.back().after;
    return trace;
}

double estimate_dense_work(const ActiveWidthTrace& trace) {
    double work = 0.0;
    for (const WidthTransition& transition : trace.transitions) {
        work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    }
    return work;
}

}  // namespace clifft
