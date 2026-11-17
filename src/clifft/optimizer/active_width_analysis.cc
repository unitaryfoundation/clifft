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

template <typename VisitOps>
ActiveWidthCost summarize_active_width_impl(const HirModule& hir, VisitOps&& visit_ops) {
    if (!can_increase_active_width(hir)) {
        return {};
    }
    DormantSubspace subspace(hir.num_qubits);
    ActiveWidthCost cost;
    cost.peak_width = subspace.active_width();
    visit_ops([&](const HeisenbergOp& op) {
        const WidthTransition transition = classify_and_apply(hir, op, subspace);
        cost.peak_width = std::max(cost.peak_width, transition.after);
        cost.dense_work +=
            detail::dense_work_contribution(transition.effect, transition.before, transition.after);
    });
    return cost;
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
    const uint64_t zero = 0;
    for (uint32_t q = 0; q < num_qubits; ++q) {
        const uint32_t w = q / 64;
        const uint64_t bit = uint64_t{1} << (q % 64);
        write_row(q, domain + q, w, w + 1, &zero, &bit);
    }
}

void DormantSubspace::compute_anticommuting_rows(PauliMaskView body,
                                                 std::span<uint64_t> anti_out) const {
    if (words_per_row_ == 1) {
        uint64_t anti = 0;
        for (const PauliWord& word : body) {
            for (uint64_t wx = word.x; wx != 0; wx &= wx - 1) {
                anti ^= col_z_[std::countr_zero(wx)];
            }
            for (uint64_t wz = word.z; wz != 0; wz &= wz - 1) {
                anti ^= col_x_[std::countr_zero(wz)];
            }
        }
        anti_out[0] = anti;
        return;
    }

    std::ranges::fill(anti_out, uint64_t{0});
    const size_t active_rw = anti_out.size();
    for (const PauliWord& word : body) {
        for (uint64_t m = word.x; m != 0; m &= m - 1) {
            const uint32_t q = word.index * 64 + static_cast<uint32_t>(std::countr_zero(m));
            const uint64_t* cz = col_z_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                anti_out[rw] ^= cz[rw];
            }
        }
        for (uint64_t m = word.z; m != 0; m &= m - 1) {
            const uint32_t q = word.index * 64 + static_cast<uint32_t>(std::countr_zero(m));
            const uint64_t* cx = col_x_.data() + static_cast<size_t>(q) * words_per_row_;
            for (size_t rw = 0; rw < active_rw; ++rw) {
                anti_out[rw] ^= cx[rw];
            }
        }
    }
}

bool DormantSubspace::commutes_with_all(PauliMaskView body, uint32_t* anticommuting_row) const {
    assert(body.end_word() <= words_per_row_ && "Pauli body is wider than the subspace");
    if (dimension_ == 0) {
        return true;
    }

    if (anticommuting_row && *anticommuting_row < dimension_) {
        const uint32_t r = *anticommuting_row;
        if (anti_commute(body, row_x(r), row_z(r))) {
            return false;
        }
    }

    const uint32_t active_rw = (dimension_ + 63) / 64;
    const std::span<uint64_t> anti(anti_scratch_.data(), active_rw);
    compute_anticommuting_rows(body, anti);
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
    for_each_set_bit(
        MaskView{std::span<const uint64_t>(vx, end_w)}, begin_w, end_w,
        [&](uint32_t q) { col_x_[static_cast<size_t>(q) * words_per_row_ + rw] ^= rbit; });
    for_each_set_bit(
        MaskView{std::span<const uint64_t>(vz, end_w)}, begin_w, end_w,
        [&](uint32_t q) { col_z_[static_cast<size_t>(q) * words_per_row_ + rw] ^= rbit; });
}

void DormantSubspace::xor_rows(const uint64_t* vx, const uint64_t* vz, uint32_t v_begin,
                               uint32_t v_end, std::span<const uint64_t> target_rows) {
    if (v_begin >= v_end) {
        return;
    }
    bool any_target = false;
    for_each_set_bit(MaskView{target_rows}, [&](uint32_t i) {
        any_target = true;
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
    });
    if (!any_target) {
        return;
    }

    const size_t active_rw = target_rows.size();
    for_each_set_bit(MaskView{std::span<const uint64_t>(vx, v_end)}, v_begin, v_end,
                     [&](uint32_t q) {
                         uint64_t* col = col_x_.data() + static_cast<size_t>(q) * words_per_row_;
                         for (size_t rw = 0; rw < active_rw; ++rw) {
                             col[rw] ^= target_rows[rw];
                         }
                     });
    for_each_set_bit(MaskView{std::span<const uint64_t>(vz, v_end)}, v_begin, v_end,
                     [&](uint32_t q) {
                         uint64_t* col = col_z_.data() + static_cast<size_t>(q) * words_per_row_;
                         for (size_t rw = 0; rw < active_rw; ++rw) {
                             col[rw] ^= target_rows[rw];
                         }
                     });
}

void DormantSubspace::write_row(uint32_t slot, uint32_t pivot, uint32_t begin_w, uint32_t end_w,
                                const uint64_t* src_x, const uint64_t* src_z) {
    assert(begin_w < end_w);
    uint64_t* rx = rows_x_.data() + static_cast<size_t>(slot) * words_per_row_;
    uint64_t* rz = rows_z_.data() + static_cast<size_t>(slot) * words_per_row_;
    const uint32_t len = end_w - begin_w;
    std::copy_n(src_x, len, rx + begin_w);
    std::copy_n(src_z, len, rz + begin_w);
    toggle_row_in_cols(slot, rx, rz, begin_w, end_w);
    row_begin_[slot] = begin_w;
    row_end_[slot] = end_w;
    pivot_[slot] = pivot;
    pivot_owner_[pivot] = slot;
}

void DormantSubspace::clear_row(uint32_t slot) {
    uint64_t* rx = rows_x_.data() + static_cast<size_t>(slot) * words_per_row_;
    uint64_t* rz = rows_z_.data() + static_cast<size_t>(slot) * words_per_row_;
    const uint32_t rb = row_begin_[slot];
    const uint32_t re = row_end_[slot];
    pivot_owner_[pivot_[slot]] = kNoOwner;
    toggle_row_in_cols(slot, rx, rz, rb, re);
    std::fill(rx + rb, rx + re, uint64_t{0});
    std::fill(rz + rb, rz + re, uint64_t{0});
    row_begin_[slot] = 0;
    row_end_[slot] = 0;
}

void DormantSubspace::move_row(uint32_t from, uint32_t to) {
    assert(from != to);
    uint64_t* fx = rows_x_.data() + static_cast<size_t>(from) * words_per_row_;
    uint64_t* fz = rows_z_.data() + static_cast<size_t>(from) * words_per_row_;
    const uint32_t fb = row_begin_[from];
    const uint32_t fe = row_end_[from];
    toggle_row_in_cols(from, fx, fz, fb, fe);
    write_row(to, pivot_[from], fb, fe, fx + fb, fz + fb);
    std::fill(fx + fb, fx + fe, uint64_t{0});
    std::fill(fz + fb, fz + fe, uint64_t{0});
    row_begin_[from] = 0;
    row_end_[from] = 0;
}

void DormantSubspace::erase_row(uint32_t k) {
    assert(k < dimension_);
    clear_row(k);
    const uint32_t last = --dimension_;
    if (k != last) {
        move_row(last, k);
    }
}

void DormantSubspace::reduce_into_scratch(PauliMaskView body) const {
    if (scratch_begin_ < scratch_end_) {
        std::fill(scratch_x_.begin() + scratch_begin_, scratch_x_.begin() + scratch_end_,
                  uint64_t{0});
        std::fill(scratch_z_.begin() + scratch_begin_, scratch_z_.begin() + scratch_end_,
                  uint64_t{0});
        scratch_begin_ = 0;
        scratch_end_ = 0;
    }

    if (body.is_identity()) {
        return;
    }

    // Each basis row is zero at every other row's pivot, so only pivots set in
    // the input body participate in its reduction.
    if (words_per_row_ == 1) {
        const PauliWord& word = body.words().front();
        uint64_t rx = word.x;
        uint64_t rz = word.z;
        for (uint64_t m = word.x; m != 0; m &= m - 1) {
            const uint32_t owner = pivot_owner_[std::countr_zero(m)];
            if (owner != kNoOwner) {
                rx ^= rows_x_[owner];
                rz ^= rows_z_[owner];
            }
        }
        for (uint64_t m = word.z; m != 0; m &= m - 1) {
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

    uint32_t sb = words_per_row_;
    uint32_t se = 0;
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

    for (const PauliWord& word : body) {
        const uint32_t w = word.index;
        scratch_x_[w] ^= word.x;
        scratch_z_[w] ^= word.z;
        sb = std::min(sb, w);
        se = std::max(se, w + 1);
        for (uint64_t m = word.x; m != 0; m &= m - 1) {
            const uint32_t owner =
                pivot_owner_[w * 64 + static_cast<uint32_t>(std::countr_zero(m))];
            if (owner != kNoOwner) {
                xor_owner(owner);
            }
        }
        for (uint64_t m = word.z; m != 0; m &= m - 1) {
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

bool DormantSubspace::intersect(PauliMaskView body, UndoLog* undo) {
    assert(body.end_word() <= words_per_row_ && "Pauli body is wider than the subspace");
    if (dimension_ == 0) {
        return false;
    }

    const uint32_t active_rw = (dimension_ + 63) / 64;
    const std::span<uint64_t> anti(anti_scratch_.data(), active_rw);
    compute_anticommuting_rows(body, anti);

    std::optional<uint32_t> pivot_row;
    for_each_set_bit(MaskView{anti}, [&](uint32_t i) {
        if (!pivot_row.has_value() || pivot_[i] > pivot_[*pivot_row]) {
            pivot_row = i;
        }
    });
    if (!pivot_row.has_value()) {
        return false;
    }

    // Using the largest pivot preserves the leading bits of all remaining
    // rows when we cancel their anticommutation with the body.
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
    write_row(dimension_++, b, scratch_begin_, scratch_end_, scratch_x_.data() + scratch_begin_,
              scratch_z_.data() + scratch_begin_);
}

void DormantSubspace::rollback(UndoLog& undo) {
    for (auto it = undo.entries_.rbegin(); it != undo.entries_.rend(); ++it) {
        const UndoLog::Entry& e = *it;
        if (e.kind == UndoLog::Kind::Intersect) {
            const uint32_t span_len = e.saved_end - e.saved_begin;
            const uint64_t* saved_x = undo.words_.data() + e.word_offset;
            const uint64_t* saved_z = saved_x + span_len;
            const std::span<const uint64_t> anti_without_k(saved_z + span_len, e.active_row_words);
            const uint32_t last = dimension_++;
            if (e.row != last) {
                move_row(e.row, last);
            }
            write_row(e.row, e.saved_pivot, e.saved_begin, e.saved_end, saved_x, saved_z);
            const uint64_t* kx = rows_x_.data() + static_cast<size_t>(e.row) * words_per_row_;
            const uint64_t* kz = rows_z_.data() + static_cast<size_t>(e.row) * words_per_row_;
            xor_rows(kx, kz, e.saved_begin, e.saved_end, anti_without_k);
        } else {
            const uint32_t idx = dimension_ - 1;
            const std::span<const uint64_t> target_rows(undo.words_.data() + e.word_offset,
                                                        e.active_row_words);
            xor_rows(rows_x_.data() + static_cast<size_t>(idx) * words_per_row_,
                     rows_z_.data() + static_cast<size_t>(idx) * words_per_row_, row_begin_[idx],
                     row_end_[idx], target_rows);
            erase_row(idx);
        }
    }
    undo.clear();
}

bool DormantSubspace::apply_rotation(PauliMaskView axis, UndoLog* undo) {
    return intersect(axis, undo);
}

DormantSubspace::MeasurementEffect DormantSubspace::apply_measurement(PauliMaskView body,
                                                                      UndoLog* undo) {
    if (intersect(body, undo)) {
        // The body anticommuted with the old S, so it is independent of the
        // remaining generators and its reduced remainder is nonzero.
        reduce_into_scratch(body);
        insert_reduced(undo);
        return MeasurementEffect::DormantRandom;
    }
    reduce_into_scratch(body);
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

bool DormantSubspace::contains(PauliMaskView body) const {
    assert(body.end_word() <= words_per_row_ && "Pauli body is wider than the subspace");
    reduce_into_scratch(body);
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
            const PauliMaskView axis = hir.mask_view(op);
            if (subspace.apply_rotation(axis, undo)) {
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::RotationPromote};
            }
            if (subspace.contains(axis)) {
                return WidthTransition{before, before, WidthEffect::RotationStabilizer};
            }
            return WidthTransition{before, before, WidthEffect::RotationNeutral};
        }
        case OpType::MEASURE: {
            switch (subspace.apply_measurement(hir.mask_view(op), undo)) {
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
            const PauliMaskView body = hir.mask_view(op);
            if (detail::instrument_damping_neglected(hir, op)) {
                if (!subspace.commutes_with_all(body)) {
                    return WidthTransition{before, before, WidthEffect::InstrumentDormantTrap};
                }
            } else if (subspace.apply_rotation(body, undo)) {
                return WidthTransition{before, subspace.active_width(),
                                       WidthEffect::InstrumentActivate};
            }
            if (subspace.contains(body)) {
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
    return summarize_active_width_impl(hir, [&](auto&& step) {
        for (const HeisenbergOp& op : hir.ops) {
            step(op);
        }
    });
}

ActiveWidthCost summarize_active_width(const HirModule& hir, std::span<const uint32_t> order) {
    assert(order.size() == hir.ops.size());
    return summarize_active_width_impl(hir, [&](auto&& step) {
        for (uint32_t idx : order) {
            step(hir.ops[idx]);
        }
    });
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
