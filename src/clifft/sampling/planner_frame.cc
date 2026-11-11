#include "clifft/sampling/planner_frame.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace clifft::sampling::internal {

namespace {

PlannerPauli single_x(uint32_t num_qubits, uint32_t q) {
    PlannerPauli result(num_qubits);
    result.set_pauli(q, true, false);
    return result;
}

PlannerPauli single_z(uint32_t num_qubits, uint32_t q) {
    PlannerPauli result(num_qubits);
    result.set_pauli(q, false, true);
    return result;
}

PlannerPauli positive_body_xor(const PlannerPauli& left, const PlannerPauli& right) {
    // Callers only combine canonical generator rows whose nonidentity supports
    // are disjoint. Overlapping supports could contribute a phase that this
    // body-only helper intentionally does not compute.
    PlannerPauli result(left);
    result.mut_x().xor_with(right.x());
    result.mut_z().xor_with(right.z());
    result.set_sign(false);
    return result;
}

bool anticommutes(const PlannerPauli& left, const PlannerPauli& right) {
    return !left.view().commutes(right.view());
}

bool has_x_below(const PlannerPauli& pauli, uint32_t end) {
    const uint32_t complete_words = end / 64;
    for (uint32_t w = 0; w < complete_words; ++w) {
        if (pauli.x().words[w] != 0) {
            return true;
        }
    }
    const uint32_t tail = end % 64;
    return tail != 0 && (pauli.x().words[complete_words] & ((uint64_t{1} << tail) - 1)) != 0;
}

template <typename Callback>
void for_each_set_bit(MaskView mask, uint32_t end, Callback&& callback) {
    for (uint32_t w = 0; w < mask.num_words() && 64 * w < end; ++w) {
        uint64_t pending = mask.words[w];
        if (64 * (w + 1) > end && end % 64 != 0) {
            pending &= (uint64_t{1} << (end % 64)) - 1;
        }
        while (pending != 0) {
            callback(64 * w + std::countr_zero(pending));
            pending &= pending - 1;
        }
    }
}

uint8_t mask_y_phase(MaskView x, MaskView z, uint32_t num_qubits) {
    assert(x.num_words() == z.num_words());
    const uint32_t num_words = (num_qubits + 63) / 64;
    assert(x.num_words() == num_words);
    const uint64_t last_mask =
        num_qubits % 64 == 0 ? ~uint64_t{0} : (uint64_t{1} << (num_qubits % 64)) - 1;
    uint32_t count = 0;
    for (uint32_t w = 0; w < num_words; ++w) {
        uint64_t overlap = x.words[w] & z.words[w];
        if (w + 1 == num_words) {
            overlap &= last_mask;
        }
        count += std::popcount(overlap);
    }
    return static_cast<uint8_t>(count & 3U);
}

}  // namespace

CoordinateFrame::CoordinateFrame(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      num_words_((num_qubits + 63) / 64),
      current_to_initial_(num_qubits),
      ix_to_cx_(static_cast<size_t>(num_qubits) * num_words_, 0),
      ix_to_cz_(ix_to_cx_.size(), 0),
      iz_to_cx_(ix_to_cx_.size(), 0),
      iz_to_cz_(ix_to_cx_.size(), 0),
      round_trip_z_(num_words_, 0) {
    for (uint32_t p = 0; p < num_qubits_; ++p) {
        const size_t entry = static_cast<size_t>(p) * num_words_ + p / 64;
        const uint64_t bit = uint64_t{1} << (p % 64);
        ix_to_cx_[entry] = bit;
        iz_to_cz_[entry] = bit;
    }
}

PlannerPauli CoordinateFrame::to_current(const PlannerPauli& initial) {
    assert(initial.num_qubits() == num_qubits_);
    return to_current(initial.x(), initial.z(), initial.sign());
}

PlannerPauli CoordinateFrame::to_current(MaskView initial_x, MaskView initial_z, bool negative) {
    assert(initial_x.num_words() == num_words_ && initial_z.num_words() == num_words_);
    PlannerPauli current(num_qubits_);
    MutableMaskView cx = current.mut_x();
    MutableMaskView cz = current.mut_z();
    for_each_set_bit(initial_x, num_qubits_, [&](uint32_t p) {
        const size_t offset = static_cast<size_t>(p) * num_words_;
        cx.xor_with(MaskView{std::span<const uint64_t>(ix_to_cx_).subspan(offset, num_words_)});
        cz.xor_with(MaskView{std::span<const uint64_t>(ix_to_cz_).subspan(offset, num_words_)});
    });
    for_each_set_bit(initial_z, num_qubits_, [&](uint32_t p) {
        const size_t offset = static_cast<size_t>(p) * num_words_;
        cx.xor_with(MaskView{std::span<const uint64_t>(iz_to_cx_).subspan(offset, num_words_)});
        cz.xor_with(MaskView{std::span<const uint64_t>(iz_to_cz_).subspan(offset, num_words_)});
    });
    current.set_sign(false);

    // Generator signs affect the sign, not the symplectic coordinates. Forward
    // evaluation recovers the phase difference against the input Hermitian Pauli
    // without allocating a round-trip PauliString.
    uint8_t round_trip_phase = current.phase();
    bool touched = false;
    auto accumulate_generator = [&](PauliStringView generator) {
        touched = true;
        bool crossing = false;
        for (uint32_t w = 0; w < num_words_; ++w) {
            crossing ^= (std::popcount(round_trip_z_[w] & generator.x().words[w]) & 1U) != 0;
            round_trip_z_[w] ^= generator.z().words[w];
        }
        round_trip_phase = (round_trip_phase + generator.phase() + (crossing ? 2U : 0U)) & 3U;
    };
    for_each_set_bit(current.x(), num_qubits_,
                     [&](uint32_t q) { accumulate_generator(current_to_initial_.x_output(q)); });
    for_each_set_bit(current.z(), num_qubits_,
                     [&](uint32_t q) { accumulate_generator(current_to_initial_.z_output(q)); });
    if (touched) {
        std::ranges::fill(round_trip_z_, uint64_t{0});
    }

    const uint8_t initial_phase =
        (mask_y_phase(initial_x, initial_z, num_qubits_) + (negative ? 2U : 0U)) & 3U;
    const uint8_t delta = (round_trip_phase - initial_phase) & 3U;
    assert(delta == 0 || delta == 2);
    if (delta == 2) {
        current.negate();
    }
    return current;
}

PlannerPauli CoordinateFrame::to_initial(const PlannerPauli& current) const {
    return current_to_initial_.apply(current.view());
}

void CoordinateFrame::rebuild_transpose() {
    std::ranges::fill(ix_to_cx_, uint64_t{0});
    std::ranges::fill(ix_to_cz_, uint64_t{0});
    std::ranges::fill(iz_to_cx_, uint64_t{0});
    std::ranges::fill(iz_to_cz_, uint64_t{0});
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        const PauliStringView x_gen = current_to_initial_.x_output(q);
        const PauliStringView z_gen = current_to_initial_.z_output(q);
        xor_x_generator_delta(q, x_gen.x(), x_gen.z());
        xor_z_generator_delta(q, z_gen.x(), z_gen.z());
    }
}

void CoordinateFrame::xor_x_generator_delta(uint32_t q, MaskView dx, MaskView dz) {
    const uint32_t w = q / 64;
    const uint64_t bit = uint64_t{1} << (q % 64);
    for_each_set_bit(dx, num_qubits_, [&](uint32_t p) {
        iz_to_cz_[static_cast<size_t>(p) * num_words_ + w] ^= bit;
    });
    for_each_set_bit(dz, num_qubits_, [&](uint32_t p) {
        ix_to_cz_[static_cast<size_t>(p) * num_words_ + w] ^= bit;
    });
}

void CoordinateFrame::xor_z_generator_delta(uint32_t q, MaskView dx, MaskView dz) {
    const uint32_t w = q / 64;
    const uint64_t bit = uint64_t{1} << (q % 64);
    for_each_set_bit(dx, num_qubits_, [&](uint32_t p) {
        iz_to_cx_[static_cast<size_t>(p) * num_words_ + w] ^= bit;
    });
    for_each_set_bit(dz, num_qubits_, [&](uint32_t p) {
        ix_to_cx_[static_cast<size_t>(p) * num_words_ + w] ^= bit;
    });
}

void CoordinateFrame::set_x_generator(uint32_t q, PauliStringView next) {
    const PauliStringView prev = current_to_initial_.x_output(q);
    xor_x_generator_delta(q, prev.x(), prev.z());
    xor_x_generator_delta(q, next.x(), next.z());
    current_to_initial_.set_x_output(q, next);
}

void CoordinateFrame::set_z_generator(uint32_t q, PauliStringView next) {
    const PauliStringView prev = current_to_initial_.z_output(q);
    xor_z_generator_delta(q, prev.x(), prev.z());
    xor_z_generator_delta(q, next.x(), next.z());
    current_to_initial_.set_z_output(q, next);
}

void CoordinateFrame::multiply_x_generator(uint32_t q, PauliStringView rhs) {
    xor_x_generator_delta(q, rhs.x(), rhs.z());
    current_to_initial_.right_multiply_x_output(q, rhs);
}

void CoordinateFrame::multiply_z_generator(uint32_t q, PauliStringView rhs) {
    xor_z_generator_delta(q, rhs.x(), rhs.z());
    current_to_initial_.right_multiply_z_output(q, rhs);
}

void CoordinateFrame::change_basis(const PlannerTableau& new_basis_in_old_coordinates) {
    PlannerTableau previous(std::move(current_to_initial_));
    assert(new_basis_in_old_coordinates.num_qubits() == previous.num_qubits());
    current_to_initial_ = PlannerTableau(new_basis_in_old_coordinates.num_qubits());
    for (uint32_t q = 0; q < new_basis_in_old_coordinates.num_qubits(); ++q) {
        const PlannerPauli x = previous.apply(new_basis_in_old_coordinates.x_output(q));
        const PlannerPauli z = previous.apply(new_basis_in_old_coordinates.z_output(q));
        current_to_initial_.set_x_output(q, x.view());
        current_to_initial_.set_z_output(q, z.view());
    }
    rebuild_transpose();
    assert(current_to_initial_.satisfies_invariants());
}

// These transformations preserve the symplectic relations by construction.
// A full invariant scan is cubic in Debug builds, so exhaustive equivalence
// tests validate them instead of rescanning the tableau after every event.
void CoordinateFrame::promote_dormant(const PlannerPauli& promoted, uint32_t active_width,
                                      uint32_t dormant_pivot) {
    const uint32_t n = current_to_initial_.num_qubits();
    assert(promoted.num_qubits() == n && active_width < n && dormant_pivot >= active_width &&
           dormant_pivot < n);

    PlannerPauli mapped_promoted = to_initial(promoted);
    PlannerPauli pivot_stabilizer(current_to_initial_.z_output(dormant_pivot));

    // The new promoted pair replaces the selected dormant pair. Every other
    // generator that anticommutes with the promoted Pauli acquires the old
    // pivot stabilizer, preserving the same symplectic basis as the generic
    // frame composition without materializing its identity rows.
    for_each_set_bit(promoted.z(), n, [&](uint32_t q) {
        if (q != dormant_pivot) {
            multiply_x_generator(q, pivot_stabilizer.view());
        }
    });
    for_each_set_bit(promoted.x(), n, [&](uint32_t q) {
        if (q != dormant_pivot) {
            multiply_z_generator(q, pivot_stabilizer.view());
        }
    });

    for (uint32_t q = dormant_pivot; q > active_width; --q) {
        set_x_generator(q, current_to_initial_.x_output(q - 1));
        set_z_generator(q, current_to_initial_.z_output(q - 1));
    }
    set_x_generator(active_width, mapped_promoted.view());
    set_z_generator(active_width, pivot_stabilizer.view());
}

void CoordinateFrame::measure_dormant(const PlannerPauli& measured, uint32_t dormant_pivot) {
    const uint32_t n = current_to_initial_.num_qubits();
    assert(measured.num_qubits() == n && dormant_pivot < n);

    PlannerPauli mapped_measured = to_initial(measured);
    // Avoid an allocation per measurement: updates skip the pivot Z row, which
    // is copied into X before the final write overwrites that source row.
    const PauliStringView pivot_stabilizer = current_to_initial_.z_output(dormant_pivot);

    for_each_set_bit(measured.z(), n, [&](uint32_t q) {
        if (q != dormant_pivot) {
            multiply_x_generator(q, pivot_stabilizer);
        }
    });
    for_each_set_bit(measured.x(), n, [&](uint32_t q) {
        if (q != dormant_pivot) {
            multiply_z_generator(q, pivot_stabilizer);
        }
    });
    set_x_generator(dormant_pivot, pivot_stabilizer);
    set_z_generator(dormant_pivot, mapped_measured.view());
}

void CoordinateFrame::measure_active(const PlannerPauli& measured, uint32_t active_width,
                                     uint32_t pivot) {
    assert(measured.num_qubits() == current_to_initial_.num_qubits() && active_width > 0 &&
           active_width <= current_to_initial_.num_qubits() && pivot < active_width);

    const bool diagonal = !has_x_below(measured, active_width);
    PlannerPauli mapped_measured = to_initial(measured);
    PlannerPauli pivot_conjugate(diagonal ? current_to_initial_.x_output(pivot)
                                          : current_to_initial_.z_output(pivot));

    if (!diagonal) {
        for_each_set_bit(measured.x(), active_width, [&](uint32_t q) {
            if (q != pivot) {
                multiply_z_generator(q, pivot_conjugate.view());
            }
        });
    }
    for_each_set_bit(measured.z(), active_width, [&](uint32_t q) {
        if (q != pivot) {
            multiply_x_generator(q, pivot_conjugate.view());
        }
    });

    for (uint32_t q = pivot; q + 1 < active_width; ++q) {
        set_x_generator(q, current_to_initial_.x_output(q + 1));
        set_z_generator(q, current_to_initial_.z_output(q + 1));
    }
    set_x_generator(active_width - 1, pivot_conjugate.view());
    set_z_generator(active_width - 1, mapped_measured.view());
}

SymbolicPauliFrame::SymbolicPauliFrame(uint32_t num_qubits, uint32_t num_symbols)
    : num_qubits_(num_qubits),
      num_symbols_(num_symbols),
      x_rows_(num_qubits),
      z_rows_(num_qubits),
      x_constants_(num_qubits, 0),
      z_constants_(num_qubits, 0),
      scratch_blocks_(((static_cast<size_t>(num_symbols) + 63) / 64 + kBlockWords - 1) /
                      kBlockWords),
      dirty_groups_((scratch_blocks_.size() + 63) / 64, 0) {}

void SymbolicPauliFrame::apply(const PlannerPauli& correction, const AffineBool& condition) {
    assert(correction.num_qubits() == num_qubits_ &&
           "symbolic Pauli correction width must match the planner frame");
    apply(correction.x(), correction.z(), condition);
}

void SymbolicPauliFrame::apply(MaskView correction_x, MaskView correction_z,
                               const AffineBool& condition) {
    const size_t num_words = (static_cast<size_t>(num_qubits_) + 63) / 64;
    assert(correction_x.num_words() == num_words && correction_z.num_words() == num_words &&
           "symbolic Pauli correction width must match the planner frame");

    const auto& terms = condition.terms();
    const bool cond_constant = condition.constant();
    if (terms.empty() && !cond_constant) {
        return;
    }
    cond_blocks_.clear();
    if (!terms.empty()) {
        assert(condition.is_canonical());
        if (index(terms.back()) >= num_symbols_) {
            throw std::logic_error(
                "sampling symbolic Pauli condition contains an out-of-range symbol");
        }
        for (size_t i = 0; i < terms.size();) {
            const uint32_t blk_idx = index(terms[i]) / (64 * kBlockWords);
            BlockEntry entry{blk_idx, {}};
            while (i < terms.size() && index(terms[i]) / (64 * kBlockWords) == blk_idx) {
                const uint32_t sym = index(terms[i]);
                entry.words[(sym / 64) & (kBlockWords - 1)] |= uint64_t{1} << (sym % 64);
                ++i;
            }
            cond_blocks_.push_back(entry);
        }
    }

    // Noise corrections are usually one- or two-qubit Paulis. Iterating their
    // packed support avoids testing every physical qubit for each noise term.
    const uint32_t tail_bits = num_qubits_ % 64;
    const uint64_t last_word_mask = tail_bits == 0 ? ~uint64_t{0} : (uint64_t{1} << tail_bits) - 1;
    for (size_t word = 0; word < num_words; ++word) {
        const uint64_t x_bits = correction_x.words[word];
        const uint64_t z_bits = correction_z.words[word];
        uint64_t support = x_bits | z_bits;
        if (word + 1 == num_words) {
            support &= last_word_mask;
        }
        while (support != 0) {
            const uint32_t bit = std::countr_zero(support);
            const uint32_t qubit = static_cast<uint32_t>(64 * word + bit);
            const uint64_t mask = uint64_t{1} << bit;
            if ((x_bits & mask) != 0) {
                xor_packed_condition(x_rows_[qubit], x_constants_[qubit], cond_constant);
            }
            if ((z_bits & mask) != 0) {
                xor_packed_condition(z_rows_[qubit], z_constants_[qubit], cond_constant);
            }
            support &= support - 1;
        }
    }
}

AffineBool SymbolicPauliFrame::sign_for(const PlannerPauli& observable) {
    assert(observable.num_qubits() == num_qubits_ &&
           "symbolic Pauli observable width must match the planner frame");
    return sign_for(observable.x(), observable.z());
}

AffineBool SymbolicPauliFrame::sign_for(MaskView observable_x, MaskView observable_z) {
    const size_t num_words = (static_cast<size_t>(num_qubits_) + 63) / 64;
    assert(observable_x.num_words() == num_words && observable_z.num_words() == num_words &&
           "symbolic Pauli observable width must match the planner frame");

    bool constant = false;
    hit_rows_.clear();
    const uint32_t tail_bits = num_qubits_ % 64;
    const uint64_t last_word_mask = tail_bits == 0 ? ~uint64_t{0} : (uint64_t{1} << tail_bits) - 1;
    for (size_t word = 0; word < num_words; ++word) {
        const uint64_t x_bits = observable_x.words[word];
        const uint64_t z_bits = observable_z.words[word];
        uint64_t support = x_bits | z_bits;
        if (word + 1 == num_words) {
            support &= last_word_mask;
        }
        while (support != 0) {
            const uint32_t bit = std::countr_zero(support);
            const uint32_t q = static_cast<uint32_t>(64 * word + bit);
            const uint64_t mask = uint64_t{1} << bit;
            if ((x_bits & mask) != 0) {
                constant ^= z_constants_[q] != 0;
                if (!z_rows_[q].empty()) {
                    hit_rows_.push_back(&z_rows_[q]);
                }
            }
            if ((z_bits & mask) != 0) {
                constant ^= x_constants_[q] != 0;
                if (!x_rows_[q].empty()) {
                    hit_rows_.push_back(&x_rows_[q]);
                }
            }
            support &= support - 1;
        }
    }

    if (hit_rows_.empty()) {
        return AffineBool(constant);
    }

    if (hit_rows_.size() == 1) {
        const auto& single = *hit_rows_[0];
        size_t term_count = 0;
        for (const BlockEntry& entry : single) {
            for (uint64_t w : entry.words) {
                term_count += std::popcount(w);
            }
        }
        std::vector<SymbolId> terms;
        terms.reserve(term_count);
        for (const BlockEntry& entry : single) {
            const size_t base_word = static_cast<size_t>(entry.block) * kBlockWords;
            for (uint32_t lane = 0; lane < kBlockWords; ++lane) {
                uint64_t word = entry.words[lane];
                while (word != 0) {
                    const uint32_t bit = std::countr_zero(word);
                    const size_t symbol = (base_word + lane) * 64 + bit;
                    assert(symbol < num_symbols_ &&
                           "symbolic Pauli frame contains an out-of-range term");
                    terms.push_back(SymbolId{static_cast<uint32_t>(symbol)});
                    word &= word - 1;
                }
            }
        }
        return AffineBool::from_canonical_terms(constant, std::move(terms));
    }

    uint32_t min_group = std::numeric_limits<uint32_t>::max();
    uint32_t max_group = 0;
    for (const auto* row : hit_rows_) {
        for (const BlockEntry& entry : *row) {
            const uint32_t group = entry.block >> 6U;
            dirty_groups_[group] |= uint64_t{1} << (entry.block & 63U);
            min_group = std::min(min_group, group);
            max_group = std::max(max_group, group);
            WordBlock& dst = scratch_blocks_[entry.block];
            for (uint32_t lane = 0; lane < kBlockWords; ++lane) {
                dst[lane] ^= entry.words[lane];
            }
        }
    }

    size_t term_count = 0;
    for (uint32_t group = min_group; group <= max_group; ++group) {
        uint64_t dirty = dirty_groups_[group];
        while (dirty != 0) {
            const uint32_t b = (group << 6U) + static_cast<uint32_t>(std::countr_zero(dirty));
            for (uint64_t w : scratch_blocks_[b]) {
                term_count += std::popcount(w);
            }
            dirty &= dirty - 1;
        }
    }
    std::vector<SymbolId> terms;
    terms.reserve(term_count);
    for (uint32_t group = min_group; group <= max_group; ++group) {
        uint64_t dirty = std::exchange(dirty_groups_[group], uint64_t{0});
        while (dirty != 0) {
            const uint32_t b = (group << 6U) + static_cast<uint32_t>(std::countr_zero(dirty));
            const size_t base_word = static_cast<size_t>(b) * kBlockWords;
            WordBlock& blk = scratch_blocks_[b];
            for (uint32_t lane = 0; lane < kBlockWords; ++lane) {
                uint64_t word = std::exchange(blk[lane], uint64_t{0});
                while (word != 0) {
                    const uint32_t bit = std::countr_zero(word);
                    const size_t symbol = (base_word + lane) * 64 + bit;
                    assert(symbol < num_symbols_ &&
                           "symbolic Pauli frame contains an out-of-range term");
                    terms.push_back(SymbolId{static_cast<uint32_t>(symbol)});
                    word &= word - 1;
                }
            }
            dirty &= dirty - 1;
        }
    }
    return AffineBool::from_canonical_terms(constant, std::move(terms));
}

void SymbolicPauliFrame::xor_packed_condition(std::vector<BlockEntry>& row, uint8_t& constant,
                                              bool cond_constant) {
    constant ^= static_cast<uint8_t>(cond_constant);
    if (cond_blocks_.empty()) {
        return;
    }

    auto xor_into_block = [](WordBlock& dst, const WordBlock& src) noexcept {
        uint64_t any = 0;
        for (uint32_t lane = 0; lane < kBlockWords; ++lane) {
            dst[lane] ^= src[lane];
            any |= dst[lane];
        }
        return any != 0;
    };

    size_t search_from = 0;
    for (const BlockEntry& cb : cond_blocks_) {
        if (row.empty() || cb.block > row.back().block) {
            row.push_back(cb);
            search_from = row.size();
            continue;
        }
        if (cb.block == row.back().block) {
            if (!xor_into_block(row.back().words, cb.words)) {
                row.pop_back();
            }
            search_from = row.size();
            continue;
        }
        auto it = std::ranges::lower_bound(row.begin() + static_cast<std::ptrdiff_t>(search_from),
                                           row.end(), cb.block, {}, &BlockEntry::block);
        if (it != row.end() && it->block == cb.block) {
            if (xor_into_block(it->words, cb.words)) {
                search_from = static_cast<size_t>(it - row.begin()) + 1;
            } else {
                const auto next_it = row.erase(it);
                search_from = static_cast<size_t>(next_it - row.begin());
            }
        } else {
            const auto inserted_it = row.insert(it, cb);
            search_from = static_cast<size_t>(inserted_it - row.begin()) + 1;
        }
    }
}

PlannerTableau dormant_promotion_frame(const PlannerPauli& promoted, uint32_t active_width,
                                       uint32_t dormant_pivot) {
    const uint32_t n = promoted.num_qubits();
    PlannerTableau frame(n);
    const PlannerPauli old_stabilizer = single_z(n, dormant_pivot);

    // CoordinateFrame::promote_dormant is the in-place production twin. This
    // builder remains necessary for final-state maps and as its test oracle;
    // keep their pivot ordering and fixups in sync.
    //
    // The rows express the new coordinate generators in the old basis. Making
    // the promoted Pauli the next X generator turns its rotation into a
    // single-coordinate expansion; the stabilizer products keep every other
    // generator in a canonical symplectic pair.
    for (uint32_t q = 0; q < active_width; ++q) {
        PlannerPauli x = single_x(n, q);
        PlannerPauli z = single_z(n, q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.set_x_output(q, x.view());
        frame.set_z_output(q, z.view());
    }

    frame.set_x_output(active_width, promoted.view());
    frame.set_z_output(active_width, old_stabilizer.view());

    uint32_t new_q = active_width + 1;
    for (uint32_t old_q = active_width; old_q < n; ++old_q) {
        if (old_q == dormant_pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, old_q);
        PlannerPauli z = single_z(n, old_q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.set_x_output(new_q, x.view());
        frame.set_z_output(new_q, z.view());
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid promotion frame");
    }
    return frame;
}

PlannerTableau dormant_measurement_frame(const PlannerPauli& measured, uint32_t dormant_pivot) {
    const uint32_t n = measured.num_qubits();
    PlannerTableau frame(n);
    const PlannerPauli old_stabilizer = single_z(n, dormant_pivot);

    // CoordinateFrame::measure_dormant is the in-place production twin. This
    // builder is its independent test oracle; keep their pivot replacement and
    // fixups in sync.
    //
    // Replacing one dormant Z generator with the measured Pauli represents the
    // collapsed eigenspace without changing the dense coefficient state.
    for (uint32_t q = 0; q < n; ++q) {
        if (q == dormant_pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, q);
        PlannerPauli z = single_z(n, q);
        if (anticommutes(x, measured)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, measured)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.set_x_output(q, x.view());
        frame.set_z_output(q, z.view());
    }

    frame.set_x_output(dormant_pivot, old_stabilizer.view());
    frame.set_z_output(dormant_pivot, measured.view());

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid dormant measurement frame");
    }
    return frame;
}

PlannerTableau active_measurement_frame(const PlannerPauli& measured, uint32_t active_width,
                                        uint32_t pivot) {
    const uint32_t n = measured.num_qubits();
    PlannerTableau frame(n);
    const bool diagonal = !has_x_below(measured, active_width);
    const PlannerPauli pivot_x = single_x(n, pivot);
    const PlannerPauli pivot_z = single_z(n, pivot);

    // CoordinateFrame::measure_active is the in-place production twin. This
    // builder is its independent test oracle; keep their pivot compaction and
    // fixups in sync.
    //
    // Moving the measured Pauli to the last active Z generator gives the
    // direct measurement kernel one coordinate to remove. The cumulative
    // frame maps later operations into the remaining packed coordinates.
    frame.set_z_output(active_width - 1, measured.view());
    frame.set_x_output(active_width - 1, diagonal ? pivot_x.view() : pivot_z.view());

    uint32_t new_q = 0;
    for (uint32_t old_q = 0; old_q < active_width; ++old_q) {
        if (old_q == pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, old_q);
        PlannerPauli z = single_z(n, old_q);
        if (diagonal) {
            if (measured.z().bit_get(old_q)) {
                x = positive_body_xor(x, pivot_x);
            }
        } else {
            if (measured.x().bit_get(old_q)) {
                z = positive_body_xor(z, pivot_z);
            }
            if (measured.z().bit_get(old_q)) {
                x = positive_body_xor(x, pivot_z);
            }
        }
        frame.set_x_output(new_q, x.view());
        frame.set_z_output(new_q, z.view());
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid active measurement frame");
    }
    return frame;
}

}  // namespace clifft::sampling::internal
