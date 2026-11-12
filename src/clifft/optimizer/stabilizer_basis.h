#pragma once

// Hybrid stabilizer-group basis shared by KnownStabilizers (fixed boolean signs)
// and SymbolicStabilizers (affine symbolic signs).
//
// Fixed-sign single-qubit +-Z_q generators are packed into two 64-bit masks per
// 64 qubits (`unit_z_mask_` and `unit_z_sign_`), while multi-qubit, X/Y, and
// symbolically signed generators are stored in `rows_` sorted by ascending
// symplectic pivot (`X` bits `[0, domain)` before `Z` bits `[domain, 2*domain)`).
// Because unit-Z pivots lie in `[domain, 2*domain)` and never coincide with a
// stored row's pivot, row reduction runs a single pass over `rows_` followed by
// a single word-wise `unit_z_mask_` sweep.

#include "clifft/optimizer/pauli_axis.h"
#include "clifft/sampling/plan.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace clifft::optimizer_detail {

struct FixedSign {};

template <typename Sign>
struct StabilizerSignTraits;

template <>
struct StabilizerSignTraits<FixedSign> {
    [[nodiscard]] static constexpr bool has_terms(const FixedSign&) noexcept { return false; }
    [[nodiscard]] static constexpr bool constant(const FixedSign&) noexcept { return false; }
    [[nodiscard]] static constexpr size_t term_count(const FixedSign&) noexcept { return 0; }
    static constexpr void xor_sign(FixedSign&, const FixedSign&) noexcept {}
};

template <>
struct StabilizerSignTraits<sampling::AffineBool> {
    [[nodiscard]] static bool has_terms(const sampling::AffineBool& sign) noexcept {
        return !sign.terms().empty();
    }
    [[nodiscard]] static bool constant(const sampling::AffineBool& sign) noexcept {
        return sign.constant();
    }
    [[nodiscard]] static size_t term_count(const sampling::AffineBool& sign) noexcept {
        return sign.terms().size();
    }
    static void xor_sign(sampling::AffineBool& dst, const sampling::AffineBool& src) { dst ^= src; }
};

template <typename Sign>
class StabilizerBasis {
  public:
    using Traits = StabilizerSignTraits<Sign>;

    struct Row {
        uint32_t pivot = 0;
        PauliString axis;
        Sign sign{};
        // Half-open mask word range containing every nonzero word of `axis`.
        // Products widen it without rescanning, so it may over-approximate.
        uint32_t begin = 0;
        uint32_t end = 0;

        [[nodiscard]] bool misses(const PauliCommutation& query) const noexcept {
            return end <= query.begin_word() || query.end_word() <= begin;
        }
    };

    StabilizerBasis() = default;

    explicit StabilizerBasis(uint32_t num_qubits)
        : num_qubits_(num_qubits), unit_z_count_(num_qubits) {
        const uint32_t num_words = (num_qubits + 63) / 64;
        unit_z_mask_.assign(num_words, ~uint64_t{0});
        if (num_qubits % 64 != 0) {
            unit_z_mask_.back() = (uint64_t{1} << (num_qubits % 64)) - 1;
        }
        unit_z_sign_.assign(num_words, 0);
    }

    [[nodiscard]] static bool identity_axis(const PauliString& axis) noexcept {
        return axis.x().is_zero() && axis.z().is_zero();
    }

    [[nodiscard]] bool empty() const noexcept { return unit_z_count_ == 0 && rows_.empty(); }
    [[nodiscard]] uint32_t num_qubits() const noexcept { return num_qubits_; }
    [[nodiscard]] size_t unit_z_count() const noexcept { return unit_z_count_; }
    [[nodiscard]] const std::vector<uint64_t>& unit_z_mask() const noexcept { return unit_z_mask_; }
    [[nodiscard]] const std::vector<uint64_t>& unit_z_sign() const noexcept { return unit_z_sign_; }
    [[nodiscard]] const std::vector<Row>& rows() const noexcept { return rows_; }
    [[nodiscard]] std::vector<Row>& mut_rows() noexcept { return rows_; }

    void clear() noexcept {
        unit_z_count_ = 0;
        unit_z_mask_.clear();
        unit_z_sign_.clear();
        rows_.clear();
    }

    void ensure_width(uint32_t num_qubits) {
        if (unit_z_mask_.empty() && num_qubits > 0) {
            num_qubits_ = num_qubits;
            const uint32_t num_words = (num_qubits + 63) / 64;
            unit_z_mask_.assign(num_words, 0);
            unit_z_sign_.assign(num_words, 0);
        }
    }

    template <typename OtherSign>
    void reset_unit_z_from(const StabilizerBasis<OtherSign>& other) {
        num_qubits_ = other.num_qubits();
        unit_z_count_ = other.unit_z_count();
        if (unit_z_count_ == 0) {
            unit_z_mask_.clear();
            unit_z_sign_.clear();
        } else {
            unit_z_mask_ = other.unit_z_mask();
            unit_z_sign_ = other.unit_z_sign();
        }
        rows_.clear();
    }

    void set_unit_z(uint32_t word, uint64_t bit, bool negative) {
        ensure_width(num_qubits_);
        assert(std::has_single_bit(bit));
        assert((unit_z_mask_[word] & bit) == 0);
        unit_z_mask_[word] |= bit;
        if (negative) {
            unit_z_sign_[word] |= bit;
        } else {
            unit_z_sign_[word] &= ~bit;
        }
        ++unit_z_count_;
    }

    uint64_t flip_unit_z_signs(uint32_t word, uint64_t mask) noexcept {
        const uint64_t hits = mask & unit_z_mask_[word];
        unit_z_sign_[word] ^= hits;
        return hits;
    }

    uint64_t take_unit_z_hits(uint32_t word, uint64_t mask, uint64_t& signs_out) noexcept {
        const uint64_t hits = mask & unit_z_mask_[word];
        signs_out = unit_z_sign_[word] & hits;
        unit_z_mask_[word] &= ~hits;
        unit_z_sign_[word] &= ~hits;
        unit_z_count_ -= static_cast<size_t>(std::popcount(hits));
        return hits;
    }

    void insert_row_at_pivot(Row row) {
        const auto it = std::ranges::lower_bound(rows_, row.pivot, {}, &Row::pivot);
        assert(it == rows_.end() || it->pivot != row.pivot);
        rows_.insert(it, std::move(row));
    }

    void insert_reduced(Row row) {
        ensure_width(row.axis.num_qubits());
        const PauliCommutation range(row.axis.x(), row.axis.z());
        assert(range.begin_word() < range.end_word());
        const uint32_t w = range.begin_word();
        if (!Traits::has_terms(row.sign) && range.end_word() == w + 1 &&
            row.axis.x().words[w] == 0 && std::has_single_bit(row.axis.z().words[w])) {
            set_unit_z(w, row.axis.z().words[w], row.axis.sign() ^ Traits::constant(row.sign));
            return;
        }
        row.pivot = pivot_of(row.axis);
        row.begin = range.begin_word();
        row.end = range.end_word();
        insert_row_at_pivot(std::move(row));
    }

    [[nodiscard]] PauliString reduce_body(PauliString axis) const {
        if (empty() || identity_axis(axis)) {
            return axis;
        }
        for (auto it = std::ranges::lower_bound(rows_, pivot_of(axis), {}, &Row::pivot);
             it != rows_.end(); ++it) {
            if (body_bit(axis, it->pivot)) {
                axis.mut_x().xor_with(it->axis.x());
                axis.mut_z().xor_with(it->axis.z());
                if (identity_axis(axis)) {
                    return axis;
                }
            }
        }
        if (unit_z_count_ != 0) {
            auto z_words = axis.mut_z().words;
            assert(z_words.size() == unit_z_mask_.size());
            for (size_t w = 0; w < z_words.size(); ++w) {
                z_words[w] &= ~unit_z_mask_[w];
            }
        }
        return axis;
    }

    [[nodiscard]] bool commutes(MaskView ax, MaskView az) const {
        if (empty()) {
            return true;
        }
        const PauliCommutation comm(ax, az);
        if (comm.begin_word() == comm.end_word()) {
            return true;
        }
        if (unit_z_count_ != 0) {
            assert(ax.num_words() == unit_z_mask_.size());
            for (uint32_t w = comm.begin_word(); w < comm.end_word(); ++w) {
                if ((ax.words[w] & unit_z_mask_[w]) != 0) {
                    return false;
                }
            }
        }
        return std::ranges::all_of(rows_, [&](const Row& row) {
            return row.misses(comm) || !comm.anti_commutes(row.axis.x(), row.axis.z());
        });
    }

    bool multiply_row(Row& left, const Row& right, size_t& products,
                      size_t max_products = std::numeric_limits<size_t>::max(),
                      size_t max_terms = std::numeric_limits<size_t>::max()) const {
        if (products == max_products) {
            return false;
        }
        ++products;
        Traits::xor_sign(left.sign, right.sign);
        if (Traits::term_count(left.sign) > max_terms) {
            return false;
        }
        left.axis.right_multiply(right.axis.view());
        left.begin = std::min(left.begin, right.begin);
        left.end = std::max(left.end, right.end);
        return true;
    }

    bool reduce(Row& target, size_t& products,
                size_t max_products = std::numeric_limits<size_t>::max(),
                size_t max_terms = std::numeric_limits<size_t>::max()) const {
        if (empty() || identity_axis(target.axis)) {
            return true;
        }
        const uint32_t domain = target.axis.x().num_words() * 64;
        const uint32_t p0 = pivot_of(target.axis);
        for (auto it = std::ranges::lower_bound(rows_, p0, {}, &Row::pivot); it != rows_.end();
             ++it) {
            if (body_bit(target.axis, it->pivot)) {
                if (!multiply_row(target, *it, products, max_products, max_terms)) {
                    return false;
                }
                if (identity_axis(target.axis)) {
                    return true;
                }
            }
        }
        if (unit_z_count_ == 0) {
            return true;
        }
        auto z_words = target.axis.mut_z().words;
        assert(z_words.size() == unit_z_mask_.size());
        bool sign_flip = false;
        for (size_t w = (p0 > domain ? (p0 - domain) / 64 : 0); w < z_words.size(); ++w) {
            const uint64_t killed = z_words[w] & unit_z_mask_[w];
            if (killed == 0) {
                continue;
            }
            const size_t cnt = static_cast<size_t>(std::popcount(killed));
            if (cnt > max_products - products) {
                products = max_products;
                return false;
            }
            products += cnt;
            sign_flip ^= (std::popcount(killed & unit_z_sign_[w]) & 1U) != 0;
            z_words[w] ^= killed;
        }
        if (sign_flip) {
            target.axis.add_phase(2);
        }
        return true;
    }

    bool intersect(MaskView ax, MaskView az,
                   size_t max_products = std::numeric_limits<size_t>::max(),
                   size_t max_terms = std::numeric_limits<size_t>::max(),
                   bool* modified = nullptr) {
        if (empty()) {
            return true;
        }
        const PauliCommutation comm(ax, az);
        if (comm.begin_word() == comm.end_word()) {
            return true;
        }

        size_t row_hit_count = 0;
        size_t last_row_hit = 0;
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (!rows_[i].misses(comm) &&
                comm.anti_commutes(rows_[i].axis.x(), rows_[i].axis.z())) {
                ++row_hit_count;
                last_row_hit = i;
            }
        }

        size_t unit_hit_count = 0;
        uint32_t max_unit_word = 0;
        if (unit_z_count_ != 0) {
            assert(ax.num_words() == unit_z_mask_.size());
            for (uint32_t w = comm.begin_word(); w < comm.end_word(); ++w) {
                const uint64_t hits = ax.words[w] & unit_z_mask_[w];
                if (hits != 0) {
                    unit_hit_count += static_cast<size_t>(std::popcount(hits));
                    max_unit_word = w;
                }
            }
        }

        const size_t total_hits = row_hit_count + unit_hit_count;
        if (total_hits == 0) {
            return true;
        }
        if (modified != nullptr) {
            *modified = true;
        }
        if (total_hits == 1) {
            if (unit_hit_count == 1) {
                uint64_t unused_signs = 0;
                take_unit_z_hits(max_unit_word, ax.words[max_unit_word], unused_signs);
            } else {
                rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(last_row_hit));
            }
            return true;
        }
        if (total_hits - 1 > max_products) {
            return false;
        }

        const uint32_t domain = ax.num_words() * 64;
        uint32_t q_max = 0;
        uint32_t max_unit_pivot = 0;
        if (unit_hit_count != 0) {
            const uint64_t top_hits = ax.words[max_unit_word] & unit_z_mask_[max_unit_word];
            const uint32_t top_bit = 63U - static_cast<uint32_t>(std::countl_zero(top_hits));
            q_max = max_unit_word * 64U + top_bit;
            max_unit_pivot = domain + q_max;
        }

        // Multiplying earlier hit rows by the largest anticommuting pivot
        // preserves their leading pivots and row-echelon order.
        if (unit_hit_count != 0 &&
            (row_hit_count == 0 || max_unit_pivot > rows_[last_row_hit].pivot)) {
            const uint64_t max_mask = uint64_t{1} << (q_max % 64U);
            uint64_t max_sign_bits = 0;
            take_unit_z_hits(max_unit_word, max_mask, max_sign_bits);
            const bool s_max = max_sign_bits != 0;

            if (row_hit_count != 0) {
                for (size_t i = 0; i <= last_row_hit; ++i) {
                    if (!rows_[i].misses(comm) &&
                        comm.anti_commutes(rows_[i].axis.x(), rows_[i].axis.z())) {
                        assert(!rows_[i].axis.x().bit_get(q_max));
                        rows_[i].axis.mut_z().bit_xor(q_max);
                        if (s_max) {
                            rows_[i].axis.add_phase(2);
                        }
                        rows_[i].begin = std::min(rows_[i].begin, max_unit_word);
                        rows_[i].end = std::max(rows_[i].end, max_unit_word + 1);
                    }
                }
            }

            for (uint32_t w = comm.begin_word(); w <= max_unit_word; ++w) {
                uint64_t signs = 0;
                uint64_t hits = take_unit_z_hits(w, ax.words[w], signs);
                while (hits != 0) {
                    const uint32_t b = static_cast<uint32_t>(std::countr_zero(hits));
                    const uint32_t q = w * 64U + b;
                    const bool s_q = ((signs >> b) & 1U) != 0;
                    PauliString pair(num_qubits_);
                    pair.set_pauli(q, false, true);
                    pair.set_pauli(q_max, false, true);
                    if (s_q ^ s_max) {
                        pair.set_sign(true);
                    }
                    insert_row_at_pivot(
                        Row{domain + q, std::move(pair), Sign{}, w, max_unit_word + 1});
                    hits &= hits - 1;
                }
            }
            return true;
        }

        size_t products = 0;
        for (size_t i = 0; i < last_row_hit; ++i) {
            if (!rows_[i].misses(comm) &&
                comm.anti_commutes(rows_[i].axis.x(), rows_[i].axis.z()) &&
                !multiply_row(rows_[i], rows_[last_row_hit], products, max_products, max_terms)) {
                return false;
            }
        }
        Row pivot_row = std::move(rows_[last_row_hit]);
        rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(last_row_hit));

        if (unit_hit_count != 0) {
            assert(Traits::term_count(pivot_row.sign) <= max_terms);
            for (uint32_t w = comm.begin_word(); w <= max_unit_word; ++w) {
                uint64_t signs = 0;
                uint64_t hits = take_unit_z_hits(w, ax.words[w], signs);
                while (hits != 0) {
                    const uint32_t b = static_cast<uint32_t>(std::countr_zero(hits));
                    const uint32_t q = w * 64U + b;
                    const bool s_q = ((signs >> b) & 1U) != 0;
                    // Since pivot_row.pivot > domain + q >= domain, pivot_row.axis has no X
                    // support.
                    PauliString prod = pivot_row.axis;
                    prod.mut_z().bit_xor(q);
                    if (s_q) {
                        prod.add_phase(2);
                    }
                    insert_row_at_pivot(Row{domain + q, std::move(prod), pivot_row.sign,
                                            std::min(w, pivot_row.begin),
                                            std::max(w + 1, pivot_row.end)});
                    hits &= hits - 1;
                }
            }
        }
        return true;
    }

  private:
    uint32_t num_qubits_ = 0;
    size_t unit_z_count_ = 0;
    std::vector<uint64_t> unit_z_mask_;
    std::vector<uint64_t> unit_z_sign_;
    std::vector<Row> rows_;
};

}  // namespace clifft::optimizer_detail
