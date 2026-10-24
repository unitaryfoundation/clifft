#pragma once

// Dynamic GF(2) vectors and an echelon basis, for the PBC reader and the
// scheduler. Clifft's MaskView and PauliString hold Pauli masks over qubits;
// the reader also eliminates over spiders, outcome parameters and noise bits
// together, whose vectors are neither Paulis nor bounded by the qubit count.

#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace clifft::zx {

class Bits {
  public:
    Bits() = default;
    explicit Bits(size_t size) : size_(size), words_((size + 63) / 64, 0) {}

    [[nodiscard]] static Bits single(size_t size, size_t bit) {
        Bits b(size);
        b.set(bit);
        return b;
    }

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] bool test(size_t i) const { return (words_[i / 64] >> (i % 64)) & 1; }
    void set(size_t i, bool value = true) {
        const uint64_t mask = uint64_t{1} << (i % 64);
        words_[i / 64] = value ? (words_[i / 64] | mask) : (words_[i / 64] & ~mask);
    }
    void flip(size_t i) { words_[i / 64] ^= uint64_t{1} << (i % 64); }

    Bits& operator^=(const Bits& other) {
        assert(size_ == other.size_);
        for (size_t w = 0; w < words_.size(); ++w) {
            words_[w] ^= other.words_[w];
        }
        return *this;
    }
    Bits& operator|=(const Bits& other) {
        assert(size_ == other.size_);
        for (size_t w = 0; w < words_.size(); ++w) {
            words_[w] |= other.words_[w];
        }
        return *this;
    }
    [[nodiscard]] Bits operator&(const Bits& other) const {
        Bits out = *this;
        for (size_t w = 0; w < words_.size(); ++w) {
            out.words_[w] &= other.words_[w];
        }
        return out;
    }

    [[nodiscard]] bool any() const {
        for (uint64_t w : words_) {
            if (w) {
                return true;
            }
        }
        return false;
    }
    [[nodiscard]] size_t popcount() const {
        size_t n = 0;
        for (uint64_t w : words_) {
            n += static_cast<size_t>(std::popcount(w));
        }
        return n;
    }
    [[nodiscard]] bool parity_with(const Bits& other) const {
        uint64_t acc = 0;
        for (size_t w = 0; w < words_.size(); ++w) {
            acc ^= words_[w] & other.words_[w];
        }
        return (std::popcount(acc) & 1) != 0;
    }
    // The highest set bit, or nullopt when zero.
    [[nodiscard]] std::optional<size_t> highest() const {
        for (size_t w = words_.size(); w-- > 0;) {
            if (words_[w]) {
                return w * 64 + 63 - static_cast<size_t>(std::countl_zero(words_[w]));
            }
        }
        return std::nullopt;
    }
    // The set bits in ascending order.
    [[nodiscard]] std::vector<size_t> ones() const {
        std::vector<size_t> out;
        for (size_t w = 0; w < words_.size(); ++w) {
            uint64_t word = words_[w];
            while (word) {
                out.push_back(w * 64 + static_cast<size_t>(std::countr_zero(word)));
                word &= word - 1;
            }
        }
        return out;
    }

    [[nodiscard]] bool operator==(const Bits&) const = default;

  private:
    size_t size_ = 0;
    std::vector<uint64_t> words_;
};

// GF(2) vectors with distinct leading (highest) bits, each with the
// combination of inserted items it came from. Reduction stops at the first
// leading bit with no row, which is enough to test membership.
class EchelonBasis {
  public:
    EchelonBasis(size_t vector_size, size_t combination_size)
        : rows_(vector_size), combination_size_(combination_size) {}

    // Insert `vector` labelled by `combination`; return the combination that
    // reduces it to zero, or nullopt if it was independent and is now a row.
    std::optional<Bits> insert(Bits vector, Bits combination) {
        reduce(vector, combination);
        if (const auto lead = vector.highest()) {
            rows_[*lead] = Row{std::move(vector), std::move(combination)};
            return std::nullopt;
        }
        return combination;
    }

    // A combination of inserted items that equals `vector`, if any.
    [[nodiscard]] std::optional<Bits> solve(Bits vector) const {
        Bits combination(combination_size_);
        reduce(vector, combination);
        if (vector.any()) {
            return std::nullopt;
        }
        return combination;
    }

    // The leading bits of the rows, ascending.
    [[nodiscard]] std::vector<size_t> leads() const {
        std::vector<size_t> out;
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (rows_[i]) {
                out.push_back(i);
            }
        }
        return out;
    }
    [[nodiscard]] bool has_lead(size_t i) const { return rows_[i].has_value(); }

  private:
    struct Row {
        Bits vector;
        Bits combination;
    };

    void reduce(Bits& vector, Bits& combination) const {
        while (const auto lead = vector.highest()) {
            const std::optional<Row>& row = rows_[*lead];
            if (!row) {
                return;
            }
            vector ^= row->vector;
            combination ^= row->combination;
        }
    }

    std::vector<std::optional<Row>> rows_;
    size_t combination_size_;
};

}  // namespace clifft::zx
