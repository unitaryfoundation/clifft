#pragma once

// Sparse Pauli masks.
//
// A Pauli mask is a signed Pauli string: X bits, Z bits, and a sign, with
// both bits set meaning Y. It is stored as the list of its nonzero 64-qubit
// words in ascending word order, so its size follows its support instead of
// the register width:
//
//   -X0*Z70 on any register  ->  sign = true,
//                                 {index 0: x = 1, z = 0},
//                                 {index 1: x = 0, z = 1 << 6}
//
// PauliMaskView borrows such a list and PauliMask owns one. A register that
// fits in one word stores at most one entry per mask.

#include "clifft/util/mask_view.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

/// One nonzero word of a Pauli mask, covering qubits [64*index, 64*index + 64).
struct PauliWord {
    uint32_t index = 0;
    uint64_t x = 0;
    uint64_t z = 0;

    [[nodiscard]] bool operator==(const PauliWord&) const = default;
};

/// The X bits or the Z bits of a sparse Pauli mask, read as one bit array.
class SparseMaskView {
  public:
    constexpr SparseMaskView(std::span<const PauliWord> words, uint64_t PauliWord::*bits)
        : words_(words), bits_(bits) {}

    /// The word at `index`, or zero when the mask stores no such word.
    [[nodiscard]] uint64_t word(uint32_t index) const {
        const auto it = std::ranges::lower_bound(words_, index, {}, &PauliWord::index);
        return it != words_.end() && it->index == index ? (*it).*bits_ : 0;
    }

    [[nodiscard]] bool bit_get(uint32_t idx) const { return (word(idx / 64) >> (idx % 64)) & 1U; }

    [[nodiscard]] bool is_zero() const {
        return std::ranges::all_of(words_, [&](const PauliWord& w) { return w.*bits_ == 0; });
    }

    [[nodiscard]] int popcount() const {
        int count = 0;
        for (const PauliWord& w : words_) {
            count += std::popcount(w.*bits_);
        }
        return count;
    }

    /// Invoke fn(bit) in ascending order for each set bit.
    template <typename Fn>
    void for_each_set_bit(Fn&& fn) const {
        for (const PauliWord& w : words_) {
            for (uint64_t pending = w.*bits_; pending != 0; pending &= pending - 1) {
                fn(w.index * 64 + static_cast<uint32_t>(std::countr_zero(pending)));
            }
        }
    }

    /// XOR these bits into a dense array wide enough to hold every stored word.
    void xor_into(MutableMaskView dense) const {
        for (const PauliWord& w : words_) {
            assert(w.index < dense.num_words());
            dense.words[w.index] ^= w.*bits_;
        }
    }

  private:
    friend bool operator==(SparseMaskView a, SparseMaskView b);

    std::span<const PauliWord> words_;
    uint64_t PauliWord::*bits_;
};

/// Equal bit arrays. The two sides may store different all-zero words.
[[nodiscard]] inline bool operator==(SparseMaskView a, SparseMaskView b) {
    size_t i = 0;
    size_t j = 0;
    while (i < a.words_.size() || j < b.words_.size()) {
        const uint32_t ai = i < a.words_.size() ? a.words_[i].index : ~uint32_t{0};
        const uint32_t bi = j < b.words_.size() ? b.words_[j].index : ~uint32_t{0};
        const uint64_t av = ai <= bi ? a.words_[i].*(a.bits_) : 0;
        const uint64_t bv = bi <= ai ? b.words_[j].*(b.bits_) : 0;
        if (av != bv) {
            return false;
        }
        i += ai <= bi;
        j += bi <= ai;
    }
    return true;
}

/// Equal to a dense bit array. Dense words beyond the stored ones must be zero.
[[nodiscard]] inline bool operator==(SparseMaskView a, MaskView dense) {
    int stored = 0;
    for (uint32_t w = 0; w < dense.num_words(); ++w) {
        if (a.word(w) != dense.words[w]) {
            return false;
        }
        stored += std::popcount(dense.words[w]);
    }
    return stored == a.popcount();
}

/// Read-only view of a sparse Pauli mask. Words are sorted by index and each
/// has a nonzero X or Z part, so equal Paulis have equal word lists.
class PauliMaskView {
  public:
    constexpr PauliMaskView() = default;
    constexpr PauliMaskView(std::span<const PauliWord> words, bool sign)
        : words_(words), sign_(sign) {}

    [[nodiscard]] std::span<const PauliWord> words() const { return words_; }
    // Iterates the stored words, so a view returned by value can be the
    // range of a for loop.
    [[nodiscard]] auto begin() const { return words_.begin(); }
    [[nodiscard]] auto end() const { return words_.end(); }
    [[nodiscard]] SparseMaskView x() const { return {words_, &PauliWord::x}; }
    [[nodiscard]] SparseMaskView z() const { return {words_, &PauliWord::z}; }
    [[nodiscard]] bool sign() const { return sign_; }

    /// True when the unsigned Pauli is the identity.
    [[nodiscard]] bool is_identity() const { return words_.empty(); }

    /// Half-open range of word indices that can hold a nonzero word.
    [[nodiscard]] uint32_t begin_word() const { return words_.empty() ? 0 : words_.front().index; }
    [[nodiscard]] uint32_t end_word() const { return words_.empty() ? 0 : words_.back().index + 1; }

    /// XOR the X and Z bits into dense arrays wide enough for every word.
    void xor_into(MutableMaskView x, MutableMaskView z) const {
        for (const PauliWord& w : words_) {
            assert(w.index < x.num_words() && w.index < z.num_words());
            x.words[w.index] ^= w.x;
            z.words[w.index] ^= w.z;
        }
    }

  private:
    std::span<const PauliWord> words_;
    bool sign_ = false;
};

/// Same unsigned Pauli.
[[nodiscard]] inline bool same_body(PauliMaskView a, PauliMaskView b) {
    return std::ranges::equal(a.words(), b.words());
}

[[nodiscard]] inline bool operator==(PauliMaskView a, PauliMaskView b) {
    return a.sign() == b.sign() && same_body(a, b);
}

/// Invoke fn(a_word, b_word) for each word index stored by both masks.
template <typename Fn>
void for_each_common_word(PauliMaskView a, PauliMaskView b, Fn&& fn) {
    const std::span<const PauliWord> aw = a.words();
    const std::span<const PauliWord> bw = b.words();
    size_t i = 0;
    size_t j = 0;
    while (i < aw.size() && j < bw.size()) {
        // Equal indices first: masks on a one-word register always match.
        if (aw[i].index == bw[j].index) {
            fn(aw[i], bw[j]);
            ++i;
            ++j;
        } else if (aw[i].index < bw[j].index) {
            ++i;
        } else {
            ++j;
        }
    }
}

/// True if the two Pauli strings anti-commute.
[[nodiscard]] inline bool anti_commute(PauliMaskView a, PauliMaskView b) {
    const std::span<const PauliWord> aw = a.words();
    const std::span<const PauliWord> bw = b.words();
    // Masks on a register of at most 64 qubits hold one word each; skip the
    // merge for them.
    if (aw.size() == 1 && bw.size() == 1) {
        const PauliWord& u = aw.front();
        const PauliWord& v = bw.front();
        return u.index == v.index && (std::popcount((u.x & v.z) ^ (u.z & v.x)) & 1) != 0;
    }
    // XOR the per-word symplectic products; the parity of the total is the
    // parity of the sum of their bit counts.
    uint64_t product = 0;
    for_each_common_word(a, b, [&](const PauliWord& u, const PauliWord& v) {
        product ^= (u.x & v.z) ^ (u.z & v.x);
    });
    return (std::popcount(product) & 1) != 0;
}

/// True if a sparse Pauli anti-commutes with a dense one. The dense arrays
/// must be wide enough for every stored word.
[[nodiscard]] inline bool anti_commute(PauliMaskView a, MaskView x, MaskView z) {
    int parity = 0;
    for (const PauliWord& w : a) {
        assert(w.index < x.num_words() && w.index < z.num_words());
        parity += std::popcount((w.x & z.words[w.index]) ^ (w.z & x.words[w.index]));
    }
    return (parity & 1) != 0;
}

/// True when the two Pauli strings act nontrivially on a common qubit.
[[nodiscard]] inline bool supports_overlap(PauliMaskView a, PauliMaskView b) {
    bool overlap = false;
    for_each_common_word(a, b, [&](const PauliWord& u, const PauliWord& v) {
        overlap |= ((u.x | u.z) & (v.x | v.z)) != 0;
    });
    return overlap;
}

/// Append the unsigned product of two Paulis (the XOR of their bits) to out.
/// out must not alias either input.
inline void append_body_product(PauliMaskView a, PauliMaskView b, std::vector<PauliWord>& out) {
    const std::span<const PauliWord> aw = a.words();
    const std::span<const PauliWord> bw = b.words();
    size_t i = 0;
    size_t j = 0;
    while (i < aw.size() && j < bw.size()) {
        if (aw[i].index < bw[j].index) {
            out.push_back(aw[i++]);
        } else if (bw[j].index < aw[i].index) {
            out.push_back(bw[j++]);
        } else {
            const PauliWord merged{aw[i].index, aw[i].x ^ bw[j].x, aw[i].z ^ bw[j].z};
            if ((merged.x | merged.z) != 0) {
                out.push_back(merged);
            }
            ++i;
            ++j;
        }
    }
    out.insert(out.end(), aw.begin() + static_cast<ptrdiff_t>(i), aw.end());
    out.insert(out.end(), bw.begin() + static_cast<ptrdiff_t>(j), bw.end());
}

/// Reduce words gathered from several masks to the union of those masks:
/// sorted by index, with the X and Z bits of words sharing an index ORed.
inline void merge_union(std::vector<PauliWord>& words) {
    std::ranges::sort(words, {}, &PauliWord::index);
    size_t merged = 0;
    for (const PauliWord& word : words) {
        if (merged > 0 && words[merged - 1].index == word.index) {
            words[merged - 1].x |= word.x;
            words[merged - 1].z |= word.z;
        } else {
            words[merged++] = word;
        }
    }
    words.resize(merged);
}

/// Owning sparse Pauli mask. Reusing one instance across assignments keeps
/// its word storage, so building many masks through it does not allocate
/// once the storage has grown to the largest support.
class PauliMask {
  public:
    PauliMask() = default;
    explicit PauliMask(PauliMaskView source) { assign(source); }
    PauliMask(MaskView x, MaskView z, bool sign) { assign(x, z, sign); }

    /// A mask on a single-word register. Convenient for narrow circuits.
    PauliMask(uint64_t x, uint64_t z, bool sign) : sign_(sign) {
        if ((x | z) != 0) {
            words_.push_back({0, x, z});
        }
    }

    [[nodiscard]] PauliMaskView view() const { return {words_, sign_}; }
    operator PauliMaskView() const { return view(); }  // NOLINT(google-explicit-constructor)

    [[nodiscard]] bool sign() const { return sign_; }
    void set_sign(bool sign) { sign_ = sign; }

    void clear() {
        words_.clear();
        sign_ = false;
    }

    void assign(PauliMaskView source) {
        words_.assign(source.words().begin(), source.words().end());
        sign_ = source.sign();
    }

    /// Copy a dense Pauli, keeping only its nonzero words.
    void assign(MaskView x, MaskView z, bool sign) {
        assert(x.num_words() == z.num_words());
        words_.clear();
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            if ((x.words[w] | z.words[w]) != 0) {
                words_.push_back({w, x.words[w], z.words[w]});
            }
        }
        sign_ = sign;
    }

    /// Multiply the unsigned Pauli by X^x Z^z on one qubit.
    void toggle(uint32_t qubit, bool x, bool z) {
        const PauliWord delta{qubit / 64, uint64_t{x} << (qubit % 64), uint64_t{z} << (qubit % 64)};
        auto it = std::ranges::lower_bound(words_, delta.index, {}, &PauliWord::index);
        if (it == words_.end() || it->index != delta.index) {
            if ((delta.x | delta.z) != 0) {
                words_.insert(it, delta);
            }
            return;
        }
        it->x ^= delta.x;
        it->z ^= delta.z;
        if ((it->x | it->z) == 0) {
            words_.erase(it);
        }
    }

    /// Multiply the unsigned Pauli by another; the sign is left unchanged.
    void multiply_body(PauliMaskView other) {
        scratch_.clear();
        append_body_product(view(), other, scratch_);
        words_.swap(scratch_);
    }

  private:
    std::vector<PauliWord> words_;
    std::vector<PauliWord> scratch_;
    bool sign_ = false;
};

/// Dense copy of one sparse Pauli at a time, for code that reads a Pauli as
/// two full-width bit arrays. Loading a mask clears only the words the
/// previous one set, so the cost follows the masks' support rather than the
/// register width.
class DensePauliScratch {
  public:
    explicit DensePauliScratch(uint32_t num_words) : x_(num_words, 0), z_(num_words, 0) {}

    void load(PauliMaskView mask) {
        for (uint32_t w : loaded_) {
            x_[w] = 0;
            z_[w] = 0;
        }
        loaded_.clear();
        for (const PauliWord& w : mask) {
            assert(w.index < x_.size());
            x_[w.index] = w.x;
            z_[w.index] = w.z;
            loaded_.push_back(w.index);
        }
    }

    [[nodiscard]] MaskView x() const { return MaskView{x_}; }
    [[nodiscard]] MaskView z() const { return MaskView{z_}; }

    /// Half-open range of word indices holding every nonzero word.
    [[nodiscard]] uint32_t begin_word() const { return loaded_.empty() ? 0 : loaded_.front(); }
    [[nodiscard]] uint32_t end_word() const { return loaded_.empty() ? 0 : loaded_.back() + 1; }

  private:
    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    std::vector<uint32_t> loaded_;
};

}  // namespace clifft
