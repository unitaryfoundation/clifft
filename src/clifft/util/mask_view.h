#pragma once

// Non-owning runtime-width references to a contiguous uint64_t bit array.
//
// `MaskView` is read-only; `MutableMaskView` allows mutation. Both are thin
// wrappers around std::span<Word> that expose the same bit-level operations
// as BitMask<N> -- bit_get, bit_xor, popcount, lowest_bit, clear_lowest_bit,
// xor/and/or_with -- without a compile-time width.

#include "clifft/util/bitmask.h"

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace clifft {

template <class Word>
    requires std::same_as<std::remove_const_t<Word>, uint64_t>
struct BasicMaskView {
    std::span<Word> words;

    constexpr BasicMaskView() = default;
    constexpr BasicMaskView(std::span<Word> w) : words(w) {}

    /// Convert a mutable view to a const view.
    template <class W2>
        requires(std::is_const_v<Word> && !std::is_const_v<W2> &&
                 std::same_as<std::remove_const_t<Word>, W2>)
    constexpr BasicMaskView(BasicMaskView<W2> other) : words(other.words) {}

    [[nodiscard]] constexpr uint32_t num_words() const {
        return static_cast<uint32_t>(words.size());
    }

    [[nodiscard]] constexpr bool bit_get(uint32_t idx) const {
        assert(idx / 64 < words.size() && "bit_get: index out of range");
        return (words[idx / 64] >> (idx % 64)) & 1ULL;
    }

    [[nodiscard]] constexpr bool is_zero() const {
        for (auto w : words) {
            if (w != 0)
                return false;
        }
        return true;
    }

    [[nodiscard]] constexpr int popcount() const {
        int c = 0;
        for (auto w : words)
            c += std::popcount(w);
        return c;
    }

    /// Returns num_words()*64 as a sentinel when no bit is set.
    [[nodiscard]] constexpr uint32_t lowest_bit() const {
        for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] != 0)
                return static_cast<uint32_t>(i * 64 + std::countr_zero(words[i]));
        }
        return static_cast<uint32_t>(words.size() * 64);
    }

    constexpr void bit_set(uint32_t idx, bool v)
        requires(!std::is_const_v<Word>)
    {
        assert(idx / 64 < words.size() && "bit_set: index out of range");
        uint64_t mask = 1ULL << (idx % 64);
        if (v)
            words[idx / 64] |= mask;
        else
            words[idx / 64] &= ~mask;
    }

    constexpr void bit_xor(uint32_t idx)
        requires(!std::is_const_v<Word>)
    {
        assert(idx / 64 < words.size() && "bit_xor: index out of range");
        words[idx / 64] ^= (1ULL << (idx % 64));
    }

    /// Multi-word analog of `x &= x - 1`. Clears the lowest set bit.
    constexpr void clear_lowest_bit()
        requires(!std::is_const_v<Word>)
    {
        for (auto& w : words) {
            if (w != 0) {
                w &= w - 1;
                return;
            }
        }
    }

    constexpr void zero_out()
        requires(!std::is_const_v<Word>)
    {
        for (auto& w : words)
            w = 0;
    }

    constexpr void xor_with(BasicMaskView<const uint64_t> other)
        requires(!std::is_const_v<Word>)
    {
        assert(words.size() == other.words.size());
        for (size_t i = 0; i < words.size(); ++i)
            words[i] ^= other.words[i];
    }

    constexpr void and_with(BasicMaskView<const uint64_t> other)
        requires(!std::is_const_v<Word>)
    {
        assert(words.size() == other.words.size());
        for (size_t i = 0; i < words.size(); ++i)
            words[i] &= other.words[i];
    }

    constexpr void or_with(BasicMaskView<const uint64_t> other)
        requires(!std::is_const_v<Word>)
    {
        assert(words.size() == other.words.size());
        for (size_t i = 0; i < words.size(); ++i)
            words[i] |= other.words[i];
    }

    constexpr void copy_from(BasicMaskView<const uint64_t> other)
        requires(!std::is_const_v<Word>)
    {
        assert(words.size() == other.words.size());
        for (size_t i = 0; i < words.size(); ++i)
            words[i] = other.words[i];
    }
};

using MaskView = BasicMaskView<const uint64_t>;
using MutableMaskView = BasicMaskView<uint64_t>;

// Adapters: expose a fixed-width BitMask<N> through the runtime view API.
template <size_t N>
inline MaskView view(const BitMask<N>& m) {
    return MaskView{std::span<const uint64_t>(m.w)};
}

template <size_t N>
inline MutableMaskView mut_view(BitMask<N>& m) {
    return MutableMaskView{std::span<uint64_t>(m.w)};
}

}  // namespace clifft
