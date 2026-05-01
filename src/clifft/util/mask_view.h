#pragma once

// Runtime-width Pauli mask views.
//
// MaskView and MutableMaskView are non-owning references to a contiguous
// uint64_t bit array, parameterized only at runtime by `num_words`. They
// expose the same bit-level operations as BitMask<N> but without the
// compile-time width constraint, so algorithms written against the view
// API work uniformly at any width.
//
// These types are the foundation for migrating Pauli mask storage off the
// CLIFFT_MAX_QUBITS compile-time bound (issue #45). BitMask<N> remains the
// inline storage type while individual algorithms migrate to views one at
// a time.

#include "clifft/util/bitmask.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace clifft {

/// Read-only view over a contiguous uint64_t bit array.
struct MaskView {
    const uint64_t* w;
    uint32_t num_words;

    [[nodiscard]] constexpr bool bit_get(uint32_t idx) const {
        assert(idx / 64 < num_words && "bit_get: index out of range");
        return (w[idx / 64] >> (idx % 64)) & 1ULL;
    }

    [[nodiscard]] constexpr bool is_zero() const {
        for (uint32_t i = 0; i < num_words; ++i) {
            if (w[i] != 0)
                return false;
        }
        return true;
    }

    [[nodiscard]] constexpr int popcount() const {
        int c = 0;
        for (uint32_t i = 0; i < num_words; ++i)
            c += std::popcount(w[i]);
        return c;
    }

    /// Returns num_words*64 as a sentinel when no bit is set.
    [[nodiscard]] constexpr uint32_t lowest_bit() const {
        for (uint32_t i = 0; i < num_words; ++i) {
            if (w[i] != 0)
                return i * 64 + std::countr_zero(w[i]);
        }
        return num_words * 64;
    }
};

/// Mutable view over a contiguous uint64_t bit array. Implicitly converts
/// to MaskView for read-only call sites.
struct MutableMaskView {
    uint64_t* w;
    uint32_t num_words;

    constexpr operator MaskView() const { return {w, num_words}; }

    [[nodiscard]] constexpr bool bit_get(uint32_t idx) const {
        return MaskView{w, num_words}.bit_get(idx);
    }
    [[nodiscard]] constexpr bool is_zero() const { return MaskView{w, num_words}.is_zero(); }
    [[nodiscard]] constexpr int popcount() const { return MaskView{w, num_words}.popcount(); }
    [[nodiscard]] constexpr uint32_t lowest_bit() const {
        return MaskView{w, num_words}.lowest_bit();
    }

    constexpr void bit_set(uint32_t idx, bool v) {
        assert(idx / 64 < num_words && "bit_set: index out of range");
        uint64_t mask = 1ULL << (idx % 64);
        if (v)
            w[idx / 64] |= mask;
        else
            w[idx / 64] &= ~mask;
    }

    constexpr void bit_xor(uint32_t idx) {
        assert(idx / 64 < num_words && "bit_xor: index out of range");
        w[idx / 64] ^= (1ULL << (idx % 64));
    }

    /// Multi-word analog of `x &= x - 1` -- clears the lowest set bit.
    constexpr void clear_lowest_bit() {
        for (uint32_t i = 0; i < num_words; ++i) {
            if (w[i] != 0) {
                w[i] &= w[i] - 1;
                return;
            }
        }
    }

    constexpr void zero_out() {
        for (uint32_t i = 0; i < num_words; ++i)
            w[i] = 0;
    }

    constexpr void xor_with(MaskView other) {
        assert(num_words == other.num_words);
        for (uint32_t i = 0; i < num_words; ++i)
            w[i] ^= other.w[i];
    }

    constexpr void and_with(MaskView other) {
        assert(num_words == other.num_words);
        for (uint32_t i = 0; i < num_words; ++i)
            w[i] &= other.w[i];
    }

    constexpr void or_with(MaskView other) {
        assert(num_words == other.num_words);
        for (uint32_t i = 0; i < num_words; ++i)
            w[i] |= other.w[i];
    }

    constexpr void copy_from(MaskView other) {
        assert(num_words == other.num_words);
        for (uint32_t i = 0; i < num_words; ++i)
            w[i] = other.w[i];
    }
};

// Adapters: expose a fixed-width BitMask<N> through the runtime view API.
template <size_t N>
inline MaskView view(const BitMask<N>& m) {
    return {m.w.data(), static_cast<uint32_t>(BitMask<N>::kWords)};
}

template <size_t N>
inline MutableMaskView mut_view(BitMask<N>& m) {
    return {m.w.data(), static_cast<uint32_t>(BitMask<N>::kWords)};
}

}  // namespace clifft
