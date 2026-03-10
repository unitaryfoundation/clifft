#pragma once

// BitMask<N>: Fixed-width bit array for inline Pauli masks.
//
// Wraps std::array<uint64_t, N/64> with bitwise operators and bit-level
// helpers. All loops are over a compile-time-constant extent so the
// compiler can fully unroll and auto-vectorize at -O3 (AVX2, AVX-512,
// NEON) without architecture-specific intrinsics in our source.
//
// At N=64 the array is a single uint64_t -- identical codegen to the
// MVP's raw uint64_t path. At N=512 it becomes 8 words (64 bytes),
// fitting exactly in one AVX-512 register or two AVX2 registers.

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace ucc {

template <size_t N>
struct BitMask {
    static_assert(N >= 64 && N % 64 == 0, "N must be a positive multiple of 64");
    static constexpr size_t kWords = N / 64;
    static constexpr size_t kBits = N;

    std::array<uint64_t, kWords> w{};

    // -- Construction --

    constexpr BitMask() = default;

    // Implicit conversion from a single uint64_t (sets word 0).
    // Enables natural use like: BitMask<512> m = 0;
    constexpr BitMask(uint64_t val) : w{} { w[0] = val; }  // NOLINT(google-explicit-constructor)

    // -- Bitwise operators --

    constexpr BitMask operator^(const BitMask& o) const {
        BitMask r;
        for (size_t i = 0; i < kWords; ++i)
            r.w[i] = w[i] ^ o.w[i];
        return r;
    }
    constexpr BitMask& operator^=(const BitMask& o) {
        for (size_t i = 0; i < kWords; ++i)
            w[i] ^= o.w[i];
        return *this;
    }
    constexpr BitMask operator&(const BitMask& o) const {
        BitMask r;
        for (size_t i = 0; i < kWords; ++i)
            r.w[i] = w[i] & o.w[i];
        return r;
    }
    constexpr BitMask& operator&=(const BitMask& o) {
        for (size_t i = 0; i < kWords; ++i)
            w[i] &= o.w[i];
        return *this;
    }
    constexpr BitMask operator|(const BitMask& o) const {
        BitMask r;
        for (size_t i = 0; i < kWords; ++i)
            r.w[i] = w[i] | o.w[i];
        return r;
    }
    constexpr BitMask& operator|=(const BitMask& o) {
        for (size_t i = 0; i < kWords; ++i)
            w[i] |= o.w[i];
        return *this;
    }
    constexpr BitMask operator~() const {
        BitMask r;
        for (size_t i = 0; i < kWords; ++i)
            r.w[i] = ~w[i];
        return r;
    }

    // -- Comparison --

    constexpr bool operator==(const BitMask& o) const {
        for (size_t i = 0; i < kWords; ++i) {
            if (w[i] != o.w[i])
                return false;
        }
        return true;
    }
    constexpr bool operator!=(const BitMask& o) const { return !(*this == o); }

    // -- Queries --

    /// True if all bits are zero.
    [[nodiscard]] constexpr bool is_zero() const {
        for (size_t i = 0; i < kWords; ++i) {
            if (w[i] != 0)
                return false;
        }
        return true;
    }

    /// Population count across all words.
    [[nodiscard]] constexpr int popcount() const {
        int c = 0;
        for (size_t i = 0; i < kWords; ++i)
            c += std::popcount(w[i]);
        return c;
    }

    /// Index of the lowest set bit. Returns N (sentinel) if is_zero().
    [[nodiscard]] constexpr uint32_t lowest_bit() const {
        for (size_t i = 0; i < kWords; ++i) {
            if (w[i] != 0) {
                return static_cast<uint32_t>(i * 64 + std::countr_zero(w[i]));
            }
        }
        return static_cast<uint32_t>(N);  // sentinel: no bit set
    }

    // -- Bit-level helpers --

    /// Get bit at position idx (0-indexed). Requires idx < N.
    [[nodiscard]] constexpr bool bit_get(uint32_t idx) const {
        assert(idx < N && "bit_get: index out of range");
        return (w[idx / 64] >> (idx % 64)) & 1ULL;
    }

    /// Set bit at position idx to val. Requires idx < N.
    constexpr void bit_set(uint32_t idx, bool val) {
        assert(idx < N && "bit_set: index out of range");
        uint64_t mask = 1ULL << (idx % 64);
        if (val)
            w[idx / 64] |= mask;
        else
            w[idx / 64] &= ~mask;
    }

    /// Toggle (XOR) bit at position idx. Requires idx < N.
    constexpr void bit_xor(uint32_t idx) {
        assert(idx < N && "bit_xor: index out of range");
        w[idx / 64] ^= (1ULL << (idx % 64));
    }

    /// Swap bits at positions i and j. Requires i < N and j < N.
    constexpr void bit_swap(uint32_t i, uint32_t j) {
        assert(i < N && "bit_swap: index i out of range");
        assert(j < N && "bit_swap: index j out of range");
        if (i == j)
            return;
        bool a = bit_get(i);
        bool b = bit_get(j);
        bit_set(i, b);
        bit_set(j, a);
    }

    /// Clear the lowest set bit (like x &= x - 1 but multi-word safe).
    constexpr void clear_lowest_bit() {
        for (size_t i = 0; i < kWords; ++i) {
            if (w[i] != 0) {
                w[i] &= w[i] - 1;
                return;
            }
        }
    }
};

}  // namespace ucc
