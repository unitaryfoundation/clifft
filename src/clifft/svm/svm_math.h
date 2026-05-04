#pragma once

#include "clifft/backend/backend.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>

#if defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#include <immintrin.h>
#endif

namespace clifft {

// =============================================================================
// Bit helpers for the SchrodingerState Pauli frame.
//
// p_x and p_z are runtime-sized std::vector<uint64_t>; std::span gives a
// uniform handle that accepts both vectors and arrays without copying.
// These are pure bit-manipulation and have no ISA dependency.
// =============================================================================

inline bool bit_get(std::span<const uint64_t> m, uint32_t idx) {
    return (m[idx / 64] >> (idx % 64)) & 1ULL;
}

inline void bit_set(std::span<uint64_t> m, uint32_t idx, bool v) {
    const uint64_t mask = 1ULL << (idx % 64);
    if (v) {
        m[idx / 64] |= mask;
    } else {
        m[idx / 64] &= ~mask;
    }
}

inline void bit_xor(std::span<uint64_t> m, uint32_t idx, bool v) {
    if (v) {
        m[idx / 64] ^= (1ULL << (idx % 64));
    }
}

inline void bit_swap(std::span<uint64_t> m1, uint32_t i1, std::span<uint64_t> m2, uint32_t i2) {
    bool b1 = (m1[i1 / 64] >> (i1 % 64)) & 1ULL;
    bool b2 = (m2[i2 / 64] >> (i2 % 64)) & 1ULL;
    if (b1 != b2) {
        m1[i1 / 64] ^= (1ULL << (i1 % 64));
        m2[i2 / 64] ^= (1ULL << (i2 % 64));
    }
}

// =============================================================================
// Bit-weaving helpers for branchless qubit-subspace iteration
// =============================================================================
//
// On x86-64 with BMI2, we use the PDEP hardware instruction which scatters
// contiguous bits of `val` into positions marked by 1s in `mask` in a single
// cycle. This replaces ~15 shift/and operations per index calculation.
//
// For 1-axis ops: pdep_mask = ~(1ULL << axis), deposits i into all bits
// except the axis bit. For 2-axis ops: pdep_mask = ~(c_bit | t_bit).
//
// These functions change implementation based on __BMI2__ compiler flags,
// so they live inside CLIFFT_SIMD_NAMESPACE to avoid ODR violations when
// compiled into separate scalar and AVX2 translation units.

#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
#define CLIFFT_HAS_PDEP 1
#else
#define CLIFFT_HAS_PDEP 0
#endif

#ifdef CLIFFT_SIMD_NAMESPACE
namespace CLIFFT_SIMD_NAMESPACE {

inline uint64_t insert_zero_bit(uint64_t val, uint16_t pos) {
    uint64_t mask = (1ULL << pos) - 1;
    return (val & mask) | ((val & ~mask) << 1);
}

inline uint64_t scatter_bits_1(uint64_t val, [[maybe_unused]] uint64_t pdep_mask,
                               [[maybe_unused]] uint16_t bit_pos) {
#if CLIFFT_HAS_PDEP
    return _pdep_u64(val, pdep_mask);
#else
    return insert_zero_bit(val, bit_pos);
#endif
}

inline uint64_t scatter_bits_2(uint64_t val, [[maybe_unused]] uint64_t pdep_mask,
                               [[maybe_unused]] uint16_t bit1, [[maybe_unused]] uint16_t bit2) {
#if CLIFFT_HAS_PDEP
    return _pdep_u64(val, pdep_mask);
#else
    uint16_t min_bit = std::min(bit1, bit2);
    uint16_t max_bit = std::max(bit1, bit2);
    val = insert_zero_bit(val, min_bit);
    return insert_zero_bit(val, max_bit);
#endif
}

}  // namespace CLIFFT_SIMD_NAMESPACE
#endif

}  // namespace clifft
