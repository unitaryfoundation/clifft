#pragma once

#include "ucc/backend/backend.h"

#include <algorithm>
#include <bit>
#include <cstdint>

#if defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#include <immintrin.h>
#endif

namespace ucc {

// =============================================================================
// Bit helpers for PauliBitMask (BitMask<kMaxInlineQubits>)
// These are pure bit-manipulation and have no ISA dependency.
// =============================================================================

inline bool bit_get(const PauliBitMask& m, uint16_t idx) {
    return m.bit_get(idx);
}

inline void bit_set(PauliBitMask& m, uint16_t idx, bool v) {
    m.bit_set(idx, v);
}

inline void bit_xor(PauliBitMask& m, uint16_t idx, bool v) {
    if (v) {
        m.bit_xor(idx);
    }
}

inline void bit_swap(PauliBitMask& m1, uint16_t i1, PauliBitMask& m2, uint16_t i2) {
    bool b1 = m1.bit_get(i1);
    bool b2 = m2.bit_get(i2);
    if (b1 != b2) {
        m1.bit_xor(i1);
        m2.bit_xor(i2);
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
// so they live inside UCC_SIMD_NAMESPACE to avoid ODR violations when
// compiled into separate scalar and AVX2 translation units.

#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
#define UCC_HAS_PDEP 1
#else
#define UCC_HAS_PDEP 0
#endif

#ifdef UCC_SIMD_NAMESPACE
namespace UCC_SIMD_NAMESPACE {

inline uint64_t insert_zero_bit(uint64_t val, uint16_t pos) {
    uint64_t mask = (1ULL << pos) - 1;
    return (val & mask) | ((val & ~mask) << 1);
}

inline uint64_t scatter_bits_1(uint64_t val, [[maybe_unused]] uint64_t pdep_mask,
                               [[maybe_unused]] uint16_t bit_pos) {
#if UCC_HAS_PDEP
    return _pdep_u64(val, pdep_mask);
#else
    return insert_zero_bit(val, bit_pos);
#endif
}

inline uint64_t scatter_bits_2(uint64_t val, [[maybe_unused]] uint64_t pdep_mask,
                               [[maybe_unused]] uint16_t bit1, [[maybe_unused]] uint16_t bit2) {
#if UCC_HAS_PDEP
    return _pdep_u64(val, pdep_mask);
#else
    uint16_t min_bit = std::min(bit1, bit2);
    uint16_t max_bit = std::max(bit1, bit2);
    val = insert_zero_bit(val, min_bit);
    return insert_zero_bit(val, max_bit);
#endif
}

}  // namespace UCC_SIMD_NAMESPACE
#endif

}  // namespace ucc
