#include "clifft/util/bitmask.h"

#include <catch2/catch_test_macros.hpp>

using clifft::BitMask;

// =========================================================================
// Basic construction and comparison
// =========================================================================

TEST_CASE("BitMask: default constructed is zero") {
    BitMask<64> m64;
    REQUIRE(m64.is_zero());
    REQUIRE(m64 == BitMask<64>(0));

    BitMask<512> m512;
    REQUIRE(m512.is_zero());
}

TEST_CASE("BitMask: implicit uint64 constructor sets word 0") {
    BitMask<128> m(0xDEADBEEF);
    REQUIRE(m.w[0] == 0xDEADBEEF);
    REQUIRE(m.w[1] == 0);
    REQUIRE(!m.is_zero());
}

TEST_CASE("BitMask: equality and inequality") {
    BitMask<256> a, b;
    REQUIRE(a == b);
    a.bit_set(200, true);
    REQUIRE(a != b);
    b.bit_set(200, true);
    REQUIRE(a == b);
}

// =========================================================================
// Bit-level helpers across word boundaries
// =========================================================================

TEST_CASE("BitMask: bit_get and bit_set at word boundaries") {
    BitMask<512> m;

    // Within word 0
    m.bit_set(0, true);
    REQUIRE(m.bit_get(0));
    REQUIRE(!m.bit_get(1));

    m.bit_set(63, true);
    REQUIRE(m.bit_get(63));

    // Word boundary: bit 64 is word 1 bit 0
    REQUIRE(!m.bit_get(64));
    m.bit_set(64, true);
    REQUIRE(m.bit_get(64));
    REQUIRE(m.w[1] == 1ULL);

    // Bit 127: word 1 bit 63
    m.bit_set(127, true);
    REQUIRE(m.bit_get(127));
    REQUIRE(m.w[1] == ((1ULL << 63) | 1ULL));

    // Last bit: 511
    m.bit_set(511, true);
    REQUIRE(m.bit_get(511));
    REQUIRE(m.w[7] == (1ULL << 63));

    // Clear a bit
    m.bit_set(0, false);
    REQUIRE(!m.bit_get(0));
}

TEST_CASE("BitMask: bit_xor toggles across words") {
    BitMask<256> m;
    m.bit_xor(130);
    REQUIRE(m.bit_get(130));
    m.bit_xor(130);
    REQUIRE(!m.bit_get(130));
    REQUIRE(m.is_zero());
}

TEST_CASE("BitMask: bit_swap across word boundary") {
    BitMask<512> m;
    m.bit_set(10, true);    // word 0
    m.bit_set(200, false);  // word 3

    m.bit_swap(10, 200);
    REQUIRE(!m.bit_get(10));
    REQUIRE(m.bit_get(200));

    // Swap back
    m.bit_swap(200, 10);
    REQUIRE(m.bit_get(10));
    REQUIRE(!m.bit_get(200));
}

TEST_CASE("BitMask: bit_swap same index is no-op") {
    BitMask<128> m;
    m.bit_set(70, true);
    m.bit_swap(70, 70);
    REQUIRE(m.bit_get(70));
}

// =========================================================================
// Bitwise operators
// =========================================================================

TEST_CASE("BitMask: XOR operator across multiple words") {
    BitMask<256> a, b;
    a.bit_set(0, true);
    a.bit_set(128, true);
    b.bit_set(128, true);
    b.bit_set(200, true);

    auto c = a ^ b;
    REQUIRE(c.bit_get(0));     // 1 ^ 0
    REQUIRE(!c.bit_get(128));  // 1 ^ 1
    REQUIRE(c.bit_get(200));   // 0 ^ 1
}

TEST_CASE("BitMask: AND operator") {
    BitMask<256> a, b;
    a.bit_set(50, true);
    a.bit_set(150, true);
    b.bit_set(150, true);
    b.bit_set(200, true);

    auto c = a & b;
    REQUIRE(!c.bit_get(50));
    REQUIRE(c.bit_get(150));
    REQUIRE(!c.bit_get(200));
}

TEST_CASE("BitMask: OR operator") {
    BitMask<256> a, b;
    a.bit_set(50, true);
    b.bit_set(200, true);

    auto c = a | b;
    REQUIRE(c.bit_get(50));
    REQUIRE(c.bit_get(200));
}

TEST_CASE("BitMask: NOT operator") {
    BitMask<128> m;
    auto inv = ~m;
    REQUIRE(inv.w[0] == ~0ULL);
    REQUIRE(inv.w[1] == ~0ULL);
    REQUIRE(inv.popcount() == 128);
}

TEST_CASE("BitMask: compound assignment XOR") {
    BitMask<512> a;
    a.bit_set(300, true);
    BitMask<512> b;
    b.bit_set(300, true);
    b.bit_set(400, true);

    a ^= b;
    REQUIRE(!a.bit_get(300));
    REQUIRE(a.bit_get(400));
}

TEST_CASE("BitMask: compound assignment AND") {
    BitMask<512> a;
    a.bit_set(100, true);
    a.bit_set(400, true);
    BitMask<512> b;
    b.bit_set(400, true);

    a &= b;
    REQUIRE(!a.bit_get(100));
    REQUIRE(a.bit_get(400));
}

TEST_CASE("BitMask: compound assignment OR") {
    BitMask<512> a;
    BitMask<512> b;
    b.bit_set(511, true);

    a |= b;
    REQUIRE(a.bit_get(511));
}

// =========================================================================
// Popcount and lowest_bit
// =========================================================================

TEST_CASE("BitMask: popcount across words") {
    BitMask<512> m;
    REQUIRE(m.popcount() == 0);

    m.bit_set(0, true);
    m.bit_set(63, true);
    m.bit_set(64, true);
    m.bit_set(511, true);
    REQUIRE(m.popcount() == 4);

    // Fill an entire word
    BitMask<128> full;
    full.w[0] = ~0ULL;
    REQUIRE(full.popcount() == 64);
    full.w[1] = ~0ULL;
    REQUIRE(full.popcount() == 128);
}

TEST_CASE("BitMask: lowest_bit finds across words") {
    BitMask<512> m;
    REQUIRE(m.lowest_bit() == 512);  // sentinel for empty

    m.bit_set(300, true);
    REQUIRE(m.lowest_bit() == 300);

    m.bit_set(100, true);
    REQUIRE(m.lowest_bit() == 100);

    m.bit_set(0, true);
    REQUIRE(m.lowest_bit() == 0);
}

TEST_CASE("BitMask: clear_lowest_bit") {
    BitMask<256> m;
    m.bit_set(10, true);
    m.bit_set(130, true);
    REQUIRE(m.popcount() == 2);

    m.clear_lowest_bit();
    REQUIRE(!m.bit_get(10));
    REQUIRE(m.bit_get(130));
    REQUIRE(m.popcount() == 1);

    m.clear_lowest_bit();
    REQUIRE(m.is_zero());
}

// =========================================================================
// 64-bit specialization: BitMask<64> matches raw uint64_t semantics
// =========================================================================

TEST_CASE("BitMask 64: single word behaves like uint64_t") {
    BitMask<64> a(0xFF00FF00ULL);
    BitMask<64> b(0x00FF00FFULL);

    auto c = a | b;
    REQUIRE(c.w[0] == 0xFFFFFFFFULL);

    auto d = a & b;
    REQUIRE(d.is_zero());

    auto e = a ^ b;
    REQUIRE(e.w[0] == 0xFFFFFFFFULL);

    REQUIRE(a.popcount() == 16);
    REQUIRE(a.lowest_bit() == 8);
}

// =========================================================================
// Constexpr validation
// =========================================================================

TEST_CASE("BitMask: constexpr operations compile") {
    // Verify key operations are usable in constexpr context
    constexpr BitMask<128> zero;
    static_assert(zero.is_zero());

    constexpr BitMask<128> val(42);
    static_assert(!val.is_zero());
    static_assert(val.w[0] == 42);
    static_assert(val.w[1] == 0);
}
