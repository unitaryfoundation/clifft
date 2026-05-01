#include "clifft/util/bitmask.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_arena.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <span>

using clifft::BitMask;
using clifft::lowest_bit_at_or_above;
using clifft::lowest_bit_below;
using clifft::MaskView;
using clifft::mut_view;
using clifft::MutableMaskView;
using clifft::MutablePauliMaskView;
using clifft::PauliMaskArena;
using clifft::PauliMaskHandle;
using clifft::PauliMaskView;
using clifft::view;

// =========================================================================
// MaskView: read-only access
// =========================================================================

TEST_CASE("MaskView: empty view reports zero") {
    MaskView v;
    REQUIRE(v.is_zero());
    REQUIRE(v.popcount() == 0);
    REQUIRE(v.lowest_bit() == 0);
    REQUIRE(v.num_words() == 0);
}

TEST_CASE("MaskView: bit_get and is_zero across word boundaries") {
    std::array<uint64_t, 3> words{0, 0, 0};
    MutableMaskView mv{std::span<uint64_t>(words)};
    REQUIRE(MaskView{std::span<const uint64_t>(words)}.is_zero());

    mv.bit_set(0, true);
    mv.bit_set(63, true);
    mv.bit_set(64, true);
    mv.bit_set(127, true);
    mv.bit_set(128, true);
    mv.bit_set(191, true);

    MaskView cv{std::span<const uint64_t>(words)};
    REQUIRE(cv.bit_get(0));
    REQUIRE(cv.bit_get(63));
    REQUIRE(cv.bit_get(64));
    REQUIRE(cv.bit_get(127));
    REQUIRE(cv.bit_get(128));
    REQUIRE(cv.bit_get(191));
    REQUIRE_FALSE(cv.bit_get(1));
    REQUIRE_FALSE(cv.bit_get(62));
    REQUIRE_FALSE(cv.bit_get(190));
    REQUIRE(cv.popcount() == 6);
}

TEST_CASE("MaskView: lowest_bit returns sentinel when empty") {
    std::array<uint64_t, 4> words{0, 0, 0, 0};
    MaskView v{std::span<const uint64_t>(words)};
    REQUIRE(v.lowest_bit() == 4 * 64);
}

TEST_CASE("MaskView: lowest_bit walks past zero words") {
    std::array<uint64_t, 4> words{0, 0, 0x1ULL << 5, 0};
    MaskView v{std::span<const uint64_t>(words)};
    REQUIRE(v.lowest_bit() == 2 * 64 + 5);
}

TEST_CASE("MaskView: implicit conversion from MutableMaskView") {
    std::array<uint64_t, 2> words{0xFF, 0};
    MutableMaskView mv{std::span<uint64_t>(words)};
    MaskView cv = mv;
    REQUIRE(cv.popcount() == 8);
}

// =========================================================================
// MutableMaskView: bulk and bit-level mutation
// =========================================================================

TEST_CASE("MutableMaskView: clear_lowest_bit walks through bits in order") {
    std::array<uint64_t, 2> words{0, 0};
    MutableMaskView mv{std::span<uint64_t>(words)};
    mv.bit_set(3, true);
    mv.bit_set(70, true);
    mv.bit_set(90, true);

    REQUIRE(mv.lowest_bit() == 3);
    mv.clear_lowest_bit();
    REQUIRE(mv.lowest_bit() == 70);
    mv.clear_lowest_bit();
    REQUIRE(mv.lowest_bit() == 90);
    mv.clear_lowest_bit();
    REQUIRE(mv.is_zero());
}

TEST_CASE("MutableMaskView: bit_xor toggles individual bits") {
    std::array<uint64_t, 1> words{0};
    MutableMaskView mv{std::span<uint64_t>(words)};
    mv.bit_xor(7);
    REQUIRE(mv.bit_get(7));
    mv.bit_xor(7);
    REQUIRE_FALSE(mv.bit_get(7));
}

TEST_CASE("MutableMaskView: xor_with composes two views") {
    std::array<uint64_t, 2> a{0xAAAAAAAAAAAAAAAAULL, 0x1};
    std::array<uint64_t, 2> b{0x5555555555555555ULL, 0x3};
    MutableMaskView ma{std::span<uint64_t>(a)};
    MaskView mb{std::span<const uint64_t>(b)};
    ma.xor_with(mb);
    REQUIRE(a[0] == 0xFFFFFFFFFFFFFFFFULL);
    REQUIRE(a[1] == 0x2);
}

TEST_CASE("MutableMaskView: and_with and or_with") {
    std::array<uint64_t, 1> a{0xF0F0};
    std::array<uint64_t, 1> b{0x0FF0};
    MutableMaskView ma{std::span<uint64_t>(a)};

    ma.and_with(MaskView{std::span<const uint64_t>(b)});
    REQUIRE(a[0] == 0x00F0);

    a[0] = 0xF0F0;
    ma.or_with(MaskView{std::span<const uint64_t>(b)});
    REQUIRE(a[0] == 0xFFF0);
}

TEST_CASE("MutableMaskView: zero_out clears all words") {
    std::array<uint64_t, 4> a{1, 2, 3, 4};
    MutableMaskView ma{std::span<uint64_t>(a)};
    ma.zero_out();
    REQUIRE(ma.is_zero());
}

TEST_CASE("MutableMaskView: copy_from replaces contents") {
    std::array<uint64_t, 3> a{0, 0, 0};
    std::array<uint64_t, 3> b{1, 2, 3};
    MutableMaskView{std::span<uint64_t>(a)}.copy_from(MaskView{std::span<const uint64_t>(b)});
    REQUIRE(a[0] == 1);
    REQUIRE(a[1] == 2);
    REQUIRE(a[2] == 3);
}

// =========================================================================
// BitMask<N> adapters
// =========================================================================

TEST_CASE("view adapter: read-only access to BitMask") {
    BitMask<256> m;
    m.bit_set(0, true);
    m.bit_set(200, true);
    auto v = view(m);
    REQUIRE(v.num_words() == 4);
    REQUIRE(v.bit_get(0));
    REQUIRE(v.bit_get(200));
    REQUIRE_FALSE(v.bit_get(199));
    REQUIRE(v.popcount() == 2);
    REQUIRE(v.lowest_bit() == 0);
}

TEST_CASE("mut_view adapter: writes through to BitMask") {
    BitMask<128> m;
    auto mv = mut_view(m);
    mv.bit_set(70, true);
    mv.bit_xor(5);
    REQUIRE(m.bit_get(70));
    REQUIRE(m.bit_get(5));
    REQUIRE(m.popcount() == 2);
}

// =========================================================================
// PauliMaskArena
// =========================================================================

TEST_CASE("PauliMaskArena: capacity and width are fixed at construction") {
    PauliMaskArena arena(128, 5);
    REQUIRE(arena.num_words() == 2);
    REQUIRE(arena.size() == 5);
}

TEST_CASE("PauliMaskArena: per-handle storage is independent") {
    PauliMaskArena arena(256, 2);
    auto m0 = arena.mut_at(PauliMaskHandle{0});
    auto m1 = arena.mut_at(PauliMaskHandle{1});

    m0.x().bit_set(5, true);
    m0.z().bit_set(200, true);
    m0.set_sign(true);

    REQUIRE(m0.x().bit_get(5));
    REQUIRE(m0.z().bit_get(200));
    REQUIRE(m0.sign());

    REQUIRE(m1.x().is_zero());
    REQUIRE(m1.z().is_zero());
    REQUIRE_FALSE(m1.sign());
}

TEST_CASE("PauliMaskArena: const at returns read-only view") {
    PauliMaskArena arena(64, 1);
    arena.mut_at(PauliMaskHandle{0}).x().bit_set(3, true);
    arena.mut_at(PauliMaskHandle{0}).set_sign(true);

    const PauliMaskArena& carena = arena;
    auto cm = carena.at(PauliMaskHandle{0});
    REQUIRE(cm.x().bit_get(3));
    REQUIRE(cm.z().is_zero());
    REQUIRE(cm.sign());
}

TEST_CASE("PauliMaskArena: width above old compile-time bound") {
    // BitMask<N> caps at CLIFFT_MAX_QUBITS at compile time. The arena
    // accepts arbitrary widths and rounds up to whole words.
    PauliMaskArena arena(1024, 1);
    REQUIRE(arena.num_words() == 16);
    arena.mut_at(PauliMaskHandle{0}).x().bit_set(1023, true);
    REQUIRE(arena.at(PauliMaskHandle{0}).x().bit_get(1023));
    REQUIRE(arena.at(PauliMaskHandle{0}).x().lowest_bit() == 1023);
}

TEST_CASE("PauliMaskArena: zero-width arena is well-formed") {
    PauliMaskArena arena(0, 3);
    REQUIRE(arena.num_words() == 0);
    REQUIRE(arena.size() == 3);
    auto m = arena.mut_at(PauliMaskHandle{1});
    REQUIRE(m.x().num_words() == 0);
    REQUIRE(m.x().is_zero());
    m.set_sign(true);
    REQUIRE(arena.at(PauliMaskHandle{1}).sign());
}

TEST_CASE("PauliMaskArena: zero-mask arena is well-formed") {
    PauliMaskArena arena(64, 0);
    REQUIRE(arena.size() == 0);
}

TEST_CASE("PauliMaskArena: mutable view converts implicitly to const view") {
    PauliMaskArena arena(64, 1);
    auto mv = arena.mut_at(PauliMaskHandle{0});
    mv.x().bit_set(7, true);
    PauliMaskView cv = mv;
    REQUIRE(cv.x().bit_get(7));
}

TEST_CASE("PauliMaskArena: const view stays live after sign mutation through mutable view") {
    // Regression: sign() must read through the arena slot, not capture
    // the value at construction. If sign_ ever reverts to a copied bool,
    // the const view here would miss the post-conversion update.
    PauliMaskArena arena(64, 1);
    auto mv = arena.mut_at(PauliMaskHandle{0});
    PauliMaskView cv = mv;
    REQUIRE_FALSE(cv.sign());
    mv.set_sign(true);
    REQUIRE(cv.sign());
    mv.set_sign(false);
    REQUIRE_FALSE(cv.sign());
}

// =========================================================================
// lowest_bit_at_or_above / lowest_bit_below
// =========================================================================
//
// These primitives must correctly handle k values both within and beyond
// a single 64-bit word, including word-aligned k.

namespace {

MaskView mv_from(const std::array<uint64_t, 4>& a) {
    return MaskView{std::span<const uint64_t>(a)};
}

constexpr uint32_t kSentinel = 4 * 64;

}  // namespace

TEST_CASE("lowest_bit_at_or_above: returns word 1 bit when k less than 64") {
    std::array<uint64_t, 4> a{0, 0x1ULL << 5, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 30) == 64 + 5);
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 0) == 64 + 5);
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 63) == 64 + 5);
}

TEST_CASE("lowest_bit_at_or_above: bit 64 is below k when k greater than 64") {
    // Only bit 64 is set. With k=70, bit 64 is below the threshold,
    // so no qualifying bit exists.
    std::array<uint64_t, 4> a{0, 1ULL, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 70) == kSentinel);
}

TEST_CASE("lowest_bit_at_or_above: chooses dormant bit past straddle word") {
    // k=70: in word 1, bits [0, 6) below k, [6, 64) at or above.
    // Set bit 65 (below) and bit 75 (at or above).
    std::array<uint64_t, 4> a{0, (1ULL << 1) | (1ULL << 11), 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 70) == 64 + 11);
}

TEST_CASE("lowest_bit_at_or_above: chooses bit in straddle word when no higher word has bits") {
    std::array<uint64_t, 4> a{0xFF, 1ULL << 16, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 70) == 64 + 16);
}

TEST_CASE("lowest_bit_at_or_above: returns earliest qualifying bit in word 0") {
    // k=30: bit 40 in word 0 qualifies; bit 5 does not.
    std::array<uint64_t, 4> a{(1ULL << 5) | (1ULL << 40), 0, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 30) == 40);
}

TEST_CASE("lowest_bit_at_or_above: returns lowest bit even when higher words have bits") {
    // k=30: bit 40 in word 0 (qualifies), bit 100 in word 1 (qualifies).
    // Lowest qualifying bit is 40, regardless of higher-word bits.
    std::array<uint64_t, 4> a{1ULL << 40, 1ULL << 36, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 30) == 40);
}

TEST_CASE("lowest_bit_at_or_above: skips empty intermediate words") {
    // k=10: lowest qualifying bit is in word 3, with words 0..2 empty
    // above k. Should still find it.
    std::array<uint64_t, 4> a{0, 0, 0, 1ULL << 7};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 10) == 3 * 64 + 7);
}

TEST_CASE("lowest_bit_at_or_above: returns straddle-word bit before higher-word bit") {
    // k=30: bit 35 in word 0 qualifies; bit 64 in word 1 also qualifies.
    // Lowest is 35.
    std::array<uint64_t, 4> a{1ULL << 35, 1ULL, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 30) == 35);
}

TEST_CASE("lowest_bit_at_or_above: returns sentinel when no bit qualifies") {
    // k=128 covers all of words 0 and 1. Bit 5 does not qualify.
    std::array<uint64_t, 4> a{1ULL << 5, 0, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 128) == kSentinel);
}

TEST_CASE("lowest_bit_at_or_above: handles k aligned to word boundary") {
    // k=64: bit 64 is at the threshold and qualifies.
    std::array<uint64_t, 4> a{0, 1ULL, 0, 0};
    REQUIRE(lowest_bit_at_or_above(mv_from(a), 64) == 64);
}

TEST_CASE("lowest_bit_below: returns earliest bit when k spans word 0") {
    std::array<uint64_t, 4> a{(1ULL << 5) | (1ULL << 40), 0, 0, 0};
    REQUIRE(lowest_bit_below(mv_from(a), 64) == 5);
}

TEST_CASE("lowest_bit_below: returns sentinel when only higher bits are set") {
    // k=10, only bit 20 set. No bit below 10.
    std::array<uint64_t, 4> a{1ULL << 20, 0, 0, 0};
    REQUIRE(lowest_bit_below(mv_from(a), 10) == kSentinel);
}

TEST_CASE("lowest_bit_below: scans full word below k when k greater than 64") {
    // k=70: word 0 fully below k. Bit 30 in word 0, bit 65 in word 1
    // (bit 1 of word 1, which is below k=70).
    std::array<uint64_t, 4> a{1ULL << 30, 1ULL << 1, 0, 0};
    REQUIRE(lowest_bit_below(mv_from(a), 70) == 30);
}

TEST_CASE("lowest_bit_below: picks straddle-word bit when word 0 empty") {
    // k=70: word 0 empty, bit 65 in word 1 (bit 1 of word 1, below k).
    std::array<uint64_t, 4> a{0, 1ULL << 1, 0, 0};
    REQUIRE(lowest_bit_below(mv_from(a), 70) == 64 + 1);
}

TEST_CASE("lowest_bit_below: handles k aligned to word boundary") {
    // k=64: word 0 fully below k. Bit 100 is in word 1, not below k.
    std::array<uint64_t, 4> a{0, 1ULL << 36, 0, 0};
    REQUIRE(lowest_bit_below(mv_from(a), 64) == kSentinel);
}
