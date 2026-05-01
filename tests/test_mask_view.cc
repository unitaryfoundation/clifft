#include "clifft/util/bitmask.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_arena.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

using clifft::BitMask;
using clifft::MaskView;
using clifft::mut_view;
using clifft::MutableMaskView;
using clifft::PauliMaskArena;
using clifft::view;

// =========================================================================
// MaskView: read-only access
// =========================================================================

TEST_CASE("MaskView: empty width reports zero") {
    MaskView v{nullptr, 0};
    REQUIRE(v.is_zero());
    REQUIRE(v.popcount() == 0);
    REQUIRE(v.lowest_bit() == 0);
}

TEST_CASE("MaskView: bit_get and is_zero across word boundaries") {
    std::array<uint64_t, 3> words{0, 0, 0};
    MutableMaskView mv{words.data(), 3};
    REQUIRE(MaskView{words.data(), 3}.is_zero());

    mv.bit_set(0, true);
    mv.bit_set(63, true);
    mv.bit_set(64, true);
    mv.bit_set(127, true);
    mv.bit_set(128, true);
    mv.bit_set(191, true);

    MaskView cv{words.data(), 3};
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
    MaskView v{words.data(), 4};
    REQUIRE(v.lowest_bit() == 4 * 64);
}

TEST_CASE("MaskView: lowest_bit walks past zero words") {
    std::array<uint64_t, 4> words{0, 0, 0x1ULL << 5, 0};
    MaskView v{words.data(), 4};
    REQUIRE(v.lowest_bit() == 2 * 64 + 5);
}

// =========================================================================
// MutableMaskView: bulk and bit-level mutation
// =========================================================================

TEST_CASE("MutableMaskView: clear_lowest_bit walks through bits in order") {
    std::array<uint64_t, 2> words{0, 0};
    MutableMaskView mv{words.data(), 2};
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
    MutableMaskView mv{words.data(), 1};
    mv.bit_xor(7);
    REQUIRE(mv.bit_get(7));
    mv.bit_xor(7);
    REQUIRE_FALSE(mv.bit_get(7));
}

TEST_CASE("MutableMaskView: xor_with composes two views") {
    std::array<uint64_t, 2> a{0xAAAAAAAAAAAAAAAAULL, 0x1};
    std::array<uint64_t, 2> b{0x5555555555555555ULL, 0x3};
    MutableMaskView ma{a.data(), 2};
    MaskView mb{b.data(), 2};
    ma.xor_with(mb);
    REQUIRE(a[0] == 0xFFFFFFFFFFFFFFFFULL);
    REQUIRE(a[1] == 0x2);
}

TEST_CASE("MutableMaskView: and_with and or_with") {
    std::array<uint64_t, 1> a{0xF0F0};
    std::array<uint64_t, 1> b{0x0FF0};
    MutableMaskView ma{a.data(), 1};

    ma.and_with(MaskView{b.data(), 1});
    REQUIRE(a[0] == 0x00F0);

    a[0] = 0xF0F0;
    ma.or_with(MaskView{b.data(), 1});
    REQUIRE(a[0] == 0xFFF0);
}

TEST_CASE("MutableMaskView: zero_out clears all words") {
    std::array<uint64_t, 4> a{1, 2, 3, 4};
    MutableMaskView ma{a.data(), 4};
    ma.zero_out();
    REQUIRE(ma.is_zero());
}

TEST_CASE("MutableMaskView: copy_from replaces contents") {
    std::array<uint64_t, 3> a{0, 0, 0};
    std::array<uint64_t, 3> b{1, 2, 3};
    MutableMaskView{a.data(), 3}.copy_from(MaskView{b.data(), 3});
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
    REQUIRE(v.num_words == 4);
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

TEST_CASE("PauliMaskArena: alloc_zero returns sequential handles") {
    PauliMaskArena arena(128);
    REQUIRE(arena.num_words() == 2);
    REQUIRE(arena.num_masks() == 0);

    REQUIRE(arena.alloc_zero() == 0);
    REQUIRE(arena.alloc_zero() == 1);
    REQUIRE(arena.alloc_zero() == 2);
    REQUIRE(arena.num_masks() == 3);
}

TEST_CASE("PauliMaskArena: per-handle storage is independent") {
    PauliMaskArena arena(256);
    uint32_t h0 = arena.alloc_zero();
    uint32_t h1 = arena.alloc_zero();

    auto m0 = arena.get(h0);
    auto m1 = arena.get(h1);
    m0.mut_x().bit_set(5, true);
    m0.mut_z().bit_set(200, true);
    *m0.sign = 1;

    REQUIRE(m0.view_x().bit_get(5));
    REQUIRE(m0.view_z().bit_get(200));
    REQUIRE(*m0.sign == 1);

    REQUIRE(m1.view_x().is_zero());
    REQUIRE(m1.view_z().is_zero());
    REQUIRE(*m1.sign == 0);
}

TEST_CASE("PauliMaskArena: const get returns read-only view") {
    PauliMaskArena arena(64);
    uint32_t h = arena.alloc_zero();
    arena.get(h).mut_x().bit_set(3, true);

    const PauliMaskArena& carena = arena;
    auto cm = carena.get(h);
    REQUIRE(cm.view_x().bit_get(3));
    REQUIRE(cm.view_z().is_zero());
}

TEST_CASE("PauliMaskArena: width above old compile-time bound") {
    // BitMask<N> caps at CLIFFT_MAX_QUBITS at compile time. The arena
    // must accept arbitrary widths and round up to whole words.
    PauliMaskArena arena(1024);
    REQUIRE(arena.num_words() == 16);
    uint32_t h = arena.alloc_zero();
    arena.get(h).mut_x().bit_set(1023, true);
    REQUIRE(arena.get(h).view_x().bit_get(1023));
    REQUIRE(arena.get(h).view_x().lowest_bit() == 1023);
}

TEST_CASE("PauliMaskArena: zero-width arena is well-formed") {
    PauliMaskArena arena(0);
    REQUIRE(arena.num_words() == 0);
    uint32_t h = arena.alloc_zero();
    auto m = arena.get(h);
    REQUIRE(m.num_words == 0);
    REQUIRE(m.view_x().is_zero());
}
