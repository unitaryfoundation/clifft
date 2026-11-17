#include "clifft/util/bitmask.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_arena.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using clifft::BitMask;
using clifft::for_each_set_bit;
using clifft::lowest_bit_at_or_above;
using clifft::lowest_bit_below;
using clifft::MaskView;
using clifft::mut_view;
using clifft::MutableMaskView;
using clifft::PauliMask;
using clifft::PauliMaskArena;
using clifft::PauliMaskHandle;
using clifft::PauliMaskView;
using clifft::view;

TEST_CASE("Pauli commutation ranges agree with scalar symplectic products") {
    std::mt19937_64 rng(71829);
    constexpr size_t words = 5;
    for (size_t begin = 0; begin <= words; ++begin) {
        for (size_t end = begin; end <= words; ++end) {
            for (size_t trial = 0; trial < 8; ++trial) {
                std::array<uint64_t, words> ax{}, az{}, bx{}, bz{};
                for (size_t w = 0; w < words; ++w) {
                    if (w >= begin && w < end) {
                        ax[w] = rng();
                        az[w] = rng();
                    }
                    bx[w] = rng();
                    bz[w] = rng();
                }
                bool odd = false;
                for (size_t q = 0; q < words * 64; ++q) {
                    odd ^= ((ax[q / 64] >> (q % 64)) & 1) && ((bz[q / 64] >> (q % 64)) & 1);
                    odd ^= ((az[q / 64] >> (q % 64)) & 1) && ((bx[q / 64] >> (q % 64)) & 1);
                }
                const clifft::PauliCommutation check(MaskView{ax}, MaskView{az});
                CHECK(check.anti_commutes(MaskView{bx}, MaskView{bz}) == odd);
            }
        }
    }
    const clifft::PauliCommutation empty(MaskView{}, MaskView{});
    CHECK_FALSE(empty.anti_commutes(MaskView{}, MaskView{}));
}

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
// Sparse Pauli masks
// =========================================================================

namespace {

struct DensePauli {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;
};

// A random Pauli that leaves most words empty, like a local operator on a
// wide register.
DensePauli random_sparse_pauli(std::mt19937_64& rng, size_t words) {
    DensePauli pauli{std::vector<uint64_t>(words, 0), std::vector<uint64_t>(words, 0)};
    for (size_t w = 0; w < words; ++w) {
        if (rng() % 3 == 0) {
            pauli.x[w] = rng() & rng();
            pauli.z[w] = rng() & rng();
        }
    }
    return pauli;
}

PauliMask sparse(const DensePauli& pauli, bool sign = false) {
    return PauliMask(MaskView{pauli.x}, MaskView{pauli.z}, sign);
}

}  // namespace

TEST_CASE("PauliMask stores only the nonzero words of a dense Pauli") {
    std::vector<uint64_t> x(80, 0);
    std::vector<uint64_t> z(80, 0);
    x[0] = 1ULL << 5;
    z[78] = 1ULL << 7;
    const PauliMask mask(MaskView{x}, MaskView{z}, true);
    const PauliMaskView view = mask;

    REQUIRE(view.words().size() == 2);
    REQUIRE(view.sign());
    REQUIRE_FALSE(view.is_identity());
    REQUIRE(view.begin_word() == 0);
    REQUIRE(view.end_word() == 79);
    REQUIRE(view.x().bit_get(5));
    REQUIRE_FALSE(view.z().bit_get(5));
    REQUIRE(view.z().bit_get(78 * 64 + 7));
    REQUIRE_FALSE(view.x().bit_get(40 * 64));
    REQUIRE(view.x().popcount() == 1);
    REQUIRE(view.x() == MaskView{x});
    REQUIRE(view.z() == MaskView{z});

    // The dense round trip restores every word.
    std::vector<uint64_t> rx(80, 0);
    std::vector<uint64_t> rz(80, 0);
    view.xor_into(MutableMaskView{rx}, MutableMaskView{rz});
    REQUIRE(rx == x);
    REQUIRE(rz == z);
}

TEST_CASE("PauliMask identity has no words and keeps its sign") {
    const std::vector<uint64_t> zeros(4, 0);
    const PauliMask mask(MaskView{zeros}, MaskView{zeros}, true);
    REQUIRE(mask.view().is_identity());
    REQUIRE(mask.view().sign());
    REQUIRE(mask.view().x().is_zero());
    REQUIRE(mask.view().begin_word() == mask.view().end_word());
    REQUIRE_FALSE(mask.view() == PauliMask().view());
    REQUIRE(same_body(mask, PauliMask()));
}

TEST_CASE("PauliMask toggle keeps words sorted and drops emptied words") {
    PauliMask mask;
    mask.toggle(700, true, false);
    mask.toggle(3, false, true);
    mask.toggle(130, true, true);
    REQUIRE(mask.view().words().size() == 3);
    REQUIRE(std::ranges::is_sorted(mask.view().words(), {}, &clifft::PauliWord::index));
    REQUIRE(mask.view().x().bit_get(700));
    REQUIRE(mask.view().z().bit_get(3));
    REQUIRE(mask.view().x().bit_get(130));
    REQUIRE(mask.view().z().bit_get(130));

    mask.toggle(130, true, false);
    REQUIRE(mask.view().words().size() == 3);
    REQUIRE(mask.view().z().bit_get(130));
    mask.toggle(130, false, true);
    REQUIRE(mask.view().words().size() == 2);
    REQUIRE_FALSE(mask.view().z().bit_get(130));
}

TEST_CASE("Sparse Pauli algebra agrees with dense word arithmetic") {
    std::mt19937_64 rng(90417);
    constexpr size_t words = 9;
    for (size_t trial = 0; trial < 200; ++trial) {
        const DensePauli a = random_sparse_pauli(rng, words);
        const DensePauli b = random_sparse_pauli(rng, words);
        const PauliMask sa = sparse(a);
        const PauliMask sb = sparse(b);

        const bool dense_anti =
            clifft::anti_commute(MaskView{a.x}, MaskView{a.z}, MaskView{b.x}, MaskView{b.z});
        CHECK(clifft::anti_commute(sa.view(), sb.view()) == dense_anti);
        CHECK(clifft::anti_commute(sa.view(), MaskView{b.x}, MaskView{b.z}) == dense_anti);

        bool dense_overlap = false;
        DensePauli product = a;
        for (size_t w = 0; w < words; ++w) {
            dense_overlap |= ((a.x[w] | a.z[w]) & (b.x[w] | b.z[w])) != 0;
            product.x[w] ^= b.x[w];
            product.z[w] ^= b.z[w];
        }
        CHECK(clifft::supports_overlap(sa, sb) == dense_overlap);

        PauliMask multiplied = sa;
        multiplied.multiply_body(sb);
        CHECK(same_body(multiplied, sparse(product)));
        // Multiplying twice by the same Pauli restores the original body.
        multiplied.multiply_body(sb);
        CHECK(same_body(multiplied, sa));

        CHECK((sa.view().x() == sb.view().x()) == (a.x == b.x));
        CHECK((sa.view() == sb.view()) == (a.x == b.x && a.z == b.z));
    }
}

TEST_CASE("DensePauliScratch holds exactly the last loaded Pauli") {
    std::mt19937_64 rng(5521);
    constexpr size_t words = 12;
    clifft::DensePauliScratch scratch(words);
    for (size_t trial = 0; trial < 50; ++trial) {
        const DensePauli pauli = random_sparse_pauli(rng, words);
        const PauliMask mask = sparse(pauli);
        scratch.load(mask);
        REQUIRE(scratch.x() == MaskView{pauli.x});
        REQUIRE(scratch.z() == MaskView{pauli.z});
        REQUIRE(scratch.begin_word() == mask.view().begin_word());
        REQUIRE(scratch.end_word() == mask.view().end_word());
    }
    scratch.load(PauliMaskView{});
    REQUIRE(scratch.x().is_zero());
    REQUIRE(scratch.z().is_zero());
}

// =========================================================================
// PauliMaskArena
// =========================================================================

TEST_CASE("PauliMaskArena: appended masks read back unchanged") {
    std::mt19937_64 rng(31337);
    constexpr size_t words = 20;
    PauliMaskArena arena(words * 64);
    REQUIRE(arena.num_words() == words);
    REQUIRE(arena.size() == 0);

    std::vector<PauliMask> expected;
    std::vector<PauliMaskHandle> handles;
    for (size_t i = 0; i < 3000; ++i) {
        expected.push_back(sparse(random_sparse_pauli(rng, words), (rng() & 1) != 0));
        handles.push_back(arena.append(expected.back()));
    }
    REQUIRE(arena.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(arena.at(handles[i]) == expected[i].view());
    }
}

TEST_CASE("PauliMaskArena: views stay valid while other masks change") {
    std::mt19937_64 rng(77);
    constexpr size_t words = 6;
    PauliMaskArena arena(words * 64);
    const PauliMask first = sparse(random_sparse_pauli(rng, words), true);
    const PauliMaskHandle first_handle = arena.append(first);
    const PauliMaskView held = arena.at(first_handle);

    // Enough appends and growing reassignments to open many new chunks.
    std::vector<PauliMaskHandle> others;
    for (size_t i = 0; i < 20000; ++i) {
        others.push_back(arena.append(sparse(random_sparse_pauli(rng, words))));
    }
    for (PauliMaskHandle h : others) {
        PauliMask grown(arena.at(h));
        for (uint32_t w = 0; w < words; ++w) {
            grown.toggle(w * 64 + 1, true, false);
        }
        arena.assign(h, grown);
    }
    REQUIRE(held == first.view());
    REQUIRE(arena.at(first_handle) == first.view());
}

TEST_CASE("PauliMaskArena: reassignment replaces one mask only") {
    PauliMaskArena arena(256);
    const PauliMask a(uint64_t{0b101}, uint64_t{0}, false);
    PauliMask wide;
    wide.toggle(5, true, false);
    wide.toggle(70, false, true);
    wide.toggle(200, true, true);
    const PauliMaskHandle ha = arena.append(a);
    const PauliMaskHandle hb = arena.append(wide);
    const PauliMaskHandle hc = arena.append(PauliMask());

    // Shrink, grow, and clear in turn; the neighbours never change.
    arena.assign(hb, a);
    REQUIRE(arena.at(hb) == a.view());
    arena.assign(hc, wide);
    REQUIRE(arena.at(hc) == wide.view());
    arena.assign(ha, PauliMask());
    REQUIRE(arena.at(ha).is_identity());
    REQUIRE(arena.at(hb) == a.view());
    REQUIRE(arena.at(hc) == wide.view());

    arena.assign(hb, wide);
    REQUIRE(arena.at(hb) == wide.view());
    REQUIRE(arena.at(hc) == wide.view());

    // Assigning a mask its own view changes at most the sign.
    arena.assign(hc, PauliMaskView{arena.at(hc).words(), true});
    REQUIRE(same_body(arena.at(hc), wide));
    REQUIRE(arena.at(hc).sign());
}

TEST_CASE("PauliMaskArena: storage stays proportional to masks that grow word by word") {
    // Each mask gains one word per step, as downstream masks do when
    // successive Clifford absorptions widen them.
    constexpr uint32_t kWords = 200;
    constexpr uint32_t kMasks = 64;
    constexpr uint32_t kSteps = 128;
    PauliMaskArena arena(kWords * 64);
    std::vector<PauliMask> expected(kMasks);
    std::vector<PauliMaskHandle> handles;
    for (uint32_t m = 0; m < kMasks; ++m) {
        expected[m].toggle(m, true, false);
        handles.push_back(arena.append(expected[m]));
    }
    for (uint32_t step = 1; step <= kSteps; ++step) {
        for (uint32_t m = 0; m < kMasks; ++m) {
            expected[m].toggle(64 * step + m, false, true);
            arena.assign(handles[m], expected[m]);
        }
    }

    size_t live_words = 0;
    for (uint32_t m = 0; m < kMasks; ++m) {
        REQUIRE(arena.at(handles[m]) == expected[m].view());
        live_words += expected[m].view().words().size();
    }
    REQUIRE(live_words == size_t{kMasks} * (kSteps + 1));
    // Reallocating to the exact size at every step would hold about
    // kSteps / 2 times the live words.
    REQUIRE(arena.allocated_words() <= 4 * live_words);
    // A copy keeps only the live words.
    REQUIRE(PauliMaskArena(arena).allocated_words() == live_words);
}

TEST_CASE("PauliMaskArena: set_sign leaves the body alone") {
    PauliMaskArena arena(64);
    const PauliMaskHandle h = arena.append(PauliMask(uint64_t{8}, uint64_t{1}, false));
    arena.set_sign(h, true);
    REQUIRE(arena.at(h).sign());
    REQUIRE(arena.at(h).x().bit_get(3));
    REQUIRE(arena.at(h).z().bit_get(0));
    arena.set_sign(h, false);
    REQUIRE(arena.at(h) == PauliMask(uint64_t{8}, uint64_t{1}, false).view());
}

TEST_CASE("PauliMaskArena: copies are independent and equal") {
    std::mt19937_64 rng(4242);
    constexpr size_t words = 5;
    PauliMaskArena original(words * 64);
    std::vector<PauliMaskHandle> handles;
    for (size_t i = 0; i < 500; ++i) {
        handles.push_back(
            original.append(sparse(random_sparse_pauli(rng, words), (rng() & 1) != 0)));
    }
    PauliMask grown(original.at(handles[3]));
    grown.toggle(64 * 4 + 9, true, true);

    // Copy once from a freshly built arena and once after a reassignment has
    // moved a mask to new storage.
    for (bool reassigned : {false, true}) {
        CAPTURE(reassigned);
        if (reassigned) {
            original.assign(handles[3], grown);
        }
        PauliMaskArena copy = original;
        REQUIRE(copy.size() == original.size());
        REQUIRE(copy.num_words() == original.num_words());
        for (PauliMaskHandle h : handles) {
            REQUIRE(copy.at(h) == original.at(h));
        }

        const PauliMask kept(original.at(handles[0]));
        copy.assign(handles[0], PauliMask());
        const PauliMaskHandle appended = copy.append(grown);
        REQUIRE(copy.at(handles[0]).is_identity());
        REQUIRE(copy.at(appended) == grown.view());
        REQUIRE(original.size() == handles.size());
        REQUIRE(original.at(handles[0]) == kept.view());
    }
}

TEST_CASE("PauliMaskArena: zero-width and empty arenas are well-formed") {
    PauliMaskArena arena;
    REQUIRE(arena.num_words() == 0);
    REQUIRE(arena.size() == 0);
    const PauliMaskHandle h = arena.append(PauliMaskView{{}, true});
    REQUIRE(arena.at(h).is_identity());
    REQUIRE(arena.at(h).sign());
    const PauliMaskArena copy = arena;
    REQUIRE(copy.at(h) == arena.at(h));
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

TEST_CASE("for_each_set_bit visits set bits in ascending order across word ranges") {
    std::array<uint64_t, 4> a{
        (1ULL << 0) | (1ULL << 63),
        0,
        (1ULL << 5) | (1ULL << 40),
        1ULL << 63,
    };
    const MaskView mv = mv_from(a);

    std::vector<uint32_t> all_bits;
    for_each_set_bit(mv, [&](uint32_t b) { all_bits.push_back(b); });
    REQUIRE(all_bits == std::vector<uint32_t>{0, 63, 128 + 5, 128 + 40, 192 + 63});

    std::vector<uint32_t> subrange_bits;
    for_each_set_bit(mv, 1, 3, [&](uint32_t b) { subrange_bits.push_back(b); });
    REQUIRE(subrange_bits == std::vector<uint32_t>{128 + 5, 128 + 40});

    std::vector<uint32_t> empty_bits;
    for_each_set_bit(mv, 2, 2, [&](uint32_t b) { empty_bits.push_back(b); });
    REQUIRE(empty_bits.empty());
}
