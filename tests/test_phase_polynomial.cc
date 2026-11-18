#include "clifft/optimizer/phase_polynomial.h"
#include "clifft/optimizer/tohpe.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace clifft::phase_detail;

namespace {

template <size_t N>
using Mask = typename PhasePolynomial<N>::Mask;

template <size_t N>
unsigned evaluate(const std::vector<std::pair<Mask<N>, uint8_t>>& phases, Mask<N> assignment) {
    unsigned value = 0;
    for (const auto& [mask, c] : phases)
        value += c * ((mask & assignment).popcount() & 1);
    return value & 7;
}

template <size_t N>
void check_synthesis(uint32_t width, unsigned trials) {
    std::mt19937_64 rng(712409 + width);
    const auto random_mask = [&] {
        Mask<N> mask;
        for (uint32_t i = 0; i < width; ++i)
            mask.bit_set(i, rng() & 1);
        return mask;
    };
    for (unsigned trial = 0; trial < trials; ++trial) {
        PhasePolynomial<N> phase;
        typename PhasePolynomial<N>::Work work;
        std::vector<std::pair<Mask<N>, uint8_t>> original;
        const auto append = [&](Mask<N> parity, uint8_t coefficient) {
            phase.add(parity, coefficient, work);
            original.emplace_back(parity, coefficient);
        };
        // Fifteen parities on four independent axes cancel modulo eight.
        // Use dense parities that cross the machine-word boundary when available.
        std::array<Mask<N>, 4> axes;
        for (auto& a : axes)
            a = random_mask();
        for (unsigned subset = 1; subset < 16; ++subset) {
            Mask<N> parity;
            for (unsigned j = 0; j < axes.size(); ++j)
                if (subset & (1U << j))
                    parity ^= axes[j];
            append(parity, 1);
        }
        for (unsigned j = 0; j < 8; ++j) {
            append(random_mask(), 2 * (rng() % 4));
            if (trial % 2)
                append(random_mask(), rng() % 8);
        }
        if (width) {
            Mask<N> highest;
            highest.bit_set(width - 1, true);
            append(highest, trial % 2 ? 7 : 1);
        }
        const auto reduced = phase.synthesize(width, work);
        REQUIRE(reduced.t_count <= phase.odd_terms());
        REQUIRE(reduced.core_width <= reduced.t_count);
        const unsigned samples = width <= 6 ? 1U << width : 256;
        for (unsigned i = 0; i < samples; ++i) {
            auto x = width <= 6 ? Mask<N>{i} : random_mask();
            REQUIRE(evaluate<N>(original, x) == evaluate<N>(reduced.phases, x));
        }
    }
}

template <size_t N>
void check_derivatives() {
    std::mt19937_64 rng(42191);
    for (unsigned trial = 0; trial < 80; ++trial) {
        const uint32_t width = 1 + rng() % 5;
        const unsigned size = 1U << width;
        PhasePolynomial<N> phase;
        typename PhasePolynomial<N>::Work work;
        std::vector<std::pair<Mask<N>, uint8_t>> original;
        for (unsigned j = 0; j < 3 * size; ++j) {
            const Mask<N> parity{trial % 2 ? rng() % (size - 1) + 1
                                           : uint64_t{1} << (rng() % width)};
            const uint8_t c = rng() % 8;
            phase.add(parity, c, work);
            original.emplace_back(parity, c);
        }
        const auto axes = phase.core_axes(width, work);
        for (unsigned flip = 0; flip < size; ++flip) {
            const auto delta = [&](unsigned x) {
                return (evaluate<N>(original, {x ^ flip}) - evaluate<N>(original, {x})) & 7;
            };
            const unsigned constant = delta(0);
            Mask<N> z;
            bool is_pauli = !(constant & 1);
            for (unsigned i = 0; i < width; ++i) {
                const unsigned slope = (delta(1U << i) - constant) & 7;
                is_pauli &= slope == 0 || slope == 4;
                z.bit_set(i, slope == 4);
            }
            for (unsigned x = 0; x < size; ++x)
                is_pauli &= delta(x) == ((constant + 4 * ((z & Mask<N>{x}).popcount() & 1)) & 7);
            REQUIRE(std::ranges::none_of(axes, [&](auto axis) {
                        return (axis & Mask<N>{flip}).popcount() & 1;
                    }) == is_pauli);
            const auto derivative = phase.derivative(Mask<N>{flip}, work);
            REQUIRE(derivative.has_value() == is_pauli);
            if (derivative) {
                REQUIRE(derivative->parity == z);
                REQUIRE(derivative->constant == constant);
            }
        }
    }
}

}  // namespace

TEST_CASE("Phase kernels and observer signs agree with exhaustive eigenphases", "[optimizer]") {
    check_derivatives<64>();
    check_derivatives<128>();
}
TEST_CASE("Parity synthesis preserves eigenphases across coordinate widths", "[optimizer]") {
    for (uint32_t width : {0, 1, 4, 6, 31, 32, 63, 64})
        check_synthesis<64>(width, 8);
    for (uint32_t width : {65, 87, 128})
        check_synthesis<128>(width, 8);
}

TEST_CASE("A dense dagger rotation stays compact after canceling parity cubes", "[optimizer]") {
    PhasePolynomial<128> phase;
    WideBudget work;
    for (uint32_t start = 0; start < 128; start += 4) {
        for (uint32_t subset = 1; subset < 16; ++subset) {
            Mask<128> mask;
            for (uint32_t i = 0; i < 4; ++i)
                mask.bit_set(start + i, subset & (1U << i));
            phase.add(mask, 1, work);
        }
    }
    // Once the cubes cancel, pulling back an observer depends only on the
    // flipped axes. All 128 pullbacks fit in less work than one raw table scan.
    const auto identity = phase.linear_reduction(128, work);
    REQUIRE(identity);
    REQUIRE(identity->phases.empty());
    WideBudget observers;
    observers.max_updates = phase.terms().size() - 1;
    for (uint32_t i = 0; i < 128; ++i) {
        Mask<128> flip;
        flip.bit_set(i, true);
        const auto difference = phase.derivative(flip, observers);
        REQUIRE(difference);
        REQUIRE(difference->parity.is_zero());
        REQUIRE(difference->constant == 0);
    }
    const auto dense = ~Mask<128>{};
    phase.add(dense, -1, work);
    const auto result = phase.synthesize(128, work);
    REQUIRE(result.core_width == 1);
    REQUIRE(result.phases.size() == 1);
    REQUIRE(result.phases.front().first == dense);
    REQUIRE(result.phases.front().second == 7);
}

TEST_CASE("Two-word nonlinear cores preserve eigenphases on each side of the column limit",
          "[optimizer]") {
    for (uint32_t width : {68U, 128U}) {
        CAPTURE(width);
        PhasePolynomial<128> phase;
        WideBudget work;
        for (uint32_t i = 0; i < width; ++i) {
            Mask<128> axis;
            axis.bit_set(i, true);
            phase.add(axis, 1, work);
        }
        // Cancel a parity cube spanning both words. The extra S rotation keeps
        // the phase nonlinear even after the cube cancels.
        const std::array<uint32_t, 4> axes{0, 1, width - 2, width - 1};
        for (unsigned subset = 1; subset < 16; ++subset) {
            Mask<128> parity;
            for (size_t j = 0; j < axes.size(); ++j)
                parity.bit_set(axes[j], subset & (1U << j));
            phase.add(parity, 1, work);
        }
        Mask<128> ends;
        ends.bit_set(0, true);
        ends.bit_set(width - 1, true);
        phase.add(ends, 2, work);
        REQUIRE(phase.odd_terms() > width);
        const auto reduced = phase.synthesize(width, work);
        REQUIRE(reduced.core_width == width);
        REQUIRE(reduced.t_count == (width == 68 ? width : phase.odd_terms()));
        std::mt19937_64 rng(218012);
        for (unsigned sample = 0; sample < 1024; ++sample) {
            Mask<128> x{rng()};
            x.w[1] = width == 128 ? rng() : rng() & 15;
            const unsigned expected = (x.popcount() + 2 * ((x & ends).popcount() & 1)) & 7;
            REQUIRE(evaluate<128>(reduced.phases, x) == expected);
        }
    }
}

TEST_CASE("TOHPE preserves cubic signatures and handles high coordinate bits", "[optimizer]") {
    std::mt19937_64 rng(861193);
    for (uint32_t width : {5, 9, 64}) {
        for (unsigned trial = 0; trial < 8; ++trial) {
            const uint64_t domain = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
            std::set<uint64_t> distinct;
            for (uint32_t i = 0; i < width; ++i) {
                distinct.insert(uint64_t{1} << i);
            }
            // Adding all parities on four independent axes is a Clifford
            // identity, so the table has a known smaller representation.
            const std::vector<uint32_t> axes{0, 1, width - 2, width - 1};
            for (unsigned mask = 1; mask < 16; ++mask) {
                uint64_t parity = 0;
                for (size_t i = 0; i < axes.size(); ++i) {
                    if ((mask >> i) & 1) {
                        parity |= uint64_t{1} << axes[i];
                    }
                }
                if (!distinct.erase(parity)) {
                    distinct.insert(parity);
                }
            }
            if (trial) {
                for (unsigned i = 0; i < 24; ++i) {
                    const uint64_t parity = rng() & domain;
                    if (parity) {
                        distinct.insert(parity);
                    }
                }
            }
            const std::vector<Mask<64>> before(distinct.begin(), distinct.end());
            const auto after = tohpe(before, width);
            REQUIRE(after.size() <= before.size());
            if (!trial) {
                REQUIRE(after.size() == width);
            }
            const auto moment = [](const auto& columns, uint64_t mask) {
                unsigned parity = 0;
                for (auto column : columns) {
                    parity ^= (column & mask) == mask;
                }
                return parity;
            };
            for (uint32_t a = 0; a < width; ++a) {
                for (uint32_t b = a; b < width; ++b) {
                    for (uint32_t c = b; c < width; ++c) {
                        const uint64_t mask =
                            (uint64_t{1} << a) | (uint64_t{1} << b) | (uint64_t{1} << c);
                        REQUIRE(moment(before, mask) == moment(after, mask));
                    }
                }
            }
        }
    }
}

TEST_CASE("TOHPE bounds large searches and keeps irreducible cubic phases", "[optimizer]") {
    // The seven-term CCZ signature admits only the unhelpful odd full-table
    // dependency. It must terminate without changing the phase.
    const std::vector<Mask<64>> ccz{1, 2, 3, 4, 5, 6, 7};
    REQUIRE(tohpe(ccz, 3) == ccz);
    std::vector<Mask<64>> large;
    for (uint64_t mask = 1; mask <= 129; ++mask) {
        large.push_back(mask);
    }
    REQUIRE(tohpe(large, 8) == large);
    REQUIRE(tohpe<64>({}, 0).empty());
}
