#include "clifft/optimizer/phase_polynomial.h"
#include "clifft/optimizer/tohpe.h"

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace clifft::phase_detail;

namespace {

unsigned evaluate(const Polynomial& polynomial, uint64_t assignment) {
    unsigned value = 0;
    for (const auto& [monomial, coefficient] : polynomial) {
        if ((assignment & monomial) == monomial) {
            value += coefficient;
        }
    }
    return value & 7;
}

void check_basis_change(const Polynomial& original, uint32_t width,
                        const std::vector<uint64_t>& assignments) {
    auto reduced = original;
    const auto basis = reduce_core(reduced, width);
    REQUIRE(basis.parities.size() == width);
    for (uint64_t assignment : assignments) {
        uint64_t changed = 0;
        for (uint32_t j = 0; j < width; ++j) {
            if (std::popcount(assignment & basis.parities[j]) & 1) {
                changed |= uint64_t{1} << j;
            }
        }
        REQUIRE(evaluate(original, assignment) == evaluate(reduced, changed));
    }
}

}  // namespace

TEST_CASE("Phase polynomial parity expansion agrees with Boolean parity", "[optimizer]") {
    for (uint64_t parity = 1; parity < 32; ++parity) {
        for (int coefficient = -7; coefficient <= 7; ++coefficient) {
            Polynomial polynomial;
            add_parity(polynomial, parity, coefficient);
            for (uint64_t x = 0; x < 32; ++x) {
                REQUIRE(evaluate(polynomial, x) ==
                        unsigned((coefficient * (std::popcount(parity & x) & 1)) & 7));
            }
        }
    }
}

TEST_CASE("Phase polynomial kernel and Pauli derivatives match exhaustive values", "[optimizer]") {
    std::mt19937_64 rng(42191);
    for (unsigned trial = 0; trial < 80; ++trial) {
        const uint32_t width = 1 + rng() % 5;
        const unsigned size = 1U << width;
        Polynomial polynomial;
        for (unsigned j = 0; j < 3 * size; ++j) {
            add_parity(polynomial, rng() % (size - 1) + 1, rng() % 8);
        }
        const auto kernel = clifford_kernel(polynomial, width);
        std::vector<bool> in_kernel(size, false);
        for (unsigned combination = 0; combination < (1U << kernel.size()); ++combination) {
            uint64_t flip = 0;
            for (size_t j = 0; j < kernel.size(); ++j) {
                if ((combination >> j) & 1) {
                    flip ^= kernel[j];
                }
            }
            in_kernel[flip] = true;
        }
        std::vector<uint64_t> assignments;
        for (unsigned flip = 0; flip < size; ++flip) {
            assignments.push_back(flip);
            const unsigned constant = (evaluate(polynomial, flip) - evaluate(polynomial, 0)) & 7;
            bool is_pauli = !(constant & 1);
            unsigned parity = 0;
            for (uint32_t j = 0; j < width; ++j) {
                const unsigned slope = (evaluate(polynomial, flip ^ (1U << j)) -
                                        evaluate(polynomial, 1U << j) - constant) &
                                       7;
                is_pauli &= slope == 0 || slope == 4;
                if (slope == 4) {
                    parity |= 1U << j;
                }
            }
            for (unsigned x = 0; x < size; ++x) {
                is_pauli &= ((evaluate(polynomial, x ^ flip) - evaluate(polynomial, x)) & 7) ==
                            ((constant + 4 * (std::popcount(parity & x) & 1)) & 7);
            }
            REQUIRE(in_kernel[flip] == is_pauli);
            const auto derivative = pauli_derivative(polynomial, flip);
            REQUIRE(derivative.has_value() == is_pauli);
            if (derivative) {
                REQUIRE(derivative->constant == constant);
                REQUIRE(derivative->parity == parity);
            }
        }
        check_basis_change(polynomial, width, assignments);
    }
}

TEST_CASE("Phase polynomial basis changes preserve high variable bits", "[optimizer]") {
    std::mt19937_64 rng(53921);
    for (uint32_t width : {31, 32, 63, 64}) {
        for (unsigned trial = 0; trial < 8; ++trial) {
            Polynomial polynomial;
            const std::vector<uint32_t> support{0, 1, 2, width - 2, width - 1};
            for (unsigned term = 0; term < 64; ++term) {
                uint64_t parity = 0;
                const unsigned selection = rng() % 31 + 1;
                for (size_t j = 0; j < support.size(); ++j) {
                    if ((selection >> j) & 1) {
                        parity |= uint64_t{1} << support[j];
                    }
                }
                add_parity(polynomial, parity, rng() % 8);
            }
            std::vector<uint64_t> assignments;
            for (unsigned sample = 0; sample < 64; ++sample) {
                assignments.push_back(width == 64 ? rng() : rng() & ((uint64_t{1} << width) - 1));
            }
            check_basis_change(polynomial, width, assignments);
        }
    }
}

TEST_CASE("TOHPE synthesis preserves exact phases and bounds non-Clifford terms", "[optimizer]") {
    std::mt19937_64 rng(712409);
    for (uint32_t width : {0, 1, 4, 6, 32, 64}) {
        const uint64_t domain = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
        for (unsigned trial = 0; trial < 30; ++trial) {
            Polynomial terms;
            Polynomial original;
            const auto append = [&](uint64_t parity, unsigned coefficient) {
                if (!parity) {
                    return;
                }
                terms[parity] = static_cast<uint8_t>((terms[parity] + coefficient) & 7);
                add_parity(original, parity, coefficient);
            };
            // Dense parity identities hide a small core behind cancellations;
            // even-weight parities supply a nontrivial Clifford remainder.
            std::vector<uint64_t> axes;
            for (uint32_t j = 0; j < std::min(width, uint32_t{4}); ++j) {
                axes.push_back(uint64_t{1} << (j == 0 ? width - 1 : j - 1));
            }
            for (unsigned mask = 1; mask < (1U << axes.size()); ++mask) {
                uint64_t parity = 0;
                for (size_t j = 0; j < axes.size(); ++j) {
                    if ((mask >> j) & 1) {
                        parity ^= axes[j];
                    }
                }
                append(parity, 1);
            }
            for (unsigned j = 0; j < 8; ++j) {
                const uint64_t parity = rng() & domain;
                append(parity, 2 * (rng() % 4));
                if (trial % 2) {
                    append(parity, rng() % 8);
                }
            }
            if (width) {
                append(uint64_t{1} << (width - 1), trial % 2 ? 7 : 1);
            }
            auto reduced = original;
            const auto basis = reduce_core(reduced, width);
            const auto synthesis = synthesize_parities(terms, reduced, basis);
            size_t input_t = 0;
            size_t output_t = 0;
            for (const auto& [parity, coefficient] : terms) {
                (void)parity;
                input_t += coefficient & 1;
            }
            for (const auto& [parity, coefficient] : synthesis) {
                output_t += coefficient & 1;
                if ((coefficient & 1) && basis.core_width < 64) {
                    REQUIRE((parity >> basis.core_width) == 0);
                }
            }
            REQUIRE(output_t <= input_t);
            const unsigned samples = width <= 6 ? (1U << width) : 64;
            for (unsigned sample = 0; sample < samples; ++sample) {
                const uint64_t assignment = width <= 6 ? sample : rng() & domain;
                uint64_t changed = 0;
                for (uint32_t j = 0; j < width; ++j) {
                    changed |= uint64_t(std::popcount(assignment & basis.parities[j]) & 1) << j;
                }
                unsigned reconstructed = 0;
                for (const auto& [parity, coefficient] : synthesis) {
                    reconstructed += coefficient * (std::popcount(parity & changed) & 1);
                }
                REQUIRE(evaluate(original, assignment) == (reconstructed & 7));
            }
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
            const std::vector<uint64_t> before(distinct.begin(), distinct.end());
            const auto after = tohpe(before, width);
            REQUIRE(after.size() <= before.size());
            if (!trial) {
                REQUIRE(after.size() == width);
            }
            const auto moment = [](const auto& columns, uint64_t mask) {
                unsigned parity = 0;
                for (uint64_t column : columns) {
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
    const std::vector<uint64_t> ccz{1, 2, 3, 4, 5, 6, 7};
    REQUIRE(tohpe(ccz, 3) == ccz);
    std::vector<uint64_t> large;
    for (uint64_t mask = 1; mask <= 129; ++mask) {
        large.push_back(mask);
    }
    REQUIRE(tohpe(large, 8) == large);
    REQUIRE(tohpe({}, 0).empty());
}
