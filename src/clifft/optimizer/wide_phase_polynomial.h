#pragma once

#include "clifft/optimizer/tohpe.h"
#include "clifft/util/bitmask.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

// Optional multiword algebra. The ordinary path retains single-word operations.
namespace clifft::phase_detail::wide {
using Mask = BitMask<128>;
struct Less {
    bool operator()(const Mask& a, const Mask& b) const {
        return a.w[1] != b.w[1] ? a.w[1] < b.w[1] : a.w[0] < b.w[0];
    }
};
// Sparse coefficients modulo eight, indexed by masks of Boolean variables.
// In an expanded polynomial, {0b0101, 4} means 4*x0*x2. Linear coefficients
// are arbitrary, quadratic coefficients are even, and cubic ones are multiples
// of four. The same storage holds XOR parity terms during collection and in
// synthesis results: there, {0b0101, 4} means 4*(x0 XOR x2).
// Zero coefficients are omitted in both representations.
using Polynomial = std::map<Mask, uint8_t, Less>;

// Only expected search limits may discard an optional candidate.
struct SearchLimit {};

struct Budget {
    size_t max_updates = 50000000;
    size_t max_terms = 200000;
    size_t updates = 0;
    bool exhausted = false;

    void charge(size_t amount = 1) {
        if (exhausted || amount > max_updates - updates) {
            exhausted = true;
            throw SearchLimit{};
        }
        updates += amount;
    }

    void add(Polynomial& p, Mask monomial, int coefficient) {
        charge();
        const auto found = p.find(monomial);
        const auto value = uint8_t((coefficient + (found == p.end() ? 0 : found->second)) & 7);
        if (value) {
            if (found == p.end() && p.size() >= max_terms) {
                exhausted = true;
                throw SearchLimit{};
            }
            p[monomial] = value;
        } else if (found != p.end()) {
            p.erase(found);
        }
    }
};

inline Mask bit(uint32_t index) {
    Mask result;
    result.bit_set(index, true);
    return result;
}

inline void add_parity(Polynomial& p, Mask parity, int coefficient, Budget& budget) {
    for (auto a = parity; !a.is_zero(); a.clear_lowest_bit()) {
        const auto first = bit(a.lowest_bit());
        budget.add(p, first, coefficient);
        auto b = a;
        b.clear_lowest_bit();
        for (; !b.is_zero(); b.clear_lowest_bit()) {
            const auto second = bit(b.lowest_bit());
            budget.add(p, first | second, -2 * coefficient);
            auto c = b;
            c.clear_lowest_bit();
            for (; !c.is_zero(); c.clear_lowest_bit()) {
                budget.add(p, first | second | bit(c.lowest_bit()), 4 * coefficient);
            }
        }
    }
}

inline void insert_row(std::map<uint32_t, Mask>& basis, Mask row) {
    while (!row.is_zero()) {
        const uint32_t pivot =
            row.w[1] ? 127 - std::countl_zero(row.w[1]) : 63 - std::countl_zero(row.w[0]);
        const auto found = basis.find(pivot);
        if (found == basis.end()) {
            basis.emplace(pivot, row);
            return;
        }
        row ^= found->second;
    }
}

// The weighted phase signature is the symmetric cubic tensor sum y*y*y
// over odd parity columns y. Its pair-versus-single flattening has the same
// kernel as the finite-difference constraints. Reject an unsupported core
// before materializing a potentially dense cubic polynomial.
inline bool core_fits(const Polynomial& terms, uint32_t width, Budget& budget) {
    std::vector<Mask> rows(width * (width + 1) / 2);
    Mask linear;
    for (const auto& [parity, coefficient] : terms) {
        budget.charge();
        if (!(coefficient & 1))
            continue;
        linear ^= parity;
        for (auto a = parity; !a.is_zero(); a.clear_lowest_bit()) {
            const auto i = a.lowest_bit();
            for (auto b = a; !b.is_zero(); b.clear_lowest_bit()) {
                budget.charge();
                const auto j = b.lowest_bit();
                rows[j * (j + 1) / 2 + i] ^= parity;
            }
        }
    }
    std::map<uint32_t, Mask> pivots;
    insert_row(pivots, linear);
    for (auto row : rows) {
        budget.charge(width + 1);
        insert_row(pivots, row);
        if (pivots.size() > 64)
            return false;
    }
    return true;
}

inline std::vector<Mask> clifford_kernel(const Polynomial& polynomial, uint32_t width,
                                         Budget& budget) {
    std::map<Mask, Mask, Less> constraints;
    // For a degree-d monomial containing j, the finite difference has
    // c*m_without_j - 2*c*m. Its weighted non-Pauli signature is c/2^(d-1)
    // modulo two in both rows; the degree-three second term vanishes mod eight.
    // Building those rows directly avoids one polynomial scan per variable.
    for (const auto& [mask, coefficient] : polynomial) {
        const auto degree = mask.popcount();
        assert(degree <= 3);
        budget.charge();
        if (!degree || !(coefficient & (1U << (degree - 1))))
            continue;
        for (auto bits = mask; !bits.is_zero(); bits.clear_lowest_bit()) {
            const auto variable = bit(bits.lowest_bit());
            budget.charge(2);
            constraints[mask ^ variable] |= variable;
            if (degree < 3)
                constraints[mask] |= variable;
        }
    }
    std::map<uint32_t, Mask> pivots;
    for (const auto& [mask, row] : constraints) {
        (void)mask;
        insert_row(pivots, row);
    }
    std::vector<Mask> kernel;
    for (uint32_t free = 0; free < width; ++free) {
        if (pivots.contains(free)) {
            continue;
        }
        auto vector = bit(free);
        for (const auto& [pivot, row] : pivots) {
            if ((row & vector).popcount() & 1) {
                vector |= bit(pivot);
            }
        }
        kernel.push_back(vector);
    }
    return kernel;
}

struct Basis {
    uint32_t core_width;
    std::vector<Mask> parities;
    std::vector<Mask> core_columns;
};

inline Basis reduce_core(Polynomial& polynomial, uint32_t width, Budget& budget) {
    if (width > 128) {
        throw SearchLimit{};
    }
    std::vector<Mask> generators;
    for (uint32_t j = 0; j < width; ++j) {
        generators.push_back(bit(j));
    }
    const auto kernel = clifford_kernel(polynomial, width, budget);
    std::map<uint32_t, Mask> span;
    for (auto vector : kernel) {
        insert_row(span, vector);
    }
    std::vector<Mask> columns;
    for (uint32_t j = 0; j < width; ++j) {
        const size_t before = span.size();
        insert_row(span, bit(j));
        if (span.size() != before) {
            columns.push_back(bit(j));
        }
    }
    const auto core_width = static_cast<uint32_t>(columns.size());
    // Existing TOHPE synthesis retains its representation and work guards.
    if (core_width > 64) {
        throw SearchLimit{};
    }
    columns.insert(columns.end(), kernel.begin(), kernel.end());
    std::vector<Mask> matrix(width);
    for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            matrix[i].bit_set(j, columns[j].bit_get(i));
        }
    }
    for (uint32_t j = 0; j < width; ++j) {
        uint32_t pivot = j;
        while (!matrix[pivot].bit_get(j)) {
            ++pivot;
            assert(pivot < width);
        }
        if (pivot != j) {
            std::swap(matrix[j], matrix[pivot]);
            std::swap(generators[j], generators[pivot]);
            Polynomial swapped;
            for (const auto& [mask, coefficient] : polynomial) {
                auto exchanged = mask;
                exchanged.bit_swap(j, pivot);
                budget.add(swapped, exchanged, coefficient);
            }
            polynomial = std::move(swapped);
        }
        for (uint32_t i = 0; i < width; ++i) {
            if (i == j || !matrix[i].bit_get(j)) {
                continue;
            }
            matrix[i] ^= matrix[j];
            Polynomial substituted;
            for (const auto& [mask, coefficient] : polynomial) {
                budget.add(substituted, mask, coefficient);
                if (mask.bit_get(i)) {
                    const auto rest = mask ^ bit(i);
                    budget.add(substituted, rest | bit(j), coefficient);
                    budget.add(substituted, rest | bit(j) | bit(i), -2 * coefficient);
                }
            }
            polynomial = std::move(substituted);
            generators[i] ^= generators[j];
        }
    }
    columns.resize(core_width);
    return {core_width, std::move(generators), std::move(columns)};
}

inline Polynomial synthesize(const Polynomial& terms, const Polynomial& reduced, const Basis& basis,
                             Budget& budget) {
    std::map<uint64_t, uint8_t> projected;
    for (const auto& [parity, coefficient] : terms) {
        uint64_t mask = 0;
        for (uint32_t j = 0; j < basis.core_width; ++j) {
            if ((parity & basis.core_columns[j]).popcount() & 1) {
                mask |= uint64_t{1} << j;
            }
        }
        if (mask) {
            projected[mask] = uint8_t((projected[mask] + coefficient) & 7);
        }
    }
    std::vector<uint64_t> columns;
    for (const auto& [parity, coefficient] : projected) {
        if (coefficient & 1) {
            columns.push_back(parity);
        }
    }
    // Precharge a conservative bound for the existing bounded TOHPE search.
    // Its guard skips the cubic search above 128 columns.
    if (columns.size() > basis.core_width && columns.size() <= 128) {
        const auto n = columns.size();
        const auto words = (basis.core_width * (basis.core_width + 1) / 2 + 63) / 64;
        budget.charge(n * n * n * (words + 1));
    }
    Polynomial out;
    auto remainder = reduced;
    for (auto parity : phase_detail::tohpe(std::move(columns), basis.core_width)) {
        budget.add(out, Mask{parity}, 1);
        add_parity(remainder, Mask{parity}, -1, budget);
    }
    for (const auto& [mask, coefficient] : remainder) {
        const auto degree = mask.popcount();
        if (degree == 0) {
            continue;
        }
        if (degree == 1 && coefficient % 2 == 0) {
            budget.add(out, mask, coefficient);
        } else if (degree == 2 && coefficient == 4) {
            const auto first = bit(mask.lowest_bit());
            budget.add(out, first, 2);
            budget.add(out, mask ^ first, 2);
            budget.add(out, mask, -2);
        } else {
            throw std::logic_error("wide synthesis changed the non-Clifford signature");
        }
    }
    return out;
}
}  // namespace clifft::phase_detail::wide
