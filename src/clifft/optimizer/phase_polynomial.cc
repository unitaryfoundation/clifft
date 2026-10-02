#include "clifft/optimizer/phase_polynomial.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <utility>

namespace clifft::phase_detail {

namespace {

void add(Polynomial& p, uint64_t monomial, int coefficient) {
    const auto found = p.find(monomial);
    const uint8_t value =
        static_cast<uint8_t>((coefficient + (found == p.end() ? 0 : found->second)) & 7);
    if (value) {
        p[monomial] = value;
    } else if (found != p.end()) {
        p.erase(found);
    }
}

void insert_binary_row(std::map<uint32_t, uint64_t>& basis, uint64_t row) {
    while (row) {
        const uint32_t pivot = 63 - std::countl_zero(row);
        auto found = basis.find(pivot);
        if (found == basis.end()) {
            basis.emplace(pivot, row);
            return;
        }
        row ^= found->second;
    }
}

void substitute_cx(Polynomial& p, uint64_t control, uint64_t target) {
    Polynomial out;
    for (const auto& [mask, c] : p) {
        add(out, mask, c);
        if (mask & target) {
            const uint64_t rest = mask ^ target;
            add(out, rest | control, c);
            add(out, rest | control | target, -2 * c);
        }
    }
    p = std::move(out);
}

Polynomial derivative(Polynomial p, uint64_t flip) {
    const auto original = p;
    while (flip) {
        const uint64_t bit = flip & -flip;
        Polynomial out;
        for (const auto& [mask, c] : p) {
            if (mask & bit) {
                add(out, mask ^ bit, c);
                add(out, mask, -c);
            } else {
                add(out, mask, c);
            }
        }
        p = std::move(out);
        flip &= flip - 1;
    }
    for (const auto& [mask, c] : original) {
        add(p, mask, -c);
    }
    return p;
}

}  // namespace

void add_parity(Polynomial& p, uint64_t parity, int coefficient) {
    // XOR has coefficients 1, -2, 4. Higher degrees vanish modulo eight.
    for (uint64_t bits = parity; bits; bits &= bits - 1) {
        const uint64_t a = bits & -bits;
        add(p, a, coefficient);
        for (uint64_t pairs = bits & (bits - 1); pairs; pairs &= pairs - 1) {
            const uint64_t b = pairs & -pairs;
            add(p, a | b, -2 * coefficient);
            for (uint64_t triples = pairs & (pairs - 1); triples; triples &= triples - 1) {
                add(p, a | b | (triples & -triples), 4 * coefficient);
            }
        }
    }
}

std::vector<uint64_t> clifford_kernel(const Polynomial& p, uint32_t width) {
    // Divide degree-d coefficients of each single-bit difference by 2^d,
    // then reduce modulo two. Weighted degree makes these obstructions linear
    // in the translation, so their nullspace gives exactly the Pauli derivatives.
    std::map<uint64_t, uint64_t> constraints;
    for (uint32_t j = 0; j < width; ++j) {
        const uint64_t bit = uint64_t{1} << j;
        Polynomial derivative;
        for (const auto& [mask, c] : p) {
            if (mask & bit) {
                add(derivative, mask ^ bit, c);
                add(derivative, mask, -2 * c);
            }
        }
        for (const auto& [mask, c] : derivative) {
            const auto degree = std::popcount(mask);
            const bool non_pauli = degree == 0   ? (c & 1) != 0
                                   : degree == 1 ? ((c / 2) & 1) != 0
                                                 : ((c / 4) & 1) != 0;
            assert(degree <= 2);
            if (non_pauli) {
                constraints[mask] |= bit;
            }
        }
    }
    std::map<uint32_t, uint64_t> pivots;
    for (const auto& [mask, row] : constraints) {
        (void)mask;
        insert_binary_row(pivots, row);
    }
    std::vector<uint64_t> kernel;
    for (uint32_t free = 0; free < width; ++free) {
        if (pivots.contains(free)) {
            continue;
        }
        uint64_t vector = uint64_t{1} << free;
        for (const auto& [pivot, row] : pivots) {
            if (std::popcount(row & vector) & 1) {
                vector |= uint64_t{1} << pivot;
            }
        }
        kernel.push_back(vector);
    }
    return kernel;
}

CoreBasis reduce_core(Polynomial& p, uint32_t width) {
    assert(width <= 64);
    std::vector<uint64_t> generators;
    for (uint32_t j = 0; j < width; ++j) {
        generators.push_back(uint64_t{1} << j);
    }
    const auto kernel = clifford_kernel(p, width);
    std::map<uint32_t, uint64_t> span;
    for (uint64_t vector : kernel) {
        insert_binary_row(span, vector);
    }
    std::vector<uint64_t> columns;
    for (uint32_t j = 0; j < width; ++j) {
        const size_t before = span.size();
        insert_binary_row(span, uint64_t{1} << j);
        if (span.size() != before) {
            columns.push_back(uint64_t{1} << j);
        }
    }
    const uint32_t core_width = static_cast<uint32_t>(columns.size());
    columns.insert(columns.end(), kernel.begin(), kernel.end());
    std::vector<uint64_t> matrix(width, 0);
    for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            if (columns[j] & (uint64_t{1} << i)) {
                matrix[i] |= uint64_t{1} << j;
            }
        }
    }
    for (uint32_t j = 0; j < width; ++j) {
        uint32_t pivot = j;
        while (!(matrix[pivot] & (uint64_t{1} << j))) {
            ++pivot;
            assert(pivot < width);
        }
        if (pivot != j) {
            std::swap(matrix[j], matrix[pivot]);
            std::swap(generators[j], generators[pivot]);
            Polynomial swapped;
            const uint64_t left = uint64_t{1} << j;
            const uint64_t right = uint64_t{1} << pivot;
            for (const auto& [mask, c] : p) {
                const bool different = bool(mask & left) != bool(mask & right);
                add(swapped, different ? mask ^ (left | right) : mask, c);
            }
            p = std::move(swapped);
        }
        for (uint32_t i = 0; i < width; ++i) {
            if (i != j && (matrix[i] & (uint64_t{1} << j))) {
                matrix[i] ^= matrix[j];
                substitute_cx(p, uint64_t{1} << j, uint64_t{1} << i);
                // Track the dual eigenbit parities alongside the substitution.
                generators[i] ^= generators[j];
            }
        }
    }
    columns.resize(core_width);
    return {core_width, std::move(generators), std::move(columns)};
}

Polynomial synthesize_parities(const Polynomial& p) {
    Polynomial out;
    for (const auto& [mask, c] : p) {
        const int degree = std::popcount(mask);
        if (degree == 0) {
            continue;
        }
        if (degree == 1) {
            add(out, mask, c);
        } else if (degree == 2) {
            assert(c % 2 == 0);
            const uint64_t a = mask & -mask;
            const uint64_t b = mask ^ a;
            add(out, a, c / 2);
            add(out, b, c / 2);
            add(out, mask, -c / 2);
        } else {
            assert(degree == 3 && c == 4);
            for (uint64_t subset = mask; subset; subset = (subset - 1) & mask) {
                add(out, subset, (std::popcount(subset) & 1) ? 1 : -1);
            }
        }
    }
    return out;
}

Polynomial project_parities(const Polynomial& terms, const Polynomial& reduced,
                            const CoreBasis& basis) {
    Polynomial out;
    auto remainder = reduced;
    for (const auto& [parity, coefficient] : terms) {
        uint64_t projected = 0;
        for (uint32_t j = 0; j < basis.core_width; ++j) {
            if (std::popcount(parity & basis.core_columns[j]) & 1) {
                projected |= uint64_t{1} << j;
            }
        }
        if (projected) {
            add(out, projected, coefficient);
            add_parity(remainder, projected, -int(coefficient));
        }
    }
    // Kernel directions have Pauli derivatives, so all remaining dependence
    // on them is Clifford. This identity avoids expanding the non-Clifford core.
    for (const auto& [mask, coefficient] : remainder) {
        const auto degree = std::popcount(mask);
        assert(degree == 0 || (degree == 1 && coefficient % 2 == 0) ||
               (degree == 2 && coefficient % 4 == 0));
    }
    for (const auto& [parity, coefficient] : synthesize_parities(remainder)) {
        assert(coefficient % 2 == 0);
        add(out, parity, coefficient);
    }
    return out;
}

std::optional<PauliDerivative> pauli_derivative(const Polynomial& p, uint64_t flip) {
    const auto difference = derivative(p, flip);
    int constant = 0;
    uint64_t z = 0;
    for (const auto& [mask, c] : difference) {
        if (!mask) {
            constant = c;
        } else if (std::popcount(mask) == 1 && c == 4) {
            z |= mask;
        } else {
            return std::nullopt;
        }
    }
    if (constant & 1) {
        return std::nullopt;
    }
    return PauliDerivative{z, static_cast<uint8_t>(constant)};
}

}  // namespace clifft::phase_detail
