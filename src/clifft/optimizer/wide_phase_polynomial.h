#pragma once

#include "clifft/optimizer/tohpe.h"
#include "clifft/util/bitmask.h"

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

// Phase synthesis on XOR parity tables with up to 128 independent variables.
namespace clifft::phase_detail::wide {
using Mask = BitMask<128>;
struct Less {
    bool operator()(const Mask& a, const Mask& b) const {
        return a.w[1] != b.w[1] ? a.w[1] < b.w[1] : a.w[0] < b.w[0];
    }
};
// Each mask selects an XOR of eigenbits, with its phase coefficient modulo eight.
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

    void add(Polynomial& p, Mask parity, int coefficient) {
        charge();
        const auto found = p.find(parity);
        const auto value = uint8_t((coefficient + (found == p.end() ? 0 : found->second)) & 7);
        if (value) {
            if (found == p.end() && p.size() >= max_terms) {
                exhausted = true;
                throw SearchLimit{};
            }
            p[parity] = value;
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

// A parity with weight c contributes c to each selected bit and -2c to each
// selected pair in the Boolean polynomial modulo eight.
// For each pair (i,j), odd_columns records the XOR of the odd parity masks
// containing both bits. Bit k of this row is the cubic signature entry ijk.
// This packs all third indices into one two-word row per pair.
struct PhaseSignature {
    struct Pair {
        uint8_t weight = 0;  // Sum of parity coefficients modulo four.
        Mask odd_columns;
    };
    std::array<uint8_t, Mask::kBits> linear{};
    std::map<std::pair<uint32_t, uint32_t>, Pair> pairs;

    void add(Mask parity, int coefficient, Budget& budget) {
        for (auto bits = parity; !bits.is_zero();) {
            const uint32_t i = bits.lowest_bit();
            bits.clear_lowest_bit();
            budget.charge();
            auto& weight = linear.at(i);
            weight = (weight + coefficient) & 7;
            if (!(coefficient & 3))
                continue;
            budget.charge(bits.popcount());
            for (auto rest = bits; !rest.is_zero(); rest.clear_lowest_bit()) {
                const auto key = std::pair{i, rest.lowest_bit()};
                const auto row = pairs.try_emplace(key).first;
                auto& pair = row->second;
                pair.weight = (pair.weight + coefficient) & 3;
                if (coefficient & 1) {
                    pair.odd_columns ^= parity;
                }
                if (!pair.weight && pair.odd_columns.is_zero()) {
                    pairs.erase(row);
                }
            }
        }
    }

    // A translation is a Clifford direction when its finite difference is a
    // Pauli phase. These constraints use the pair-versus-single flattening
    // of the symmetric signature, including the repeated-index rows (i,i).
    std::vector<Mask> clifford_kernel(uint32_t width, Budget& budget) const {
        assert(width <= Mask::kBits);
        std::map<uint32_t, Mask> pivots;
        std::vector<Mask> diagonal(width);
        Mask constant;
        for (uint32_t i = 0; i < width; ++i) {
            if (linear[i] & 1) {
                constant |= bit(i);
                diagonal[i] = bit(i);
            }
        }
        insert_row(pivots, constant);
        for (const auto& [indices, pair] : pairs) {
            const auto [i, j] = indices;
            if (pair.weight & 1) {
                diagonal[i] |= bit(j);
                diagonal[j] |= bit(i);
            }
            budget.charge(width + 1);
            insert_row(pivots, pair.odd_columns);
        }
        for (auto row : diagonal) {
            budget.charge(width + 1);
            insert_row(pivots, row);
        }
        std::vector<Mask> kernel;
        for (uint32_t free = 0; free < width; ++free) {
            if (pivots.contains(free))
                continue;
            auto vector = bit(free);
            for (const auto& [pivot, row] : pivots) {
                if ((row & vector).popcount() & 1)
                    vector |= bit(pivot);
            }
            kernel.push_back(vector);
        }
        return kernel;
    }
};

struct Basis {
    uint32_t core_width;
    std::vector<Mask> parities;
    // Columns map assignments in the new coordinates back to the collected basis.
    std::vector<Mask> columns;
};

inline std::optional<Basis> reduce_core(const PhaseSignature& signature, uint32_t width,
                                        Budget& budget) {
    const auto kernel = signature.clifford_kernel(width, budget);
    const auto core_width = width - static_cast<uint32_t>(kernel.size());
    if (core_width > 64)
        return std::nullopt;
    budget.charge(width * width);
    std::vector<Mask> generators;
    for (uint32_t j = 0; j < width; ++j) {
        generators.push_back(bit(j));
    }
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
        }
        for (uint32_t i = 0; i < width; ++i) {
            if (i == j || !matrix[i].bit_get(j)) {
                continue;
            }
            matrix[i] ^= matrix[j];
            generators[i] ^= generators[j];
        }
    }
    return Basis{core_width, std::move(generators), std::move(columns)};
}

inline Polynomial synthesize(const Polynomial& terms, const Basis& basis, Budget& budget) {
    PhaseSignature remainder;
    const uint64_t core_mask =
        basis.core_width == 64 ? ~uint64_t{0} : (uint64_t{1} << basis.core_width) - 1;
    std::map<uint64_t, uint8_t> projected;
    for (const auto& [parity, coefficient] : terms) {
        Mask converted;
        budget.charge(basis.columns.size());
        for (uint32_t j = 0; j < basis.columns.size(); ++j) {
            converted.bit_set(j, (parity & basis.columns[j]).popcount() & 1);
        }
        remainder.add(converted, coefficient, budget);
        const uint64_t mask = converted.w[0] & core_mask;
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
    for (auto parity : phase_detail::tohpe(std::move(columns), basis.core_width)) {
        budget.add(out, Mask{parity}, 1);
        remainder.add(Mask{parity}, -1, budget);
    }
    // Subtracting the synthesized columns leaves only S, Z and CZ phases,
    // including couplings to the discarded Clifford directions.
    for (uint32_t i = 0; i < basis.columns.size(); ++i) {
        const auto coefficient = remainder.linear[i];
        if (coefficient & 1)
            throw std::logic_error("wide synthesis left a non-Clifford rotation");
        if (coefficient)
            budget.add(out, bit(i), coefficient);
    }
    for (const auto& [indices, pair] : remainder.pairs) {
        if (pair.weight != 2 || !pair.odd_columns.is_zero()) {
            throw std::logic_error("wide synthesis changed the non-Clifford signature");
        }
        const auto [i, j] = indices;
        budget.add(out, bit(i), 2);
        budget.add(out, bit(j), 2);
        budget.add(out, bit(i) | bit(j), -2);
    }
    return out;
}

struct PauliDerivative {
    Mask parity;
    uint8_t constant = 0;
};

// Returns p(x xor flip) - p(x) = constant + 4 * parity(x), or no value when
// conjugation of a Pauli with this flip is no longer a Pauli.
inline std::optional<PauliDerivative> pauli_derivative(const Polynomial& p, Mask flip,
                                                       Budget& budget) {
    if (flip.is_zero()) {
        return PauliDerivative{{}, 0};
    }
    budget.charge(p.size());
    int constant = 0;
    std::array<uint8_t, Mask::kBits> linear{};
    std::array<Mask, Mask::kBits> quadratic{};
    // Flipping an odd number of bits in a parity changes p to 1-p.
    // Multiplication by -2 leaves only linear and quadratic terms modulo eight.
    for (const auto& [parity, coefficient] : p) {
        if (!((parity & flip).popcount() & 1))
            continue;
        constant = (constant + coefficient) & 7;
        budget.charge(parity.popcount());
        for (auto bits = parity; !bits.is_zero();) {
            const uint32_t i = bits.lowest_bit();
            bits.clear_lowest_bit();
            auto& weight = linear.at(i);
            weight = (weight - 2 * coefficient) & 7;
            if (coefficient & 1)
                quadratic.at(i) ^= bits;
        }
    }
    if ((constant & 1) != 0) {
        return std::nullopt;
    }
    for (const Mask& row : quadratic) {
        if (!row.is_zero()) {
            return std::nullopt;
        }
    }
    Mask z;
    for (uint32_t i = 0; i < linear.size(); ++i) {
        if (linear[i] == 4) {
            z.bit_set(i, true);
        } else if (linear[i] != 0) {
            return std::nullopt;
        }
    }
    return PauliDerivative{z, static_cast<uint8_t>(constant)};
}
}  // namespace clifft::phase_detail::wide
