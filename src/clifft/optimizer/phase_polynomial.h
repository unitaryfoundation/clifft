#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace clifft::phase_detail {

// A weighted Boolean polynomial modulo eight: linear coefficients are arbitrary,
// quadratic coefficients are even, and cubic coefficients are multiples of four.
// Monomials are variable masks; zero coefficients are omitted.
using Polynomial = std::map<uint64_t, uint8_t>;

void add_parity(Polynomial& polynomial, uint64_t parity, int coefficient);

// Translations whose finite differences have an even constant and only linear
// coefficients divisible by four. These directions carry no non-Clifford work.
std::vector<uint64_t> clifford_kernel(const Polynomial& polynomial, uint32_t width);

struct CoreBasis {
    uint32_t core_width;
    // New eigenbit parities in the original coordinates, including the core
    // complement first. The polynomial is rewritten in these new coordinates.
    std::vector<uint64_t> parities;
    // Original eigenbit assignments for retained coordinates with kernel bits zero.
    std::vector<uint64_t> core_columns;
};

CoreBasis reduce_core(Polynomial& polynomial, uint32_t width);
Polynomial synthesize_parities(const Polynomial& polynomial);

// Preserve the input parity list on the retained coordinates and account for
// the removed dependence with an exact diagonal Clifford correction.
Polynomial project_parities(const Polynomial& terms, const Polynomial& reduced,
                            const CoreBasis& basis);

struct PauliDerivative {
    uint64_t parity;
    uint8_t constant;
};

// Returns p(x xor flip) - p(x) = constant + 4 * parity(x), or no value when
// conjugation of an observer with this flip is not a Pauli.
std::optional<PauliDerivative> pauli_derivative(const Polynomial& polynomial, uint64_t flip);

}  // namespace clifft::phase_detail
