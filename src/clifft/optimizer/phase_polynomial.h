#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

// Phase algebra for separating commuting T rotations into a smaller
// non-Clifford core and a diagonal Clifford correction.
namespace clifft::phase_detail {

// A weighted Boolean polynomial modulo eight: linear coefficients are arbitrary,
// quadratic coefficients are even, and cubic coefficients are multiples of four.
// Maps monomial variable masks to coefficients: {0b0101, 4} means 4*x0*x2.
// Zero coefficients are omitted.
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
// Project onto the core, reduce odd parity terms with TOHPE, and restore the
// exact Clifford correction. The result never has more T terms than the input.
// In terms and the result, masks select XOR parities rather than monomials.
Polynomial synthesize_parities(const Polynomial& terms, const Polynomial& reduced,
                               const CoreBasis& basis);

struct PauliDerivative {
    uint64_t parity;
    uint8_t constant;
};

// Returns p(x xor flip) - p(x) = constant + 4 * parity(x), or no value when
// conjugation of a Pauli with this flip is no longer a Pauli.
std::optional<PauliDerivative> pauli_derivative(const Polynomial& polynomial, uint64_t flip);

}  // namespace clifft::phase_detail
