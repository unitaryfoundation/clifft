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
};

CoreBasis reduce_core(Polynomial& polynomial, uint32_t width);
Polynomial synthesize_parities(const Polynomial& polynomial);

struct PauliDerivative {
    uint64_t parity;
    uint8_t constant;
};

// Returns p(x xor flip) - p(x) = constant + 4 * parity(x), or no value when
// conjugation of an observer with this flip is not a Pauli.
std::optional<PauliDerivative> pauli_derivative(const Polynomial& polynomial, uint64_t flip);

}  // namespace clifft::phase_detail
