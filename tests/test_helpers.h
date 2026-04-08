#pragma once

// Shared test helpers for Clifft Catch2 tests.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>

namespace clifft {
namespace test {

constexpr double kInvSqrt2 = 1.0 / std::numbers::sqrt2;

// Bitmask helpers for readable Pauli construction.
// Usage: make_pauli(n, X(0) | X(1), Z(2) | Z(3))
inline uint64_t X(size_t qubit) {
    return 1ULL << qubit;
}
inline uint64_t Z(size_t qubit) {
    return 1ULL << qubit;
}

// Convert a Pauli string like "XYZ" to (destab_mask, stab_mask) pair.
// Qubit 0 is the rightmost character: "XYZ" means X on q2, Y on q1, Z on q0.
// Returns {destab, stab} where destab has X bits and stab has Z bits.
// Y = iXZ, so both bits are set for Y.
inline std::pair<uint64_t, uint64_t> pauli_masks(const std::string& pauli) {
    uint64_t destab = 0;
    uint64_t stab = 0;
    size_t n = pauli.size();
    for (size_t i = 0; i < n; ++i) {
        size_t qubit = n - 1 - i;
        char c = pauli[i];
        if (c == 'X') {
            destab |= (1ULL << qubit);
        } else if (c == 'Z') {
            stab |= (1ULL << qubit);
        } else if (c == 'Y') {
            destab |= (1ULL << qubit);
            stab |= (1ULL << qubit);
        }
    }
    return {destab, stab};
}

// Deterministic LCG for test-local RNG.
// Constants from Knuth's MMIX (same as PCG's default step).
inline uint64_t test_lcg(uint64_t& seed) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
}

// Check if two complex numbers are close (uses Catch2 CHECK_THAT).
constexpr double kDefaultTol = 1e-12;
inline void check_complex(std::complex<double> actual, std::complex<double> expected,
                          double tol = kDefaultTol) {
    CHECK_THAT(actual.real(), Catch::Matchers::WithinAbs(expected.real(), tol));
    CHECK_THAT(actual.imag(), Catch::Matchers::WithinAbs(expected.imag(), tol));
}

}  // namespace test
}  // namespace clifft
