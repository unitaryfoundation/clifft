#pragma once

// Shared test helpers for UCC Catch2 tests.

#include <cstddef>
#include <cstdint>

namespace ucc {
namespace test {

// Bitmask helpers for readable Pauli construction.
// Usage: make_pauli(n, X(0) | X(1), Z(2) | Z(3))
inline uint64_t X(size_t qubit) {
    return 1ULL << qubit;
}
inline uint64_t Z(size_t qubit) {
    return 1ULL << qubit;
}
inline uint64_t Y(size_t qubit) {
    return 1ULL << qubit;
}  // use for both X and Z masks

}  // namespace test
}  // namespace ucc
