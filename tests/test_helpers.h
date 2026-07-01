#pragma once

// Shared test helpers for Clifft Catch2 tests.

#include "clifft/frontend/hir.h"
#include "clifft/util/bitmask.h"
#include "clifft/util/config.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/stim_mask.h"

#include "stim.h"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

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

// HirModule append_* test helpers.
//
// All current tests build single-word Paulis (n <= 64). Wider patterns
// can call HirModule::append_*(args, fill_lambda) directly.
inline HeisenbergOp& append_tgate(HirModule& hir, uint64_t destab, uint64_t stab, bool sign,
                                  bool dagger = false) {
    return hir.append_tgate(dagger, [&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

inline HeisenbergOp& append_measure(HirModule& hir, uint64_t destab, uint64_t stab, bool sign,
                                    MeasRecordIdx idx) {
    return hir.append_measure(idx, [&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

inline HeisenbergOp& append_conditional(HirModule& hir, uint64_t destab, uint64_t stab, bool sign,
                                        ControllingMeasIdx idx) {
    return hir.append_conditional(idx, [&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

inline HeisenbergOp& append_phase_rotation(HirModule& hir, uint64_t destab, uint64_t stab,
                                           bool sign, double alpha) {
    return hir.append_phase_rotation(alpha, [&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

inline HeisenbergOp& append_exp_val(HirModule& hir, uint64_t destab, uint64_t stab, bool sign,
                                    ExpValIdx idx) {
    return hir.append_exp_val(idx, [&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

inline PauliMaskHandle claim_noise_channel_mask(HirModule& hir, uint64_t destab, uint64_t stab) {
    auto h = hir.claim_empty_noise_channel_mask();
    auto slot = hir.noise_channel_masks.mut_at(h);
    slot.x().words[0] = destab;
    slot.z().words[0] = stab;
    return h;
}

inline void set_pauli(HirModule& hir, const HeisenbergOp& op, uint64_t destab, uint64_t stab,
                      bool sign) {
    auto slot = hir.mask_at(op);
    slot.x().words[0] = destab;
    slot.z().words[0] = stab;
    slot.set_sign(sign);
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

// Dense-matrix oracle helpers for canonical-phase tests. Row-major
// dim x dim complex matrices in little-endian basis order.
using DenseMatrix = std::vector<std::complex<double>>;

inline DenseMatrix dense_tableau_matrix(const stim::Tableau<kStimWidth>& tab) {
    auto flat = tab.to_flat_unitary_matrix(true);
    DenseMatrix m(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        m[i] = {flat[i].real(), flat[i].imag()};
    }
    return m;
}

inline DenseMatrix dense_matmul(const DenseMatrix& a, const DenseMatrix& b, uint64_t dim) {
    DenseMatrix r(dim * dim, {0.0, 0.0});
    for (uint64_t i = 0; i < dim; ++i) {
        for (uint64_t k = 0; k < dim; ++k) {
            for (uint64_t j = 0; j < dim; ++j) {
                r[i * dim + j] += a[i * dim + k] * b[k * dim + j];
            }
        }
    }
    return r;
}

// Dense matrix of the projector-form rotation Pi_+ + e^{i*alpha*pi} Pi_- on
// the signed Pauli (x, z, sign) over n qubits, little-endian basis order.
// The fused S/S_dag the peephole absorbs is alpha = 0.5 / 1.5.
inline DenseMatrix dense_axis_rotation(uint64_t x, uint64_t z, bool sign, double alpha, size_t n) {
    const uint64_t dim = uint64_t{1} << n;
    constexpr std::complex<double> kIPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const std::complex<double> eig{std::cos(alpha * std::numbers::pi),
                                   std::sin(alpha * std::numbers::pi)};
    const std::complex<double> a = (1.0 + eig) / 2.0;
    const std::complex<double> b = (1.0 - eig) / 2.0;

    DenseMatrix r(dim * dim, {0.0, 0.0});
    for (uint64_t c = 0; c < dim; ++c) {
        r[c * dim + c] += a;
        uint32_t phase_idx = (sign ? 2U : 0U) + static_cast<uint32_t>(std::popcount(x & z)) +
                             2U * (static_cast<uint32_t>(std::popcount(c & z)) & 1U);
        r[(c ^ x) * dim + c] += b * kIPow[phase_idx & 3U];
    }
    return r;
}

}  // namespace test

/// Compare a MaskView to a fixed-width BitMask<N>. The two may have
/// different word counts; trailing words of either side must be zero.
template <size_t N>
inline bool operator==(MaskView v, const BitMask<N>& m) {
    auto vm = view(m);
    uint32_t common = std::min(v.num_words(), vm.num_words());
    for (uint32_t i = 0; i < common; ++i) {
        if (v.words[i] != vm.words[i])
            return false;
    }
    for (uint32_t i = common; i < v.num_words(); ++i) {
        if (v.words[i] != 0)
            return false;
    }
    for (uint32_t i = common; i < vm.num_words(); ++i) {
        if (vm.words[i] != 0)
            return false;
    }
    return true;
}
template <size_t N>
inline bool operator==(const BitMask<N>& m, MaskView v) {
    return v == m;
}

/// Compare a runtime-width MaskView to a uint64_t (interpreted as the
/// lower 64 bits, with all higher words required to be zero).
inline bool operator==(MaskView v, uint64_t expected) {
    if (v.num_words() == 0)
        return expected == 0;
    if (v.words[0] != expected)
        return false;
    for (uint32_t i = 1; i < v.num_words(); ++i) {
        if (v.words[i] != 0)
            return false;
    }
    return true;
}
inline bool operator==(uint64_t expected, MaskView v) {
    return v == expected;
}

}  // namespace clifft
