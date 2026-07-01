#pragma once

// Canonical tableau phase tracking.
//
// A stim::Tableau fixes its Clifford only up to global phase;
// to_flat_unitary_matrix() resolves the ambiguity by scaling the row-major
// flat matrix so its first nonzero entry is real positive. Any rewrite
// that replaces a physical unitary factor with a tableau update therefore
// shifts the API-visible global phase, and the compiler must fold the
// difference into global_weight to keep get_statevector() exact.
//
// The flat matrix of a tableau is the Choi state of the Clifford: a
// 2n-qubit stabilizer state whose flat index packs (row << n) | col with
// qubit q <-> bit q, stabilized by X_k (x) U(X_k) and Z_k (x) U(Z_k) on
// the (input, output) halves. The first nonzero flat-matrix entry sits at
// the minimum-integer support index of that state, and entries relative
// to it follow by walking support generators, so phase deltas between
// canonical forms are computable in polynomial time with no dense
// matrices.

#include "clifft/util/stim_mask.h"

#include <complex>
#include <cstdint>
#include <vector>

namespace clifft::internal {

/// Flat Choi-state index, 2n bits packed little-endian into 64-bit words.
using ChoiIndex = std::vector<uint64_t>;

/// Row-reduced description of a tableau's Choi state: support generators
/// in descending-pivot order plus the minimum-integer support index, which
/// stim's matrix canonicalization anchors at a positive real amplitude.
struct ChoiSupport {
    std::vector<stim::PauliString<kStimWidth>> x_rows;
    std::vector<uint32_t> pivots;
    std::vector<ChoiIndex> x_masks;
    ChoiIndex anchor;
};

/// Gauss-Jordan reduce the Choi stabilizer group of `tab` and locate the
/// minimum-integer support index of its flat unitary matrix.
[[nodiscard]] ChoiSupport build_choi_support(const stim::Tableau<kStimWidth>& tab);

/// Amplitude of |index> in the Choi state, in units where the anchor
/// (stim's positive-real first nonzero matrix entry) has amplitude 1.
/// Returns 0 when the index lies outside the support.
[[nodiscard]] std::complex<double> choi_amplitude(const ChoiSupport& s, const ChoiIndex& index);

inline constexpr std::complex<double> kImagPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

[[nodiscard]] inline bool choi_index_bit(const ChoiIndex& idx, uint32_t bit) {
    return ((idx[bit / 64] >> (bit % 64)) & 1U) != 0;
}

inline void choi_index_flip_bit(ChoiIndex& idx, uint32_t bit) {
    idx[bit / 64] ^= uint64_t{1} << (bit % 64);
}

}  // namespace clifft::internal
