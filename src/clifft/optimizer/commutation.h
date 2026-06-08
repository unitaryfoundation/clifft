#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"

#include <bit>

#include <cassert>
#include <cstdint>
#include <vector>

namespace clifft {

/// Symplectic inner product over runtime-width mask views. Returns true if
/// the two Pauli strings anti-commute. All four views must share num_words().
inline bool anti_commute(MaskView x1, MaskView z1, MaskView x2, MaskView z2) {
    assert(x1.num_words() == z1.num_words() && z1.num_words() == x2.num_words() &&
           x2.num_words() == z2.num_words());
    int parity = 0;
    for (uint32_t i = 0; i < x1.num_words(); ++i) {
        parity += std::popcount((x1.words[i] & z2.words[i]) ^ (z1.words[i] & x2.words[i]));
    }
    return (parity & 1) != 0;
}

/// Per-qubit Pauli product phase exponent mod 4 for masks (x1, z1) * (x2, z2).
inline int pauli_product_phase_mod4(MaskView x1, MaskView z1, MaskView x2, MaskView z2) {
    int phase = 0;
    for (uint32_t w = 0; w < x1.num_words(); ++w) {
        uint64_t X1 = x1.words[w];
        uint64_t Z1 = z1.words[w];
        uint64_t X2 = x2.words[w];
        uint64_t Z2 = z2.words[w];
        uint64_t mask_plus = (X1 & ~Z1 & X2 & Z2) | (X1 & Z1 & ~X2 & Z2) | (~X1 & Z1 & X2 & ~Z2);
        uint64_t mask_minus = (X1 & ~Z1 & ~X2 & Z2) | (X1 & Z1 & X2 & ~Z2) | (~X1 & Z1 & X2 & Z2);
        phase += std::popcount(mask_plus);
        phase -= std::popcount(mask_minus);
    }
    return ((phase % 4) + 4) % 4;
}

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir);

/// Absorb a virtual S gate on Pauli generator (x_v, z_v, sign_v) into all
/// downstream HIR operations starting at start_idx and into the final tableau.
/// is_dagger=false means S; is_dagger=true means S_dag.
/// deleted[k] skips ops that have already been removed from this pass's sweep.
void apply_virtual_s_downstream(HirModule& hir, size_t start_idx, MaskView x_v, MaskView z_v,
                                 bool sign_v, bool is_dagger,
                                 const std::vector<uint8_t>& deleted);

}  // namespace clifft
