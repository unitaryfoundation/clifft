#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"

#include <cassert>
#include <cstdint>

namespace clifft {

/// Symplectic inner product over fixed-width BitMask masks.
/// Returns true if the two Pauli strings anti-commute.
inline bool anti_commute(const PauliBitMask& x1, const PauliBitMask& z1, const PauliBitMask& x2,
                         const PauliBitMask& z2) {
    return (((x1 & z2) ^ (z1 & x2)).popcount() & 1) != 0;
}

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

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir);

}  // namespace clifft
