#pragma once

#include "clifft/frontend/hir.h"

namespace clifft {

/// Symplectic inner product over BitMask masks.
/// Returns true if the two Pauli strings anti-commute.
inline bool anti_commute(const PauliBitMask& x1, const PauliBitMask& z1, const PauliBitMask& x2,
                         const PauliBitMask& z2) {
    return (((x1 & z2) ^ (z1 & x2)).popcount() & 1) != 0;
}

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir);

}  // namespace clifft
