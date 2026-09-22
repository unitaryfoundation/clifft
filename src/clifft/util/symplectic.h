#pragma once

// Symplectic inner product over runtime-width mask views, needed by both
// optimizer/ (commutation analysis) and sampling/ (the planner's
// logical-noise-crossing sign correction) without either depending on the
// other.

#include "clifft/util/mask_view.h"

#include <bit>
#include <cassert>
#include <cstdint>

namespace clifft {

// Returns true if the two Pauli strings anti-commute. All four views must
// share num_words().
inline bool anti_commute(MaskView x1, MaskView z1, MaskView x2, MaskView z2) {
    assert(x1.num_words() == z1.num_words() && z1.num_words() == x2.num_words() &&
           x2.num_words() == z2.num_words());
    int parity = 0;
    for (uint32_t i = 0; i < x1.num_words(); ++i) {
        parity += std::popcount((x1.words[i] & z2.words[i]) ^ (z1.words[i] & x2.words[i]));
    }
    return (parity & 1) != 0;
}

}  // namespace clifft
