#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/symplectic.h"

#include <cassert>
#include <cstdint>

namespace clifft {

/// Returns true if the two HIR operations can be safely swapped in the
/// ops vector without changing program semantics or PRNG trajectory.
bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir);

}  // namespace clifft
