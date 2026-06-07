#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/util/mask_view.h"

#include <cstddef>
#include <vector>

namespace clifft {

/// Absorb a virtual S gate on Pauli generator (x_v, z_v) into downstream HIR ops
/// and the final tableau.
void apply_virtual_s_downstream(HirModule& hir, size_t start_idx, MaskView x_v, MaskView z_v,
                                bool sign_v, bool is_dagger, const std::vector<uint8_t>& deleted);

}  // namespace clifft
