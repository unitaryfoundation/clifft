#pragma once

#include "clifft/frontend/hir.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

/// Normalize a T gate to sign=false, folding the sign bit into global_weight.
void normalize_t_sign(HirModule& hir, HeisenbergOp& op);

/// Returns true when op_i cannot commute past op_j during a forward T scan.
bool blocks_scan_path(const HeisenbergOp& op_i, const HeisenbergOp& op_j, const HirModule& hir);

struct SameAxisTStats {
    size_t merges = 0;
    size_t t_removed = 0;
};

/// Fuse or cancel same-axis T/T_dag pairs within [range_start, range_end).
/// range_end is updated when ops are deleted.
void fuse_same_axis_t_in_range(HirModule& hir, size_t range_start, size_t& range_end,
                               SameAxisTStats& stats);

/// Erase ops marked in deleted and optionally compact source_map.
void compact_deleted_ops(HirModule& hir, const std::vector<uint8_t>& deleted);

}  // namespace clifft
