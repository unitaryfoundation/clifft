#pragma once

#include "clifft/frontend/hir.h"

#include <cstddef>

namespace clifft {

struct McrTcountStats {
    size_t window_scans = 0;
    size_t window_scans_over_lookahead_cap = 0;
    size_t quadruples_found = 0;
    size_t swaps_applied = 0;
    size_t merges = 0;
    size_t t_removed = 0;
};

/// Bounded MCR reordering on contiguous T-gate windows. Keeps a rewrite only
/// when same-axis fusion inside the window reduces T count.
void run_mcr_tcount(HirModule& hir, McrTcountStats& stats);

}  // namespace clifft
