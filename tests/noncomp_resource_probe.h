#pragma once

#include "clifft/sampling/branch_choice.h"

#include <cstddef>

namespace clifft::test {

struct HotPathCounters {
    size_t depth = 0, intervals = 0, allocations = 0;
};
inline thread_local HotPathCounters hot_path_counters;
using OnlineBoundaryProbe = void (*)(detail::ReplayOnlineBoundary,
                                     const noncomp_detail::OnlineState&);
inline thread_local OnlineBoundaryProbe online_boundary_probe = nullptr;

}  // namespace clifft::test
