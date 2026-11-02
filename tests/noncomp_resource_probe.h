#pragma once

#include "clifft/sampling/branch_choice.h"

#include <cstddef>

namespace clifft::test {

struct HotPathCounters {
    size_t depth = 0, intervals = 0, allocations = 0;
};
inline thread_local HotPathCounters hot_path_counters;

}  // namespace clifft::test
