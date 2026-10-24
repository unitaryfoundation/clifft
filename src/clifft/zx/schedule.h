#pragma once

// Ordering a group's PBC operations to keep the dense array small.
//
// Two operations must keep their relative order exactly when they
// anticommute; every order that respects those pairs gives the same
// distribution. The active width is the number of lines minus the dimension of
// the Pauli group that stabilizes every trajectory up to sign. The greedy
// order is Tsim's (tsim/statevector/schedule.py): measurements run as early
// as possible and widening rotations as late as possible; when something must
// widen, it runs the pending measurement whose unrun ancestors reach the
// smallest width, the earliest on a tie. Clifft's ActiveWidthSchedulePass can
// refine the result afterwards, but it cannot find this global order itself.

#include "clifft/zx/pbc.h"

#include <cstdint>
#include <vector>

namespace clifft::zx {

struct Schedule {
    std::vector<uint32_t> order;  // indices into the group's ops
    uint32_t width = 0;           // the largest active width along the order
};

[[nodiscard]] Schedule greedy_schedule(const PbcGroup& group);

// The largest active width along `order`.
[[nodiscard]] uint32_t peak_width(const PbcGroup& group, const std::vector<uint32_t>& order);

}  // namespace clifft::zx
