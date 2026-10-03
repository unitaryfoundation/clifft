#pragma once

#include <cstdint>
#include <vector>

namespace clifft::phase_detail {

// Reduce distinct nonzero parity columns modulo a diagonal Clifford.
// core_width is the dimension after removing all Clifford directions, hence
// a lower bound on the number of columns. Larger tables retain their input.
std::vector<uint64_t> tohpe(std::vector<uint64_t> columns, uint32_t core_width);

}  // namespace clifft::phase_detail
