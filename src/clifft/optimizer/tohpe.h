#pragma once

#include <cstdint>
#include <vector>

namespace clifft::phase_detail {

// TOHPE from Vandaele, "Lower T-count with faster algorithms", Algorithm 2:
// https://arxiv.org/abs/2407.08695
// Author's Rust implementation at the revision used for validation:
// https://github.com/VivienVandaele/quantum-circuit-optimization/blob/231e6fe9f92d5bb1ebf7459c2a9233f5e74d148e/src/t_opt.rs#L79
// Clifft supplies the state constraints, core extraction and Clifford correction.
//
// Reduce distinct nonzero parity columns modulo a diagonal Clifford.
// core_width is the dimension after removing all Clifford directions, hence
// a lower bound on the number of columns. Larger tables retain their input.
std::vector<uint64_t> tohpe(std::vector<uint64_t> columns, uint32_t core_width);

}  // namespace clifft::phase_detail
