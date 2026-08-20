#pragma once

#include "clifft/sampling/executable_plan.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

// Exact final-state queries execute one prepared trajectory, then reconstruct
// physical-basis results through the plan's final coordinate map.
[[nodiscard]] std::vector<double> basis_probabilities(const ExecutablePlan& plan,
                                                      std::span<const uint64_t> basis_masks,
                                                      size_t num_basis_masks,
                                                      size_t words_per_basis_mask);

// The returned vector is normalized and represents the final state ray. Its
// global phase is unspecified.
[[nodiscard]] std::vector<std::complex<double>> get_statevector(const ExecutablePlan& plan);

}  // namespace clifft::sampling
