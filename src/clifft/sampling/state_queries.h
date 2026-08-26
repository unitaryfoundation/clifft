#pragma once

#include "clifft/sampling/executable_plan.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

class PhaseAwareCliffordFrame;

namespace sampling {

// Exact final-state queries execute one prepared trajectory, then reconstruct
// physical-basis results through the plan's final coordinate map.
[[nodiscard]] std::vector<double> basis_probabilities(const ExecutablePlan& plan,
                                                      std::span<const uint64_t> basis_masks,
                                                      size_t num_basis_masks,
                                                      size_t words_per_basis_mask);

namespace internal {

// Phase-aware compilation calibrates the exact Clifford representative once,
// then supplies the resulting scalar to the selected-basis amplitude walk.
[[nodiscard]] std::complex<double> clifford_row_phase(const Tableau& final_tableau,
                                                      const PhaseAwareCliffordFrame& exact_frame,
                                                      std::span<const uint64_t> physical_basis);

[[nodiscard]] std::vector<std::complex<double>> basis_amplitudes(
    const ExecutablePlan& plan, std::complex<double> phase, std::span<const uint64_t> basis_masks,
    size_t num_basis_masks, size_t words_per_basis_mask);

}  // namespace internal

// The returned vector is normalized and represents the final state ray. Its
// global phase is unspecified.
[[nodiscard]] std::vector<std::complex<double>> get_statevector(const ExecutablePlan& plan);

}  // namespace sampling
}  // namespace clifft
