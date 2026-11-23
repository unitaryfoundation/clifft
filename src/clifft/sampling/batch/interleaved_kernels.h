#pragma once

#include "clifft/sampling/batch/interleaved_state.h"
#include "clifft/sampling/fused_rotation.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/pauli_preparation.h"

#include <cstdint>
#include <span>

namespace clifft::sampling {

// Coefficient kernels for InterleavedBatchState. Each kernel loops over basis
// states and applies the same prepared action to every live shot lane, so the
// inner loop vectorizes across contiguous lanes.
//
// The loops are portable C++, and their speed follows the vector width they
// are compiled for. A table holds the entry points of one such build:
// the portable build uses the library's baseline flags, and x86 builds with
// runtime dispatch compile the same bodies again for AVX2 and AVX-512.
struct InterleavedKernels {
    // Materializing signed sines lets arithmetic kernels vectorize across
    // contiguous shot coefficients. The output span selects active lanes after
    // compaction; packed signs can retain the executor's full capacity.
    void (*prepare_rotation_sines)(std::span<double> output, double sine,
                                   std::span<const uint64_t> signs) noexcept;

    void (*apply_rotation)(InterleavedBatchState& state, const PreparedRotation& rotation,
                           std::span<const double> signed_sines) noexcept;

    void (*apply_promotion)(InterleavedBatchState& state, const PreparedPromotion& promotion,
                            std::span<const double> signed_sines) noexcept;

    void (*apply_fused_rotation)(InterleavedBatchState& state,
                                 const PreparedFusedRotation& rotation) noexcept;
    void (*apply_dynamic_fused_rotation)(InterleavedBatchState& state,
                                         std::span<const PreparedFusedRotation* const> variants,
                                         std::span<const uint8_t> lane_variants) noexcept;

    void (*measurement_probabilities)(const InterleavedBatchState& state,
                                      const PreparedMeasurement& measurement,
                                      std::span<double> probability_zero,
                                      std::span<double> probability_one) noexcept;
    // Each lane's collapsed state is scaled by branch_scales[lane], the
    // inverse square root of its selected branch probability. The caller
    // computes it once per lane, which keeps the coefficient loops free of
    // library calls.
    void (*collapse_measurement)(InterleavedBatchState& state,
                                 const PreparedMeasurement& measurement,
                                 std::span<const uint8_t> branches,
                                 std::span<const double> branch_scales) noexcept;
    void (*expectation_values)(const InterleavedBatchState& state, const PreparedPauli& pauli,
                               std::span<double> output) noexcept;
};

// Kernels for a plan's executor backend. The AVX2 and AVX-512 backends return
// their own builds where runtime dispatch is compiled; every other backend
// returns the portable build. A caller reads the table once and then makes one
// indirect call per prepared action.
[[nodiscard]] const InterleavedKernels& interleaved_kernels(ExecutorBackend backend) noexcept;

}  // namespace clifft::sampling
