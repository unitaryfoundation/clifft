#pragma once

#include "clifft/sampling/fused_rotation.h"
#include "clifft/sampling/interleaved_batch_state.h"
#include "clifft/sampling/pauli_preparation.h"

#include <cstdint>
#include <span>

namespace clifft::sampling {

// Fills one signed sine per live lane. Keeping this materialization explicit
// lets the arithmetic kernels vectorize across contiguous shot coefficients.
void prepare_interleaved_rotation_sines(std::span<double> output, double sine,
                                        std::span<const uint8_t> signs) noexcept;

void apply_interleaved_rotation(InterleavedBatchState& state,
                                const PreparedRotation& rotation,
                                std::span<const double> signed_sines) noexcept;

void apply_interleaved_promotion(InterleavedBatchState& state,
                                 const PreparedPromotion& promotion,
                                 std::span<const double> signed_sines) noexcept;

void apply_interleaved_fused_rotation(InterleavedBatchState& state,
                                      const PreparedFusedRotation& rotation) noexcept;
void apply_interleaved_dynamic_fused_rotation(
    InterleavedBatchState& state, std::span<const PreparedFusedRotation> variants,
    std::span<const uint8_t> lane_variants) noexcept;

}  // namespace clifft::sampling
