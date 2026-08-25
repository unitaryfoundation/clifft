#include "clifft/sampling/kernel_dispatch.h"

#include "clifft/sampling/indexing.h"
#include "clifft/util/intra_shot_parallel.h"

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace clifft::sampling {

namespace {

constexpr size_t kLanes = 2;
constexpr size_t kDimension = 4;
constexpr size_t kMatrixSize = kDimension * kDimension;

struct alignas(16) LaneWeights {
    std::array<double, kLanes> real{};
    std::array<double, kLanes> imag{};
};

struct FusedRotationNeonSidecar {
    std::array<bool, kDimension> swap_lanes{};
    std::vector<LaneWeights> weights;
};

float64x2_t maybe_swap_lanes(float64x2_t value, bool swap) noexcept {
    return swap ? vextq_f64(value, value, 1) : value;
}

void apply_fused_rotation_neon_range(State& state, const PreparedFusedRotation& rotation,
                                     const FusedRotationNeonSidecar& sidecar,
                                     uint64_t vector_begin, uint64_t vector_end) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();

    // A pivot above bit zero lets each vector lane carry an independent orbit;
    // this avoids horizontal reductions in the dense four-by-four multiply.
    for (uint64_t vector_index = vector_begin; vector_index < vector_end; ++vector_index) {
        const uint64_t packed = vector_index * kLanes;
        uint64_t representative = insert_zero_bit(packed, rotation.orbit_pivots[0]);
        representative = insert_zero_bit(representative, rotation.orbit_pivots[1]);
        const LaneWeights* const matrix =
            sidecar.weights.data() +
            selector_index(representative, rotation.selector_masks) * kMatrixSize;

        std::array<float64x2_t, kDimension> input_real;
        std::array<float64x2_t, kDimension> input_imag;
        for (size_t column = 0; column < kDimension; ++column) {
            uint64_t index = representative;
            if ((column & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((column & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            input_real[column] = maybe_swap_lanes(vld1q_f64(real + physical_base),
                                                  sidecar.swap_lanes[column]);
            input_imag[column] = maybe_swap_lanes(vld1q_f64(imag + physical_base),
                                                  sidecar.swap_lanes[column]);
        }

        for (size_t row = 0; row < kDimension; ++row) {
            float64x2_t output_real = vdupq_n_f64(0.0);
            float64x2_t output_imag = vdupq_n_f64(0.0);
            for (size_t column = 0; column < kDimension; ++column) {
                const LaneWeights& weight = matrix[row * kDimension + column];
                const float64x2_t weight_real = vld1q_f64(weight.real.data());
                const float64x2_t weight_imag = vld1q_f64(weight.imag.data());
                output_real = vfmaq_f64(output_real, weight_real, input_real[column]);
                output_real = vfmsq_f64(output_real, weight_imag, input_imag[column]);
                output_imag = vfmaq_f64(output_imag, weight_real, input_imag[column]);
                output_imag = vfmaq_f64(output_imag, weight_imag, input_real[column]);
            }

            uint64_t index = representative;
            if ((row & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((row & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            vst1q_f64(real + physical_base,
                      maybe_swap_lanes(output_real, sidecar.swap_lanes[row]));
            vst1q_f64(imag + physical_base,
                      maybe_swap_lanes(output_imag, sidecar.swap_lanes[row]));
        }
    }
}

void apply_fused_rotation_neon(State& state, const PreparedFusedRotation& rotation,
                               const void* opaque_sidecar) noexcept {
    const auto& sidecar = *static_cast<const FusedRotationNeonSidecar*>(opaque_sidecar);
    assert(rotation.orbit_rank == 2 && rotation.orbit_pivots[0] >= 1 &&
           rotation.orbit_pivots[0] < rotation.orbit_pivots[1] &&
           "NEON fused rotation requires ordered pivots above the lane bit");
    assert(state.active_width() == rotation.active_width &&
           "fused rotation width must match the active state");

    apply_fused_rotation_neon_range(state, rotation, sidecar, 0,
                                    state.size() / (kDimension * kLanes));
}

void apply_fused_rotation_neon_parallel(State& state, const PreparedFusedRotation& rotation,
                                        const void* opaque_sidecar, uint32_t workers,
                                        uint32_t min_active_width) noexcept {
    if (!should_parallelize_intra_shot(state.active_width(), workers, min_active_width)) {
        apply_fused_rotation_neon(state, rotation, opaque_sidecar);
        return;
    }
    const auto& sidecar = *static_cast<const FusedRotationNeonSidecar*>(opaque_sidecar);
    assert(rotation.orbit_rank == 2 && rotation.orbit_pivots[0] >= 1 &&
           rotation.orbit_pivots[0] < rotation.orbit_pivots[1] &&
           "NEON fused rotation requires ordered pivots above the lane bit");
    assert(state.active_width() == rotation.active_width &&
           "fused rotation width must match the active state");

    const uint64_t vector_count = state.size() / (kDimension * kLanes);
    intra_shot_parallel_ranges(vector_count, workers, [&](uint64_t begin, uint64_t end) noexcept {
        apply_fused_rotation_neon_range(state, rotation, sidecar, begin, end);
    });
}

}  // namespace

FusedRotationSidecar prepare_fused_rotation_neon_sidecar(
    const PreparedFusedRotation& rotation) {
    if (rotation.orbit_rank != 2 || rotation.orbit_pivots[0] < 1 ||
        rotation.orbit_pivots[0] >= rotation.orbit_pivots[1]) {
        return {};
    }

    auto sidecar = std::make_shared<FusedRotationNeonSidecar>();
    for (size_t member = 0; member < kDimension; ++member) {
        uint64_t mask = 0;
        if ((member & 1U) != 0) {
            mask ^= rotation.orbit_masks[0];
        }
        if ((member & 2U) != 0) {
            mask ^= rotation.orbit_masks[1];
        }
        sidecar->swap_lanes[member] = (mask & 1U) != 0;
    }

    const size_t num_variants = size_t{1} << rotation.selector_masks.size();
    assert(rotation.matrices.size() == num_variants * kMatrixSize &&
           "fused rotation matrix table must cover every selector value");
    sidecar->weights.resize(num_variants * kMatrixSize);
    // Neighboring representatives can select different matrices. Expanding
    // those weights here keeps selector-dependent gathers out of hot execution.
    for (size_t base_selector = 0; base_selector < num_variants; ++base_selector) {
        for (size_t lane = 0; lane < kLanes; ++lane) {
            const size_t selector = base_selector ^ selector_index(lane, rotation.selector_masks);
            for (size_t element = 0; element < kMatrixSize; ++element) {
                const std::complex<double> weight =
                    rotation.matrices[selector * kMatrixSize + element];
                LaneWeights& expanded = sidecar->weights[base_selector * kMatrixSize + element];
                expanded.real[lane] = weight.real();
                expanded.imag[lane] = weight.imag();
            }
        }
    }

    return FusedRotationSidecar{std::move(sidecar), apply_fused_rotation_neon,
                                apply_fused_rotation_neon_parallel};
}

}  // namespace clifft::sampling
