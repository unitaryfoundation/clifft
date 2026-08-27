#include "clifft/sampling/indexing.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/util/intra_shot_parallel.h"

#include <arm_neon.h>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
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
    bool pair_vectors = false;
    std::vector<LaneWeights> weights;
};

float64x2_t maybe_swap_lanes(float64x2_t value, bool swap) noexcept {
    return swap ? vextq_f64(value, value, 1) : value;
}

float64x2_t signed_sine_lanes(uint64_t basis, uint64_t z, double sine) noexcept {
    const bool high_parity = (std::popcount(basis & z) & 1U) != 0;
    const double first = high_parity ? -sine : sine;
    const double second = (z & 1U) != 0 ? -first : first;
    return vsetq_lane_f64(second, vdupq_n_f64(first), 1);
}

uint64_t inclusive_prefix_parities(uint64_t value) noexcept {
    value ^= value << 1;
    value ^= value << 2;
    value ^= value << 4;
    value ^= value << 8;
    value ^= value << 16;
    value ^= value << 32;
    return value;
}

void apply_diagonal_rotation_neon_range(State& state, const PreparedRotation& rotation, double sine,
                                        uint64_t vector_begin, uint64_t vector_end) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const float64x2_t cosine = vdupq_n_f64(rotation.cosine);
#pragma clang loop unroll_count(2)
    for (uint64_t vector_index = vector_begin; vector_index < vector_end; ++vector_index) {
        const uint64_t basis = vector_index * kLanes;
        const float64x2_t input_real = vld1q_f64(real + basis);
        const float64x2_t input_imag = vld1q_f64(imag + basis);
        const float64x2_t signed_sine = signed_sine_lanes(basis, rotation.pauli.z, sine);
        const float64x2_t output_real =
            vfmaq_f64(vmulq_f64(cosine, input_real), signed_sine, input_imag);
        const float64x2_t output_imag =
            vfmsq_f64(vmulq_f64(cosine, input_imag), signed_sine, input_real);
        vst1q_f64(real + basis, output_real);
        vst1q_f64(imag + basis, output_imag);
    }
}

template <bool RealPhase>
void apply_lane_paired_rotation_neon_range(State& state, const PreparedRotation& rotation,
                                           double sine, uint64_t vector_begin,
                                           uint64_t vector_end) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const float64x2_t cosine = vdupq_n_f64(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();

#pragma clang loop unroll_count(2)
    for (uint64_t vector_index = vector_begin; vector_index < vector_end; ++vector_index) {
        const uint64_t basis = vector_index * kLanes;
        const float64x2_t input_real = vld1q_f64(real + basis);
        const float64x2_t input_imag = vld1q_f64(imag + basis);
        const float64x2_t partner_real = vextq_f64(input_real, input_real, 1);
        const float64x2_t partner_imag = vextq_f64(input_imag, input_imag, 1);
        const float64x2_t basis_sine =
            signed_sine_lanes(basis, rotation.pauli.z, sine * base_phase);
        const float64x2_t partner_sine = RealPhase ? basis_sine : vnegq_f64(basis_sine);

        float64x2_t output_real;
        float64x2_t output_imag;
        if constexpr (RealPhase) {
            output_real = vfmaq_f64(vmulq_f64(cosine, input_real), partner_sine, partner_imag);
            output_imag = vfmsq_f64(vmulq_f64(cosine, input_imag), partner_sine, partner_real);
        } else {
            output_real = vfmaq_f64(vmulq_f64(cosine, input_real), partner_sine, partner_real);
            output_imag = vfmaq_f64(vmulq_f64(cosine, input_imag), partner_sine, partner_imag);
        }
        vst1q_f64(real + basis, output_real);
        vst1q_f64(imag + basis, output_imag);
    }
}

template <bool RealPhase>
void apply_high_pivot_rotation_neon_range(State& state, const PreparedRotation& rotation,
                                          double sine, uint64_t vector_begin,
                                          uint64_t vector_end) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pair_stride = rotation.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lower_mask = pair_stride - 1;
    const bool swap = (rotation.pauli.x & 1U) != 0;
    const float64x2_t cosine = vdupq_n_f64(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();
    const double even_left_sine = sine * base_phase;
    // Compress away the lane and pairing bits so adjacent vector indices can
    // update phase parity without a popcount in every hot-loop iteration.
    const uint64_t compressed_z = ((rotation.pauli.z & (pair_stride - 1) & ~uint64_t{1}) >> 1) |
                                  ((rotation.pauli.z & ~(pair_period - 1)) >> 2);
    const uint64_t parity_transitions = inclusive_prefix_parities(compressed_z);
    float64x2_t left_sine =
        vsetq_lane_f64((rotation.pauli.z & 1U) != 0 ? -even_left_sine : even_left_sine,
                       vdupq_n_f64(even_left_sine), 1);
    if ((std::popcount(vector_begin & compressed_z) & 1U) != 0) {
        left_sine = vnegq_f64(left_sine);
    }

#pragma clang loop unroll_count(2)
    for (uint64_t vector_index = vector_begin; vector_index < vector_end; ++vector_index) {
        const uint64_t packed = vector_index * kLanes;
        const uint64_t left = (packed & lower_mask) | ((packed & ~lower_mask) << 1);
        const uint64_t right_base = (left ^ rotation.pauli.x) & ~(uint64_t{kLanes - 1});
        const float64x2_t left_real = vld1q_f64(real + left);
        const float64x2_t left_imag = vld1q_f64(imag + left);
        const float64x2_t right_real = maybe_swap_lanes(vld1q_f64(real + right_base), swap);
        const float64x2_t right_imag = maybe_swap_lanes(vld1q_f64(imag + right_base), swap);
        float64x2_t output_left_real;
        float64x2_t output_left_imag;
        float64x2_t output_right_real;
        float64x2_t output_right_imag;
        if constexpr (RealPhase) {
            output_left_real = vfmaq_f64(vmulq_f64(cosine, left_real), left_sine, right_imag);
            output_left_imag = vfmsq_f64(vmulq_f64(cosine, left_imag), left_sine, right_real);
            output_right_real = vfmaq_f64(vmulq_f64(cosine, right_real), left_sine, left_imag);
            output_right_imag = vfmsq_f64(vmulq_f64(cosine, right_imag), left_sine, left_real);
        } else {
            output_left_real = vfmsq_f64(vmulq_f64(cosine, left_real), left_sine, right_real);
            output_left_imag = vfmsq_f64(vmulq_f64(cosine, left_imag), left_sine, right_imag);
            output_right_real = vfmaq_f64(vmulq_f64(cosine, right_real), left_sine, left_real);
            output_right_imag = vfmaq_f64(vmulq_f64(cosine, right_imag), left_sine, left_imag);
        }

        vst1q_f64(real + left, output_left_real);
        vst1q_f64(imag + left, output_left_imag);
        vst1q_f64(real + right_base, maybe_swap_lanes(output_right_real, swap));
        vst1q_f64(imag + right_base, maybe_swap_lanes(output_right_imag, swap));
        const uint64_t next_vector = vector_index + 1;
        if (next_vector < vector_end &&
            ((parity_transitions >> std::countr_zero(next_vector)) & 1U) != 0) {
            left_sine = vnegq_f64(left_sine);
        }
    }
}

template <size_t Groups>
void apply_fused_rotation_neon_group(State& state, const PreparedFusedRotation& rotation,
                                     const FusedRotationNeonSidecar& sidecar,
                                     uint64_t vector_index) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    std::array<uint64_t, Groups> representatives{};
    for (size_t group = 0; group < Groups; ++group) {
        const uint64_t packed = (vector_index + group) * kLanes;
        uint64_t representative = insert_zero_bit(packed, rotation.orbit_pivots[0]);
        representative = insert_zero_bit(representative, rotation.orbit_pivots[1]);
        representatives[group] = representative;
    }
    const size_t selector = selector_index(representatives[0], rotation.selector_masks);
    const LaneWeights* const matrix = sidecar.weights.data() + selector * kMatrixSize;
    if constexpr (Groups > 1) {
        assert(selector_index(representatives[1], rotation.selector_masks) == selector &&
               "paired NEON orbit groups must share a matrix selector");
    }

    // A pivot above bit zero lets each vector lane carry an independent orbit;
    // this avoids horizontal reductions in the dense four-by-four multiply.
    std::array<std::array<float64x2_t, kDimension>, Groups> input_real;
    std::array<std::array<float64x2_t, kDimension>, Groups> input_imag;
    for (size_t group = 0; group < Groups; ++group) {
        for (size_t column = 0; column < kDimension; ++column) {
            uint64_t index = representatives[group];
            if ((column & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((column & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            input_real[group][column] =
                maybe_swap_lanes(vld1q_f64(real + physical_base), sidecar.swap_lanes[column]);
            input_imag[group][column] =
                maybe_swap_lanes(vld1q_f64(imag + physical_base), sidecar.swap_lanes[column]);
        }
    }

    for (size_t row = 0; row < kDimension; ++row) {
        std::array<float64x2_t, Groups> output_real;
        std::array<float64x2_t, Groups> output_imag;
        for (size_t group = 0; group < Groups; ++group) {
            output_real[group] = vdupq_n_f64(0.0);
            output_imag[group] = vdupq_n_f64(0.0);
        }
        for (size_t column = 0; column < kDimension; ++column) {
            const LaneWeights& weight = matrix[row * kDimension + column];
            const float64x2_t weight_real = vld1q_f64(weight.real.data());
            const float64x2_t weight_imag = vld1q_f64(weight.imag.data());
            for (size_t group = 0; group < Groups; ++group) {
                output_real[group] =
                    vfmaq_f64(output_real[group], weight_real, input_real[group][column]);
                output_real[group] =
                    vfmsq_f64(output_real[group], weight_imag, input_imag[group][column]);
                output_imag[group] =
                    vfmaq_f64(output_imag[group], weight_real, input_imag[group][column]);
                output_imag[group] =
                    vfmaq_f64(output_imag[group], weight_imag, input_real[group][column]);
            }
        }

        for (size_t group = 0; group < Groups; ++group) {
            uint64_t index = representatives[group];
            if ((row & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((row & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            vst1q_f64(real + physical_base,
                      maybe_swap_lanes(output_real[group], sidecar.swap_lanes[row]));
            vst1q_f64(imag + physical_base,
                      maybe_swap_lanes(output_imag[group], sidecar.swap_lanes[row]));
        }
    }
}

void apply_fused_rotation_neon_range(State& state, const PreparedFusedRotation& rotation,
                                     const FusedRotationNeonSidecar& sidecar, uint64_t vector_begin,
                                     uint64_t vector_end) noexcept {
    if (!sidecar.pair_vectors) {
        for (uint64_t vector_index = vector_begin; vector_index < vector_end; ++vector_index) {
            apply_fused_rotation_neon_group<1>(state, rotation, sidecar, vector_index);
        }
        return;
    }
    if ((vector_begin & 1U) != 0 && vector_begin < vector_end) {
        apply_fused_rotation_neon_group<1>(state, rotation, sidecar, vector_begin++);
    }
    while (vector_begin + 1 < vector_end) {
        apply_fused_rotation_neon_group<2>(state, rotation, sidecar, vector_begin);
        vector_begin += 2;
    }
    if (vector_begin < vector_end) {
        apply_fused_rotation_neon_group<1>(state, rotation, sidecar, vector_begin);
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

MeasurementProbabilities active_diagonal_measurement_probabilities_neon_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    float64x2_t zero_sum = vdupq_n_f64(0.0);
    float64x2_t one_sum = vdupq_n_f64(0.0);
    const float64x2_t zero = vdupq_n_f64(0.0);

#pragma clang loop unroll_count(2)
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const float64x2_t input_real = vld1q_f64(real + basis);
        const float64x2_t input_imag = vld1q_f64(imag + basis);
        const float64x2_t norm =
            vfmaq_f64(vmulq_f64(input_imag, input_imag), input_real, input_real);
        const float64x2_t parity = signed_sine_lanes(basis, measurement.pauli.z, 1.0);
        const uint64x2_t zero_mask = vcgtq_f64(parity, zero);
        zero_sum = vaddq_f64(zero_sum, vbslq_f64(zero_mask, norm, zero));
        one_sum = vaddq_f64(one_sum, vbslq_f64(zero_mask, zero, norm));
    }
    return MeasurementProbabilities{vaddvq_f64(zero_sum), vaddvq_f64(one_sum)};
}

void collapse_active_diagonal_measurement_neon_impl(State& state,
                                                    const PreparedMeasurement& measurement,
                                                    bool branch,
                                                    double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pivot_bit = uint64_t{1} << measurement.pivot;
    const double scalar_scale = 1.0 / std::sqrt(branch_probability);
    assert((measurement.pauli.z & (pivot_bit - 1)) == 0 &&
           "NEON diagonal compaction requires the lowest measured pivot");

    if (measurement.pivot >= 1) {
        const float64x2_t scale = vdupq_n_f64(scalar_scale);
        for (uint64_t packed = 0; packed < measurement.output_size; packed += kLanes) {
            const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
            const bool other_parity =
                (std::popcount(source0 & measurement.z_without_pivot) & 1U) != 0;
            const uint64_t source = source0 | (branch != other_parity ? pivot_bit : 0);
            vst1q_f64(real + packed, vmulq_f64(scale, vld1q_f64(real + source)));
            vst1q_f64(imag + packed, vmulq_f64(scale, vld1q_f64(imag + source)));
        }
    } else {
        for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
            const bool other_parity =
                (std::popcount(basis & measurement.z_without_pivot) & 1U) != 0;
            const bool selected_lane_one = branch != other_parity;
            const float64x2_t selected_real = vld1q_f64(real + basis);
            const float64x2_t selected_imag = vld1q_f64(imag + basis);
            real[basis / 2] = scalar_scale * (selected_lane_one ? vgetq_lane_f64(selected_real, 1)
                                                                : vgetq_lane_f64(selected_real, 0));
            imag[basis / 2] = scalar_scale * (selected_lane_one ? vgetq_lane_f64(selected_imag, 1)
                                                                : vgetq_lane_f64(selected_imag, 0));
        }
    }
    state.set_active_width(state.active_width() - 1);
}

}  // namespace

MeasurementProbabilities active_measurement_probabilities_neon(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept {
    (void)kernel;
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 6 && kernel == ActiveMeasurementKernel::Diagonal &&
           measurement.pauli.is_diagonal() &&
           "NEON active measurement requires a profitable diagonal shape");
    return active_diagonal_measurement_probabilities_neon_impl(state, measurement);
}

void collapse_active_measurement_neon(State& state, const PreparedMeasurement& measurement,
                                      ActiveMeasurementKernel kernel, bool branch,
                                      double branch_probability) noexcept {
    (void)kernel;
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 6 && kernel == ActiveMeasurementKernel::Diagonal &&
           measurement.pauli.is_diagonal() && is_finite_robust(branch_probability) &&
           branch_probability > 0.0 &&
           "NEON active measurement requires a positive-probability diagonal shape");
    collapse_active_diagonal_measurement_neon_impl(state, measurement, branch, branch_probability);
}

void apply_direct_rotation_neon(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept {
    assert(state.active_width() == rotation.pauli.active_width && state.active_width() >= 3 &&
           "NEON rotation requires a profitable matching active width");
    const double sine = sign ? -rotation.sine : rotation.sine;
    switch (kernel) {
        case DirectRotationKernel::Diagonal:
            assert(rotation.pauli.is_diagonal() &&
                   "NEON diagonal rotation requires a diagonal Pauli");
            apply_diagonal_rotation_neon_range(state, rotation, sine, 0, state.size() / kLanes);
            return;
        case DirectRotationKernel::HighPivot:
            assert(!rotation.pauli.is_diagonal() && rotation.pauli.pairing_bit >= kLanes &&
                   "NEON high-pivot rotation requires a non-diagonal high pairing bit");
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_high_pivot_rotation_neon_range<true>(state, rotation, sine, 0,
                                                           state.size() / (2 * kLanes));
            } else {
                apply_high_pivot_rotation_neon_range<false>(state, rotation, sine, 0,
                                                            state.size() / (2 * kLanes));
            }
            return;
        case DirectRotationKernel::LanePaired:
            assert(!rotation.pauli.is_diagonal() && rotation.pauli.x == 1 &&
                   "NEON lane-paired rotation requires the low pairing bit");
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_lane_paired_rotation_neon_range<true>(state, rotation, sine, 0,
                                                            state.size() / kLanes);
            } else {
                apply_lane_paired_rotation_neon_range<false>(state, rotation, sine, 0,
                                                             state.size() / kLanes);
            }
            return;
        case DirectRotationKernel::Scalar:
            assert(false && "scalar rotations must not enter the NEON kernel");
            return;
    }
    assert(false && "unknown direct rotation kernel");
}

void apply_direct_rotation_neon_parallel(State& state, const PreparedRotation& rotation,
                                         DirectRotationKernel kernel, bool sign, uint32_t workers,
                                         uint32_t min_active_width) noexcept {
    if (!should_parallelize_intra_shot(state.active_width(), workers, min_active_width)) {
        apply_direct_rotation_neon(state, rotation, kernel, sign);
        return;
    }
    assert(state.active_width() == rotation.pauli.active_width && state.active_width() >= 3 &&
           "NEON rotation requires a profitable matching active width");
    const double sine = sign ? -rotation.sine : rotation.sine;
    switch (kernel) {
        case DirectRotationKernel::Diagonal:
            intra_shot_parallel_ranges(
                state.size() / kLanes, workers, [&](uint64_t begin, uint64_t end) noexcept {
                    apply_diagonal_rotation_neon_range(state, rotation, sine, begin, end);
                });
            return;
        case DirectRotationKernel::HighPivot:
            intra_shot_parallel_ranges(state.size() / (2 * kLanes), workers,
                                       [&](uint64_t begin, uint64_t end) noexcept {
                                           if (rotation.pauli.even_phase.real() != 0.0) {
                                               apply_high_pivot_rotation_neon_range<true>(
                                                   state, rotation, sine, begin, end);
                                           } else {
                                               apply_high_pivot_rotation_neon_range<false>(
                                                   state, rotation, sine, begin, end);
                                           }
                                       });
            return;
        case DirectRotationKernel::LanePaired:
            intra_shot_parallel_ranges(state.size() / kLanes, workers,
                                       [&](uint64_t begin, uint64_t end) noexcept {
                                           if (rotation.pauli.even_phase.real() != 0.0) {
                                               apply_lane_paired_rotation_neon_range<true>(
                                                   state, rotation, sine, begin, end);
                                           } else {
                                               apply_lane_paired_rotation_neon_range<false>(
                                                   state, rotation, sine, begin, end);
                                           }
                                       });
            return;
        case DirectRotationKernel::Scalar:
            assert(false && "scalar rotations must not enter the NEON kernel");
            return;
    }
    assert(false && "unknown direct rotation kernel");
}

FusedRotationSidecar prepare_fused_rotation_neon_sidecar(const PreparedFusedRotation& rotation) {
    if (rotation.orbit_rank != 2 || rotation.orbit_pivots[0] < 1 ||
        rotation.orbit_pivots[0] >= rotation.orbit_pivots[1]) {
        return {};
    }

    auto sidecar = std::make_shared<FusedRotationNeonSidecar>();
    // Adjacent vector groups can reuse all matrix weights when their varying
    // representative bit is absent from every selector parity.
    uint32_t adjacent_physical_bit = 1;
    for (uint32_t pivot : rotation.orbit_pivots) {
        if (pivot == adjacent_physical_bit) {
            ++adjacent_physical_bit;
        }
    }
    const uint64_t adjacent_physical_mask = uint64_t{1} << adjacent_physical_bit;
    sidecar->pair_vectors = true;
    for (uint64_t selector_mask : rotation.selector_masks) {
        sidecar->pair_vectors &= (selector_mask & adjacent_physical_mask) == 0;
    }
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
