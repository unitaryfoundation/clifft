// AVX2+BMI2+FMA sampling kernels. This translation unit is compiled with
// explicit ISA flags so portable builds can select it at runtime.

#include "clifft/sampling/indexing.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/simd_width.h"
#include "clifft/util/numeric.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <utility>
#include <vector>

namespace clifft::sampling {

namespace {

// State stores real and imaginary parts separately. Matching lanes from the
// two arrays therefore represent four consecutive complex amplitudes.
constexpr size_t kLanes = kAvx2DoubleLanes;
constexpr size_t kDimension = 4;
constexpr size_t kMatrixSize = kDimension * kDimension;
constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

using LanePermutationIndices = std::array<int32_t, 2 * kLanes>;
using LaneSigns = std::array<double, kLanes>;

constexpr std::array<LanePermutationIndices, kLanes> make_lane_permutations() {
    std::array<LanePermutationIndices, kLanes> result{};
    for (size_t lane_xor = 0; lane_xor < kLanes; ++lane_xor) {
        for (size_t lane = 0; lane < kLanes; ++lane) {
            const int32_t source = static_cast<int32_t>(lane ^ lane_xor);
            result[lane_xor][2 * lane] = 2 * source;
            result[lane_xor][2 * lane + 1] = 2 * source + 1;
        }
    }
    return result;
}

constexpr std::array<std::array<LanePermutationIndices, 2>, kLanes>
make_diagonal_compaction_permutations() {
    std::array<std::array<LanePermutationIndices, 2>, kLanes> result{};
    for (size_t z = 1; z < kLanes; ++z) {
        for (size_t parity = 0; parity < 2; ++parity) {
            size_t destination = 0;
            for (size_t lane = 0; lane < kLanes; ++lane) {
                if ((std::popcount(z & lane) & 1U) == parity) {
                    result[z][parity][2 * destination] = static_cast<int32_t>(2 * lane);
                    result[z][parity][2 * destination + 1] = static_cast<int32_t>(2 * lane + 1);
                    ++destination;
                }
            }
        }
    }
    return result;
}

constexpr std::array<LaneSigns, kLanes> make_lane_parity_signs() {
    std::array<LaneSigns, kLanes> result{};
    for (size_t z = 0; z < kLanes; ++z) {
        for (size_t lane = 0; lane < kLanes; ++lane) {
            result[z][lane] = (std::popcount(z & lane) & 1U) != 0 ? -1.0 : 1.0;
        }
    }
    return result;
}

alignas(32) constexpr auto kLanePermutations = make_lane_permutations();
alignas(32) constexpr auto kLaneParitySigns = make_lane_parity_signs();
alignas(32) constexpr auto kDiagonalCompactionPermutations =
    make_diagonal_compaction_permutations();

alignas(32) constexpr std::array<LanePermutationIndices, 2> kMeasurementCompactionPermutations = {{
    {0, 1, 4, 5, 0, 0, 0, 0},
    {0, 1, 2, 3, 0, 0, 0, 0},
}};
alignas(32) constexpr std::array<LaneSigns, 2> kMeasurementSourceWeights = {{
    {1.0, 0.0, 1.0, 0.0},
    {1.0, 1.0, 0.0, 0.0},
}};

struct alignas(32) LanePermutation {
    LanePermutationIndices indices{};
};

struct alignas(32) LaneWeights {
    std::array<double, kLanes> real{};
    std::array<double, kLanes> imag{};
};

struct FusedRotationAvx2Sidecar {
    std::array<LanePermutation, kDimension> permutations;
    std::vector<LaneWeights> weights;
};

__m256d permute_lanes(__m256d input, __m256i permutation) noexcept {
    return _mm256_castsi256_pd(
        _mm256_permutevar8x32_epi32(_mm256_castpd_si256(input), permutation));
}

double reduce_add(__m256d value) noexcept {
    const __m128d halves =
        _mm_add_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
    return _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)));
}

__m256d signed_sine_lanes(uint64_t basis, uint64_t z, double sine) noexcept {
    const bool high_parity = (std::popcount(basis & z) & 1U) != 0;
    const double block_sine = high_parity ? -sine : sine;
    const __m256d lane_signs = _mm256_load_pd(kLaneParitySigns[z & (kLanes - 1)].data());
    return _mm256_mul_pd(_mm256_set1_pd(block_sine), lane_signs);
}

void apply_diagonal_rotation_avx2(State& state, const PreparedRotation& rotation,
                                  double sine) noexcept {
    assert(rotation.pauli.is_diagonal() && !rotation.pauli.is_identity() &&
           rotation.pauli.active_width >= 2 &&
           "AVX2 diagonal rotation requires at least one vector block");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const __m256d cosine = _mm256_set1_pd(rotation.cosine);
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        const __m256d signed_sine = signed_sine_lanes(basis, rotation.pauli.z, sine);
        const __m256d output_real =
            _mm256_fmadd_pd(signed_sine, input_imag, _mm256_mul_pd(cosine, input_real));
        const __m256d output_imag =
            _mm256_fnmadd_pd(signed_sine, input_real, _mm256_mul_pd(cosine, input_imag));
        _mm256_store_pd(real + basis, output_real);
        _mm256_store_pd(imag + basis, output_imag);
    }
}

template <bool RealPhase>
void apply_lane_paired_rotation_avx2(State& state, const PreparedRotation& rotation,
                                     double sine) noexcept {
    assert(!rotation.pauli.is_diagonal() && rotation.pauli.pairing_bit < kLanes &&
           rotation.pauli.active_width >= 2 &&
           "AVX2 lane-paired rotation requires at least one vector block");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t lane_xor = rotation.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const __m256d cosine = _mm256_set1_pd(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();

    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        const __m256d partner_real = permute_lanes(input_real, permutation);
        const __m256d partner_imag = permute_lanes(input_imag, permutation);
        const __m256d basis_sine = signed_sine_lanes(basis, rotation.pauli.z, sine * base_phase);
        const __m256d partner_sine =
            RealPhase ? basis_sine : _mm256_sub_pd(_mm256_setzero_pd(), basis_sine);

        __m256d output_real;
        __m256d output_imag;
        if constexpr (RealPhase) {
            output_real =
                _mm256_fmadd_pd(partner_sine, partner_imag, _mm256_mul_pd(cosine, input_real));
            output_imag =
                _mm256_fnmadd_pd(partner_sine, partner_real, _mm256_mul_pd(cosine, input_imag));
        } else {
            output_real =
                _mm256_fmadd_pd(partner_sine, partner_real, _mm256_mul_pd(cosine, input_real));
            output_imag =
                _mm256_fmadd_pd(partner_sine, partner_imag, _mm256_mul_pd(cosine, input_imag));
        }
        _mm256_store_pd(real + basis, output_real);
        _mm256_store_pd(imag + basis, output_imag);
    }
}

template <bool RealPhase>
void apply_nondiagonal_rotation_avx2(State& state, const PreparedRotation& rotation,
                                     double sine) noexcept {
    assert(!rotation.pauli.is_diagonal() && rotation.pauli.pairing_bit >= kLanes &&
           "AVX2 non-diagonal rotation requires a high pairing pivot");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pair_stride = rotation.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = rotation.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const __m256d cosine = _mm256_set1_pd(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();
    const double even_left_sine = sine * base_phase;

    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ rotation.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m256d left_real = _mm256_load_pd(real + left);
            const __m256d left_imag = _mm256_load_pd(imag + left);
            const __m256d right_real =
                permute_lanes(_mm256_load_pd(real + right_base), permutation);
            const __m256d right_imag =
                permute_lanes(_mm256_load_pd(imag + right_base), permutation);
            const __m256d left_sine = signed_sine_lanes(left, rotation.pauli.z, even_left_sine);

            __m256d output_left_real;
            __m256d output_left_imag;
            __m256d output_right_real;
            __m256d output_right_imag;
            if constexpr (RealPhase) {
                output_left_real =
                    _mm256_fmadd_pd(left_sine, right_imag, _mm256_mul_pd(cosine, left_real));
                output_left_imag =
                    _mm256_fnmadd_pd(left_sine, right_real, _mm256_mul_pd(cosine, left_imag));
                output_right_real =
                    _mm256_fmadd_pd(left_sine, left_imag, _mm256_mul_pd(cosine, right_real));
                output_right_imag =
                    _mm256_fnmadd_pd(left_sine, left_real, _mm256_mul_pd(cosine, right_imag));
            } else {
                output_left_real =
                    _mm256_fnmadd_pd(left_sine, right_real, _mm256_mul_pd(cosine, left_real));
                output_left_imag =
                    _mm256_fnmadd_pd(left_sine, right_imag, _mm256_mul_pd(cosine, left_imag));
                output_right_real =
                    _mm256_fmadd_pd(left_sine, left_real, _mm256_mul_pd(cosine, right_real));
                output_right_imag =
                    _mm256_fmadd_pd(left_sine, left_imag, _mm256_mul_pd(cosine, right_imag));
            }

            _mm256_store_pd(real + left, output_left_real);
            _mm256_store_pd(imag + left, output_left_imag);
            _mm256_store_pd(real + right_base, permute_lanes(output_right_real, permutation));
            _mm256_store_pd(imag + right_base, permute_lanes(output_right_imag, permutation));
        }
    }
}

void apply_fused_rotation_avx2(State& state, const PreparedFusedRotation& rotation,
                               const void* opaque_sidecar) noexcept {
    const auto& sidecar = *static_cast<const FusedRotationAvx2Sidecar*>(opaque_sidecar);
    assert(rotation.orbit_rank == 2 && rotation.orbit_pivots[0] >= 2 &&
           rotation.orbit_pivots[0] < rotation.orbit_pivots[1] &&
           "AVX2 fused rotation requires ordered high-pivot rank-two orbits");
    assert(state.active_width() == rotation.active_width &&
           "fused rotation width must match the active state");

    const uint64_t orbit_count = state.size() / kDimension;
    double* const real = state.real_data();
    double* const imag = state.imag_data();

    for (uint64_t packed = 0; packed < orbit_count; packed += kLanes) {
        uint64_t representative = insert_zero_bit(packed, rotation.orbit_pivots[0]);
        representative = insert_zero_bit(representative, rotation.orbit_pivots[1]);
        const LaneWeights* const matrix =
            sidecar.weights.data() +
            selector_index(representative, rotation.selector_masks) * kMatrixSize;

        std::array<__m256d, kDimension> input_real;
        std::array<__m256d, kDimension> input_imag;
        for (size_t column = 0; column < kDimension; ++column) {
            uint64_t index = representative;
            if ((column & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((column & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            const __m256i permutation = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(sidecar.permutations[column].indices.data()));
            input_real[column] = permute_lanes(_mm256_load_pd(real + physical_base), permutation);
            input_imag[column] = permute_lanes(_mm256_load_pd(imag + physical_base), permutation);
        }

        for (size_t row = 0; row < kDimension; ++row) {
            __m256d output_real = _mm256_setzero_pd();
            __m256d output_imag = _mm256_setzero_pd();
            for (size_t column = 0; column < kDimension; ++column) {
                const LaneWeights& weight = matrix[row * kDimension + column];
                const __m256d weight_real = _mm256_load_pd(weight.real.data());
                const __m256d weight_imag = _mm256_load_pd(weight.imag.data());
                output_real = _mm256_fmadd_pd(weight_real, input_real[column], output_real);
                output_real = _mm256_fnmadd_pd(weight_imag, input_imag[column], output_real);
                output_imag = _mm256_fmadd_pd(weight_real, input_imag[column], output_imag);
                output_imag = _mm256_fmadd_pd(weight_imag, input_real[column], output_imag);
            }

            uint64_t index = representative;
            if ((row & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((row & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            const __m256i permutation = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(sidecar.permutations[row].indices.data()));
            _mm256_store_pd(real + physical_base, permute_lanes(output_real, permutation));
            _mm256_store_pd(imag + physical_base, permute_lanes(output_imag, permutation));
        }
    }
}

MeasurementProbabilities active_diagonal_measurement_probabilities_avx2_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    __m256d zero_sum = _mm256_setzero_pd();
    __m256d one_sum = _mm256_setzero_pd();

    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        const __m256d norm =
            _mm256_fmadd_pd(input_real, input_real, _mm256_mul_pd(input_imag, input_imag));
        const __m256d parity = signed_sine_lanes(basis, measurement.pauli.z, 1.0);
        const __m256d zero_mask = _mm256_cmp_pd(parity, _mm256_setzero_pd(), _CMP_GT_OQ);
        zero_sum = _mm256_add_pd(zero_sum, _mm256_and_pd(zero_mask, norm));
        one_sum = _mm256_add_pd(one_sum, _mm256_andnot_pd(zero_mask, norm));
    }
    return MeasurementProbabilities{reduce_add(zero_sum), reduce_add(one_sum)};
}

void collapse_active_diagonal_measurement_avx2_impl(State& state,
                                                    const PreparedMeasurement& measurement,
                                                    bool branch,
                                                    double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pivot_bit = uint64_t{1} << measurement.pivot;
    const __m256d scale = _mm256_set1_pd(1.0 / std::sqrt(branch_probability));
    assert((measurement.pauli.z & (pivot_bit - 1)) == 0 &&
           "AVX2 diagonal compaction requires the lowest measured pivot");

    if (measurement.pivot >= 2) {
        for (uint64_t packed = 0; packed < measurement.output_size; packed += kLanes) {
            const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
            const bool other_parity =
                (std::popcount(source0 & measurement.z_without_pivot) & 1U) != 0;
            const uint64_t source = source0 | (branch != other_parity ? pivot_bit : 0);
            const __m256d selected_real = _mm256_load_pd(real + source);
            const __m256d selected_imag = _mm256_load_pd(imag + source);
            _mm256_store_pd(real + packed, _mm256_mul_pd(scale, selected_real));
            _mm256_store_pd(imag + packed, _mm256_mul_pd(scale, selected_imag));
        }
    } else {
        const uint64_t lane_z = measurement.pauli.z & (kLanes - 1);
        for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
            const bool high_parity = (std::popcount(basis & measurement.pauli.z) & 1U) != 0;
            const size_t selected_lane_parity = static_cast<size_t>(branch != high_parity);
            const __m256i compaction = _mm256_load_si256(reinterpret_cast<const __m256i*>(
                kDiagonalCompactionPermutations[lane_z][selected_lane_parity].data()));
            const __m128d selected_real =
                _mm256_castpd256_pd128(permute_lanes(_mm256_load_pd(real + basis), compaction));
            const __m128d selected_imag =
                _mm256_castpd256_pd128(permute_lanes(_mm256_load_pd(imag + basis), compaction));
            _mm_storeu_pd(real + basis / 2,
                          _mm_mul_pd(_mm256_castpd256_pd128(scale), selected_real));
            _mm_storeu_pd(imag + basis / 2,
                          _mm_mul_pd(_mm256_castpd256_pd128(scale), selected_imag));
        }
    }
    state.set_active_width(state.active_width() - 1);
}

template <bool RealPhase>
MeasurementProbabilities active_measurement_probabilities_avx2_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const __m256d source_weights =
        _mm256_load_pd(kMeasurementSourceWeights[measurement.pivot & 1U].data());
    const double base_coefficient =
        RealPhase ? measurement.pauli.even_phase.real() : -measurement.pauli.even_phase.imag();
    __m256d zero_sum = _mm256_setzero_pd();
    __m256d one_sum = _mm256_setzero_pd();

    // For eigenvalue e in {+1, -1}, form both projected branches together.
    // Only pivot-zero representatives contribute to their squared norms; the
    // source weights select those two lanes without an AVX-512-style mask.
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        const __m256d partner_real = permute_lanes(input_real, permutation);
        const __m256d partner_imag = permute_lanes(input_imag, permutation);
        const __m256d coefficient = signed_sine_lanes(basis, measurement.pauli.z, base_coefficient);

        __m256d zero_real;
        __m256d zero_imag;
        __m256d one_real;
        __m256d one_imag;
        // Prepared Pauli phases are in {+1, -1, +i, -i}. Splitting their real
        // and imaginary cases turns the complex multiply into fixed FMAs.
        if constexpr (RealPhase) {
            zero_real = _mm256_fmadd_pd(coefficient, partner_real, input_real);
            zero_imag = _mm256_fmadd_pd(coefficient, partner_imag, input_imag);
            one_real = _mm256_fnmadd_pd(coefficient, partner_real, input_real);
            one_imag = _mm256_fnmadd_pd(coefficient, partner_imag, input_imag);
        } else {
            zero_real = _mm256_fnmadd_pd(coefficient, partner_imag, input_real);
            zero_imag = _mm256_fmadd_pd(coefficient, partner_real, input_imag);
            one_real = _mm256_fmadd_pd(coefficient, partner_imag, input_real);
            one_imag = _mm256_fnmadd_pd(coefficient, partner_real, input_imag);
        }
        const __m256d zero_norm =
            _mm256_fmadd_pd(zero_real, zero_real, _mm256_mul_pd(zero_imag, zero_imag));
        const __m256d one_norm =
            _mm256_fmadd_pd(one_real, one_real, _mm256_mul_pd(one_imag, one_imag));
        zero_sum = _mm256_fmadd_pd(source_weights, zero_norm, zero_sum);
        one_sum = _mm256_fmadd_pd(source_weights, one_norm, one_sum);
    }
    return MeasurementProbabilities{0.5 * reduce_add(zero_sum), 0.5 * reduce_add(one_sum)};
}

template <bool RealPhase>
MeasurementProbabilities active_measurement_probabilities_high_pivot_avx2_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    const uint64_t pair_stride = measurement.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const double base_coefficient =
        RealPhase ? measurement.pauli.even_phase.real() : -measurement.pauli.even_phase.imag();
    __m256d zero_sum = _mm256_setzero_pd();
    __m256d one_sum = _mm256_setzero_pd();

    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ measurement.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m256d left_real = _mm256_load_pd(real + left);
            const __m256d left_imag = _mm256_load_pd(imag + left);
            const __m256d right_real =
                permute_lanes(_mm256_load_pd(real + right_base), permutation);
            const __m256d right_imag =
                permute_lanes(_mm256_load_pd(imag + right_base), permutation);
            const __m256d coefficient =
                signed_sine_lanes(left, measurement.pauli.z, base_coefficient);

            __m256d zero_real;
            __m256d zero_imag;
            __m256d one_real;
            __m256d one_imag;
            if constexpr (RealPhase) {
                zero_real = _mm256_fmadd_pd(coefficient, right_real, left_real);
                zero_imag = _mm256_fmadd_pd(coefficient, right_imag, left_imag);
                one_real = _mm256_fnmadd_pd(coefficient, right_real, left_real);
                one_imag = _mm256_fnmadd_pd(coefficient, right_imag, left_imag);
            } else {
                zero_real = _mm256_fnmadd_pd(coefficient, right_imag, left_real);
                zero_imag = _mm256_fmadd_pd(coefficient, right_real, left_imag);
                one_real = _mm256_fmadd_pd(coefficient, right_imag, left_real);
                one_imag = _mm256_fnmadd_pd(coefficient, right_real, left_imag);
            }
            const __m256d zero_norm =
                _mm256_fmadd_pd(zero_real, zero_real, _mm256_mul_pd(zero_imag, zero_imag));
            const __m256d one_norm =
                _mm256_fmadd_pd(one_real, one_real, _mm256_mul_pd(one_imag, one_imag));
            zero_sum = _mm256_add_pd(zero_sum, zero_norm);
            one_sum = _mm256_add_pd(one_sum, one_norm);
        }
    }
    return MeasurementProbabilities{0.5 * reduce_add(zero_sum), 0.5 * reduce_add(one_sum)};
}

template <bool RealPhase>
void collapse_active_measurement_avx2_impl(State& state, const PreparedMeasurement& measurement,
                                           bool branch, double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const __m256i compaction = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        kMeasurementCompactionPermutations[measurement.pivot & 1U].data()));
    const double eigenvalue = branch ? -1.0 : 1.0;
    const double base_coefficient = eigenvalue * (RealPhase ? measurement.pauli.even_phase.real()
                                                            : -measurement.pauli.even_phase.imag());
    const __m256d scale = _mm256_set1_pd(kInvSqrt2 / std::sqrt(branch_probability));

    // Removing either low-lane pivot maps each four-amplitude block to two
    // consecutive outputs. Every source block is loaded before its compacted
    // destination is written, so forward traversal cannot overwrite a future source.
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        const __m256d partner_real = permute_lanes(input_real, permutation);
        const __m256d partner_imag = permute_lanes(input_imag, permutation);
        const __m256d coefficient = signed_sine_lanes(basis, measurement.pauli.z, base_coefficient);

        __m256d output_real;
        __m256d output_imag;
        if constexpr (RealPhase) {
            output_real = _mm256_fmadd_pd(coefficient, partner_real, input_real);
            output_imag = _mm256_fmadd_pd(coefficient, partner_imag, input_imag);
        } else {
            output_real = _mm256_fnmadd_pd(coefficient, partner_imag, input_real);
            output_imag = _mm256_fmadd_pd(coefficient, partner_real, input_imag);
        }
        output_real = _mm256_mul_pd(scale, output_real);
        output_imag = _mm256_mul_pd(scale, output_imag);
        const __m128d compact_real = _mm256_castpd256_pd128(permute_lanes(output_real, compaction));
        const __m128d compact_imag = _mm256_castpd256_pd128(permute_lanes(output_imag, compaction));
        _mm_storeu_pd(real + basis / 2, compact_real);
        _mm_storeu_pd(imag + basis / 2, compact_imag);
    }
    state.set_active_width(state.active_width() - 1);
}

template <bool RealPhase>
void collapse_active_measurement_high_pivot_avx2_impl(State& state,
                                                      const PreparedMeasurement& measurement,
                                                      bool branch,
                                                      double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pair_stride = measurement.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m256i permutation =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanePermutations[lane_xor].data()));
    const double eigenvalue = branch ? -1.0 : 1.0;
    const double base_coefficient = eigenvalue * (RealPhase ? measurement.pauli.even_phase.real()
                                                            : -measurement.pauli.even_phase.imag());
    const __m256d scale = _mm256_set1_pd(kInvSqrt2 / std::sqrt(branch_probability));

    // A highest-bit pivot splits each pair period into contiguous source
    // halves. Loading both halves before writing their compacted destination
    // keeps forward traversal safe without scratch storage.
    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ measurement.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m256d left_real = _mm256_load_pd(real + left);
            const __m256d left_imag = _mm256_load_pd(imag + left);
            const __m256d right_real =
                permute_lanes(_mm256_load_pd(real + right_base), permutation);
            const __m256d right_imag =
                permute_lanes(_mm256_load_pd(imag + right_base), permutation);
            const __m256d coefficient =
                signed_sine_lanes(left, measurement.pauli.z, base_coefficient);

            __m256d output_real;
            __m256d output_imag;
            if constexpr (RealPhase) {
                output_real = _mm256_fmadd_pd(coefficient, right_real, left_real);
                output_imag = _mm256_fmadd_pd(coefficient, right_imag, left_imag);
            } else {
                output_real = _mm256_fnmadd_pd(coefficient, right_imag, left_real);
                output_imag = _mm256_fmadd_pd(coefficient, right_real, left_imag);
            }
            const uint64_t destination = block / 2 + offset;
            _mm256_store_pd(real + destination, _mm256_mul_pd(scale, output_real));
            _mm256_store_pd(imag + destination, _mm256_mul_pd(scale, output_imag));
        }
    }
    state.set_active_width(state.active_width() - 1);
}

}  // namespace

MeasurementProbabilities active_measurement_probabilities_avx2(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept {
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 2 && kernel != ActiveMeasurementKernel::Scalar &&
           "AVX2 active measurement requires a profitable vector width");
    if (kernel == ActiveMeasurementKernel::Diagonal) {
        assert(measurement.pauli.is_diagonal() &&
               "AVX2 diagonal measurement requires a diagonal Pauli");
        return active_diagonal_measurement_probabilities_avx2_impl(state, measurement);
    }
    assert(!measurement.pauli.is_diagonal() &&
           "AVX2 paired measurement requires a non-diagonal Pauli");
    if (kernel == ActiveMeasurementKernel::HighPivot) {
        assert(measurement.pauli.pairing_bit == (uint64_t{1} << measurement.pivot) &&
               measurement.pauli.pairing_bit >= kLanes &&
               "AVX2 high-pivot measurement requires the highest X bit");
        if (measurement.pauli.even_phase.real() != 0.0) {
            return active_measurement_probabilities_high_pivot_avx2_impl<true>(state, measurement);
        }
        return active_measurement_probabilities_high_pivot_avx2_impl<false>(state, measurement);
    }
    assert(kernel == ActiveMeasurementKernel::LanePaired && measurement.pauli.x < kLanes &&
           measurement.pivot < 2 &&
           "AVX2 lane-paired measurement requires a low-lane Pauli pairing");
    if (measurement.pauli.even_phase.real() != 0.0) {
        return active_measurement_probabilities_avx2_impl<true>(state, measurement);
    }
    return active_measurement_probabilities_avx2_impl<false>(state, measurement);
}

void collapse_active_measurement_avx2(State& state, const PreparedMeasurement& measurement,
                                      ActiveMeasurementKernel kernel, bool branch,
                                      double branch_probability) noexcept {
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 2 && kernel != ActiveMeasurementKernel::Scalar &&
           is_finite_robust(branch_probability) && branch_probability > 0.0 &&
           "AVX2 active measurement requires a positive-probability vector width");
    if (kernel == ActiveMeasurementKernel::Diagonal) {
        assert(measurement.pauli.is_diagonal() &&
               "AVX2 diagonal measurement requires a diagonal Pauli");
        collapse_active_diagonal_measurement_avx2_impl(state, measurement, branch,
                                                       branch_probability);
        return;
    }
    assert(!measurement.pauli.is_diagonal() &&
           "AVX2 paired measurement requires a non-diagonal Pauli");
    if (kernel == ActiveMeasurementKernel::HighPivot) {
        assert(measurement.pauli.pairing_bit == (uint64_t{1} << measurement.pivot) &&
               measurement.pauli.pairing_bit >= kLanes &&
               "AVX2 high-pivot measurement requires the highest X bit");
        if (measurement.pauli.even_phase.real() != 0.0) {
            collapse_active_measurement_high_pivot_avx2_impl<true>(state, measurement, branch,
                                                                   branch_probability);
        } else {
            collapse_active_measurement_high_pivot_avx2_impl<false>(state, measurement, branch,
                                                                    branch_probability);
        }
        return;
    }
    assert(kernel == ActiveMeasurementKernel::LanePaired && measurement.pauli.x < kLanes &&
           measurement.pivot < 2 &&
           "AVX2 lane-paired measurement requires a low-lane Pauli pairing");
    if (measurement.pauli.even_phase.real() != 0.0) {
        collapse_active_measurement_avx2_impl<true>(state, measurement, branch, branch_probability);
    } else {
        collapse_active_measurement_avx2_impl<false>(state, measurement, branch,
                                                     branch_probability);
    }
}

void apply_direct_rotation_avx2(State& state, const PreparedRotation& rotation,
                                DirectRotationKernel kernel, bool sign) noexcept {
    assert(state.active_width() == rotation.pauli.active_width &&
           "AVX2 rotation width must match the active state");
    const double sine = sign ? -rotation.sine : rotation.sine;
    switch (kernel) {
        case DirectRotationKernel::Diagonal:
            apply_diagonal_rotation_avx2(state, rotation, sine);
            return;
        case DirectRotationKernel::HighPivot:
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_nondiagonal_rotation_avx2<true>(state, rotation, sine);
            } else {
                apply_nondiagonal_rotation_avx2<false>(state, rotation, sine);
            }
            return;
        case DirectRotationKernel::LanePaired:
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_lane_paired_rotation_avx2<true>(state, rotation, sine);
            } else {
                apply_lane_paired_rotation_avx2<false>(state, rotation, sine);
            }
            return;
        case DirectRotationKernel::Scalar:
            assert(false && "scalar rotations must not enter the AVX2 kernel");
            return;
    }
    assert(false && "unknown direct rotation kernel");
}

FusedRotationSidecar prepare_fused_rotation_avx2_sidecar(const PreparedFusedRotation& rotation) {
    if (rotation.orbit_rank != 2 || rotation.orbit_pivots[0] < 2 ||
        rotation.orbit_pivots[0] >= rotation.orbit_pivots[1]) {
        return {};
    }

    auto sidecar = std::make_shared<FusedRotationAvx2Sidecar>();
    for (size_t member = 0; member < kDimension; ++member) {
        uint64_t mask = 0;
        if ((member & 1U) != 0) {
            mask ^= rotation.orbit_masks[0];
        }
        if ((member & 2U) != 0) {
            mask ^= rotation.orbit_masks[1];
        }
        const uint64_t lane_xor = mask & (kLanes - 1);
        sidecar->permutations[member].indices = kLanePermutations[lane_xor];
    }

    const size_t num_variants = size_t{1} << rotation.selector_masks.size();
    assert(rotation.matrices.size() == num_variants * kMatrixSize &&
           "fused rotation matrix table must cover every selector value");
    sidecar->weights.resize(num_variants * kMatrixSize);
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

    return FusedRotationSidecar{std::move(sidecar), apply_fused_rotation_avx2};
}

void apply_new_x_instrument_no_fire_avx2(State& state, double factor_zero, double factor_one,
                                         double no_fire_probability) noexcept {
    assert(state.active_width() >= 2 && state.active_width() < state.max_active_width() &&
           "AVX2 new-X activation requires at least one vector block and spare capacity");
    assert(factor_zero >= 0.0 && factor_zero <= 1.0 && factor_one >= 0.0 && factor_one <= 1.0 &&
           is_finite_robust(no_fire_probability) && no_fire_probability > 0.0 &&
           "AVX2 new-X no-fire expansion requires valid factors and probability");
    const double inv_norm = 1.0 / std::sqrt(no_fire_probability);
    const __m256d identity_factor = _mm256_set1_pd(0.5 * (factor_zero + factor_one) * inv_norm);
    const __m256d pauli_factor = _mm256_set1_pd(0.5 * (factor_zero - factor_one) * inv_norm);
    const uint64_t old_size = state.size();
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    for (uint64_t basis = 0; basis < old_size; basis += kLanes) {
        const __m256d input_real = _mm256_load_pd(real + basis);
        const __m256d input_imag = _mm256_load_pd(imag + basis);
        _mm256_store_pd(real + basis, _mm256_mul_pd(identity_factor, input_real));
        _mm256_store_pd(imag + basis, _mm256_mul_pd(identity_factor, input_imag));
        _mm256_store_pd(real + old_size + basis, _mm256_mul_pd(pauli_factor, input_real));
        _mm256_store_pd(imag + old_size + basis, _mm256_mul_pd(pauli_factor, input_imag));
    }
    state.set_active_width(state.active_width() + 1);
}

}  // namespace clifft::sampling
