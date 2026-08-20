// AVX-512F+AVX-512DQ sampling kernels. This translation unit is compiled with
// explicit AVX2/BMI2/FMA/AVX-512 flags so portable code retains its baseline ISA.

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
#include <vector>

namespace clifft::sampling {

namespace {

// State stores amplitudes in separate real and imaginary arrays. A SIMD lane
// is one basis index in either array, so matching lanes from a real vector and
// an imaginary vector represent eight consecutive complex amplitudes.
constexpr size_t kLanes = kAvx512DoubleLanes;
constexpr size_t kDimension = 4;
constexpr size_t kMatrixSize = kDimension * kDimension;
constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

// Measuring a Pauli P pairs |b> with |b xor x>. Lanes whose measurement pivot
// is zero enumerate the four pairs in a vector block exactly once: pivot zero,
// for example, selects |0>, |2>, |4>, and |6>. Fixed permutations place those
// representatives in the low half for an ordinary four-double store; hardware
// compress-stores were substantially slower on the performance host.
constexpr std::array<__mmask8, 3> kMeasurementSourceMasks = {0x55, 0x33, 0x0f};

using LaneIndices = std::array<uint64_t, kLanes>;
using LaneSigns = std::array<double, kLanes>;

alignas(64) constexpr std::array<LaneIndices, 3> kMeasurementCompactionPermutations = {{
    {0, 2, 4, 6, 0, 0, 0, 0},
    {0, 1, 4, 5, 0, 0, 0, 0},
    {0, 1, 2, 3, 0, 0, 0, 0},
}};

constexpr std::array<LaneIndices, kLanes> make_lane_permutations() {
    std::array<LaneIndices, kLanes> result{};
    for (size_t lane_xor = 0; lane_xor < kLanes; ++lane_xor) {
        for (size_t lane = 0; lane < kLanes; ++lane) {
            result[lane_xor][lane] = lane ^ lane_xor;
        }
    }
    return result;
}

constexpr std::array<std::array<LaneIndices, 2>, kLanes> make_diagonal_compaction_permutations() {
    std::array<std::array<LaneIndices, 2>, kLanes> result{};
    for (size_t z = 1; z < kLanes; ++z) {
        for (size_t parity = 0; parity < 2; ++parity) {
            size_t destination = 0;
            for (size_t lane = 0; lane < kLanes; ++lane) {
                if ((std::popcount(z & lane) & 1U) == parity) {
                    result[z][parity][destination++] = lane;
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

alignas(64) constexpr auto kLanePermutations = make_lane_permutations();
alignas(64) constexpr auto kLaneParitySigns = make_lane_parity_signs();
alignas(64) constexpr auto kDiagonalCompactionPermutations =
    make_diagonal_compaction_permutations();

struct alignas(64) LanePermutation {
    std::array<uint64_t, kLanes> indices{};
};

struct alignas(64) LaneWeights {
    std::array<double, kLanes> real{};
    std::array<double, kLanes> imag{};
};

struct FusedRotationAvx512Sidecar {
    std::array<LanePermutation, kDimension> permutations;
    std::vector<LaneWeights> weights;
};

// Each vector block shares the parity from basis bits above the lanes. The
// lookup supplies the remaining per-lane signs without scalar branching.
__m512d signed_sine_lanes(uint64_t basis, uint64_t z, double sine) noexcept {
    const bool high_parity = (std::popcount(basis & z) & 1U) != 0;
    const double block_sine = high_parity ? -sine : sine;
    const __m512d lane_signs = _mm512_load_pd(kLaneParitySigns[z & (kLanes - 1)].data());
    return _mm512_mul_pd(_mm512_set1_pd(block_sine), lane_signs);
}

// A diagonal Pauli never moves coefficients, so eight amplitudes can be
// updated in place with only lane-dependent parity signs.
void apply_diagonal_rotation_avx512(State& state, const PreparedRotation& rotation,
                                    double sine) noexcept {
    assert(rotation.pauli.is_diagonal() && !rotation.pauli.is_identity() &&
           rotation.pauli.active_width >= 3 &&
           "AVX-512 diagonal rotation requires at least one vector block");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const __m512d cosine = _mm512_set1_pd(rotation.cosine);
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m512d input_real = _mm512_load_pd(real + basis);
        const __m512d input_imag = _mm512_load_pd(imag + basis);
        const __m512d signed_sine = signed_sine_lanes(basis, rotation.pauli.z, sine);
        const __m512d output_real =
            _mm512_fmadd_pd(signed_sine, input_imag, _mm512_mul_pd(cosine, input_real));
        const __m512d output_imag =
            _mm512_fnmadd_pd(signed_sine, input_real, _mm512_mul_pd(cosine, input_imag));
        _mm512_store_pd(real + basis, output_real);
        _mm512_store_pd(imag + basis, output_imag);
    }
}

// A low pairing pivot keeps both members of every coefficient pair in one
// vector. Computing all lanes from the original block avoids scalar pair
// enumeration. Hermiticity determines the partner phase from the current lane,
// avoiding another permutation or a gather.
template <bool RealPhase>
void apply_lane_paired_rotation_avx512(State& state, const PreparedRotation& rotation,
                                       double sine) noexcept {
    assert(!rotation.pauli.is_diagonal() && rotation.pauli.pairing_bit < kLanes &&
           rotation.pauli.active_width >= 3 &&
           "AVX-512 lane-paired rotation requires one vector block");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t lane_xor = rotation.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const __m512d cosine = _mm512_set1_pd(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();

    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m512d input_real = _mm512_load_pd(real + basis);
        const __m512d input_imag = _mm512_load_pd(imag + basis);
        const __m512d partner_real = _mm512_permutexvar_pd(permutation, input_real);
        const __m512d partner_imag = _mm512_permutexvar_pd(permutation, input_imag);
        const __m512d basis_sine = signed_sine_lanes(basis, rotation.pauli.z, sine * base_phase);
        const __m512d partner_sine =
            RealPhase ? basis_sine : _mm512_sub_pd(_mm512_setzero_pd(), basis_sine);

        __m512d output_real;
        __m512d output_imag;
        if constexpr (RealPhase) {
            output_real =
                _mm512_fmadd_pd(partner_sine, partner_imag, _mm512_mul_pd(cosine, input_real));
            output_imag =
                _mm512_fnmadd_pd(partner_sine, partner_real, _mm512_mul_pd(cosine, input_imag));
        } else {
            output_real =
                _mm512_fmadd_pd(partner_sine, partner_real, _mm512_mul_pd(cosine, input_real));
            output_imag =
                _mm512_fmadd_pd(partner_sine, partner_imag, _mm512_mul_pd(cosine, input_imag));
        }
        _mm512_store_pd(real + basis, output_real);
        _mm512_store_pd(imag + basis, output_imag);
    }
}

// A pivot at or above the lane bits pairs two aligned vector blocks. XORing the
// full X mask finds the partner block, while one fixed permutation accounts
// for X bits within each block.
template <bool RealPhase>
void apply_nondiagonal_rotation_avx512(State& state, const PreparedRotation& rotation,
                                       double sine) noexcept {
    assert(!rotation.pauli.is_diagonal() && rotation.pauli.pairing_bit >= kLanes &&
           "AVX-512 non-diagonal rotation requires a high pairing pivot");
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pair_stride = rotation.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = rotation.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const __m512d cosine = _mm512_set1_pd(rotation.cosine);
    const double base_phase =
        RealPhase ? rotation.pauli.even_phase.real() : rotation.pauli.even_phase.imag();
    const double even_left_sine = sine * base_phase;

    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ rotation.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m512d left_real = _mm512_load_pd(real + left);
            const __m512d left_imag = _mm512_load_pd(imag + left);
            const __m512d right_real =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(real + right_base));
            const __m512d right_imag =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(imag + right_base));
            const __m512d left_sine = signed_sine_lanes(left, rotation.pauli.z, even_left_sine);

            __m512d output_left_real;
            __m512d output_left_imag;
            __m512d output_right_real;
            __m512d output_right_imag;
            if constexpr (RealPhase) {
                output_left_real =
                    _mm512_fmadd_pd(left_sine, right_imag, _mm512_mul_pd(cosine, left_real));
                output_left_imag =
                    _mm512_fnmadd_pd(left_sine, right_real, _mm512_mul_pd(cosine, left_imag));
                output_right_real =
                    _mm512_fmadd_pd(left_sine, left_imag, _mm512_mul_pd(cosine, right_real));
                output_right_imag =
                    _mm512_fnmadd_pd(left_sine, left_real, _mm512_mul_pd(cosine, right_imag));
            } else {
                output_left_real =
                    _mm512_fnmadd_pd(left_sine, right_real, _mm512_mul_pd(cosine, left_real));
                output_left_imag =
                    _mm512_fnmadd_pd(left_sine, right_imag, _mm512_mul_pd(cosine, left_imag));
                output_right_real =
                    _mm512_fmadd_pd(left_sine, left_real, _mm512_mul_pd(cosine, right_real));
                output_right_imag =
                    _mm512_fmadd_pd(left_sine, left_imag, _mm512_mul_pd(cosine, right_imag));
            }

            _mm512_store_pd(real + left, output_left_real);
            _mm512_store_pd(imag + left, output_left_imag);
            _mm512_store_pd(real + right_base,
                            _mm512_permutexvar_pd(permutation, output_right_real));
            _mm512_store_pd(imag + right_base,
                            _mm512_permutexvar_pd(permutation, output_right_imag));
        }
    }
}

// Fused matrices vary with representative parity. The sidecar expands those
// choices by lane so the hot loop can traverse eight independent orbits without
// gathers or selector branches.
void apply_fused_rotation_avx512(State& state, const PreparedFusedRotation& rotation,
                                 const void* opaque_sidecar) noexcept {
    const auto& sidecar = *static_cast<const FusedRotationAvx512Sidecar*>(opaque_sidecar);
    assert(rotation.orbit_rank == 2 && rotation.orbit_pivots[0] >= 3 &&
           rotation.orbit_pivots[0] < rotation.orbit_pivots[1] &&
           "AVX-512 fused rotation requires ordered high-pivot rank-two orbits");
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

        std::array<__m512d, kDimension> input_real;
        std::array<__m512d, kDimension> input_imag;
        for (size_t column = 0; column < kDimension; ++column) {
            uint64_t index = representative;
            if ((column & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((column & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            const __m512i permutation =
                _mm512_load_si512(sidecar.permutations[column].indices.data());
            input_real[column] =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(real + physical_base));
            input_imag[column] =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(imag + physical_base));
        }

        for (size_t row = 0; row < kDimension; ++row) {
            __m512d output_real = _mm512_setzero_pd();
            __m512d output_imag = _mm512_setzero_pd();
            for (size_t column = 0; column < kDimension; ++column) {
                const LaneWeights& weight = matrix[row * kDimension + column];
                const __m512d weight_real = _mm512_load_pd(weight.real.data());
                const __m512d weight_imag = _mm512_load_pd(weight.imag.data());
                output_real = _mm512_fmadd_pd(weight_real, input_real[column], output_real);
                output_real = _mm512_fnmadd_pd(weight_imag, input_imag[column], output_real);
                output_imag = _mm512_fmadd_pd(weight_real, input_imag[column], output_imag);
                output_imag = _mm512_fmadd_pd(weight_imag, input_real[column], output_imag);
            }

            uint64_t index = representative;
            if ((row & 1U) != 0) {
                index ^= rotation.orbit_masks[0];
            }
            if ((row & 2U) != 0) {
                index ^= rotation.orbit_masks[1];
            }
            const uint64_t physical_base = index & ~(uint64_t{kLanes - 1});
            const __m512i permutation = _mm512_load_si512(sidecar.permutations[row].indices.data());
            _mm512_store_pd(real + physical_base, _mm512_permutexvar_pd(permutation, output_real));
            _mm512_store_pd(imag + physical_base, _mm512_permutexvar_pd(permutation, output_imag));
        }
    }
}

MeasurementProbabilities active_diagonal_measurement_probabilities_avx512_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    __m512d zero_sum = _mm512_setzero_pd();
    __m512d one_sum = _mm512_setzero_pd();

    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m512d input_real = _mm512_load_pd(real + basis);
        const __m512d input_imag = _mm512_load_pd(imag + basis);
        const __m512d norm =
            _mm512_fmadd_pd(input_real, input_real, _mm512_mul_pd(input_imag, input_imag));
        const __m512d parity = signed_sine_lanes(basis, measurement.pauli.z, 1.0);
        const __mmask8 zero_mask = _mm512_cmp_pd_mask(parity, _mm512_setzero_pd(), _CMP_GT_OQ);
        zero_sum = _mm512_mask_add_pd(zero_sum, zero_mask, zero_sum, norm);
        one_sum = _mm512_mask_add_pd(one_sum, static_cast<__mmask8>(~zero_mask), one_sum, norm);
    }
    return MeasurementProbabilities{_mm512_reduce_add_pd(zero_sum), _mm512_reduce_add_pd(one_sum)};
}

void collapse_active_diagonal_measurement_avx512_impl(State& state,
                                                      const PreparedMeasurement& measurement,
                                                      bool branch,
                                                      double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pivot_bit = uint64_t{1} << measurement.pivot;
    const __m512d scale = _mm512_set1_pd(1.0 / std::sqrt(branch_probability));
    assert((measurement.pauli.z & (pivot_bit - 1)) == 0 &&
           "AVX-512 diagonal compaction requires the lowest measured pivot");

    if (measurement.pivot >= 3) {
        for (uint64_t packed = 0; packed < measurement.output_size; packed += kLanes) {
            const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
            const bool other_parity =
                (std::popcount(source0 & measurement.z_without_pivot) & 1U) != 0;
            const uint64_t source = source0 | (branch != other_parity ? pivot_bit : 0);
            const __m512d selected_real = _mm512_load_pd(real + source);
            const __m512d selected_imag = _mm512_load_pd(imag + source);
            _mm512_store_pd(real + packed, _mm512_mul_pd(scale, selected_real));
            _mm512_store_pd(imag + packed, _mm512_mul_pd(scale, selected_imag));
        }
    } else {
        const uint64_t lane_z = measurement.pauli.z & (kLanes - 1);
        for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
            const bool high_parity = (std::popcount(basis & measurement.pauli.z) & 1U) != 0;
            const size_t selected_lane_parity = static_cast<size_t>(branch != high_parity);
            const __m512i compaction = _mm512_load_si512(
                kDiagonalCompactionPermutations[lane_z][selected_lane_parity].data());
            const __m256d selected_real = _mm512_castpd512_pd256(
                _mm512_permutexvar_pd(compaction, _mm512_load_pd(real + basis)));
            const __m256d selected_imag = _mm512_castpd512_pd256(
                _mm512_permutexvar_pd(compaction, _mm512_load_pd(imag + basis)));
            _mm256_storeu_pd(real + basis / 2,
                             _mm256_mul_pd(_mm512_castpd512_pd256(scale), selected_real));
            _mm256_storeu_pd(imag + basis / 2,
                             _mm256_mul_pd(_mm512_castpd512_pd256(scale), selected_imag));
        }
    }
    state.set_active_width(state.active_width() - 1);
}

template <bool RealPhase>
MeasurementProbabilities active_measurement_probabilities_avx512_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const __mmask8 source_mask = kMeasurementSourceMasks[measurement.pivot];
    const double base_coefficient =
        RealPhase ? measurement.pauli.even_phase.real() : -measurement.pauli.even_phase.imag();
    __m512d zero_sum = _mm512_setzero_pd();
    __m512d one_sum = _mm512_setzero_pd();

    // For eigenvalue e in {+1, -1}, one compacted branch amplitude is
    //   (psi[b] + e * conj(phase(P, b)) * psi[b xor x]) / sqrt(2).
    // Build both eigenvalue branches together, then count only the pivot-zero
    // representative of each pair when accumulating their squared norms.
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m512d input_real = _mm512_load_pd(real + basis);
        const __m512d input_imag = _mm512_load_pd(imag + basis);
        const __m512d partner_real = _mm512_permutexvar_pd(permutation, input_real);
        const __m512d partner_imag = _mm512_permutexvar_pd(permutation, input_imag);
        const __m512d coefficient = signed_sine_lanes(basis, measurement.pauli.z, base_coefficient);

        __m512d zero_real;
        __m512d zero_imag;
        __m512d one_real;
        __m512d one_imag;
        // Prepared Pauli phases are in {+1, -1, +i, -i}. Splitting real and
        // imaginary phase classes turns the complex multiply into FMAs.
        if constexpr (RealPhase) {
            zero_real = _mm512_fmadd_pd(coefficient, partner_real, input_real);
            zero_imag = _mm512_fmadd_pd(coefficient, partner_imag, input_imag);
            one_real = _mm512_fnmadd_pd(coefficient, partner_real, input_real);
            one_imag = _mm512_fnmadd_pd(coefficient, partner_imag, input_imag);
        } else {
            zero_real = _mm512_fnmadd_pd(coefficient, partner_imag, input_real);
            zero_imag = _mm512_fmadd_pd(coefficient, partner_real, input_imag);
            one_real = _mm512_fmadd_pd(coefficient, partner_imag, input_real);
            one_imag = _mm512_fnmadd_pd(coefficient, partner_real, input_imag);
        }
        const __m512d zero_norm =
            _mm512_fmadd_pd(zero_real, zero_real, _mm512_mul_pd(zero_imag, zero_imag));
        const __m512d one_norm =
            _mm512_fmadd_pd(one_real, one_real, _mm512_mul_pd(one_imag, one_imag));
        zero_sum = _mm512_mask_add_pd(zero_sum, source_mask, zero_sum, zero_norm);
        one_sum = _mm512_mask_add_pd(one_sum, source_mask, one_sum, one_norm);
    }
    return MeasurementProbabilities{0.5 * _mm512_reduce_add_pd(zero_sum),
                                    0.5 * _mm512_reduce_add_pd(one_sum)};
}

template <bool RealPhase>
MeasurementProbabilities active_measurement_probabilities_high_pivot_avx512_impl(
    const State& state, const PreparedMeasurement& measurement) noexcept {
    const double* const real = state.real_data();
    const double* const imag = state.imag_data();
    const uint64_t pair_stride = measurement.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const double base_coefficient =
        RealPhase ? measurement.pauli.even_phase.real() : -measurement.pauli.even_phase.imag();
    __m512d zero_sum = _mm512_setzero_pd();
    __m512d one_sum = _mm512_setzero_pd();

    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ measurement.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m512d left_real = _mm512_load_pd(real + left);
            const __m512d left_imag = _mm512_load_pd(imag + left);
            const __m512d right_real =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(real + right_base));
            const __m512d right_imag =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(imag + right_base));
            const __m512d coefficient =
                signed_sine_lanes(left, measurement.pauli.z, base_coefficient);

            __m512d zero_real;
            __m512d zero_imag;
            __m512d one_real;
            __m512d one_imag;
            if constexpr (RealPhase) {
                zero_real = _mm512_fmadd_pd(coefficient, right_real, left_real);
                zero_imag = _mm512_fmadd_pd(coefficient, right_imag, left_imag);
                one_real = _mm512_fnmadd_pd(coefficient, right_real, left_real);
                one_imag = _mm512_fnmadd_pd(coefficient, right_imag, left_imag);
            } else {
                zero_real = _mm512_fnmadd_pd(coefficient, right_imag, left_real);
                zero_imag = _mm512_fmadd_pd(coefficient, right_real, left_imag);
                one_real = _mm512_fmadd_pd(coefficient, right_imag, left_real);
                one_imag = _mm512_fnmadd_pd(coefficient, right_real, left_imag);
            }
            const __m512d zero_norm =
                _mm512_fmadd_pd(zero_real, zero_real, _mm512_mul_pd(zero_imag, zero_imag));
            const __m512d one_norm =
                _mm512_fmadd_pd(one_real, one_real, _mm512_mul_pd(one_imag, one_imag));
            zero_sum = _mm512_add_pd(zero_sum, zero_norm);
            one_sum = _mm512_add_pd(one_sum, one_norm);
        }
    }
    return MeasurementProbabilities{0.5 * _mm512_reduce_add_pd(zero_sum),
                                    0.5 * _mm512_reduce_add_pd(one_sum)};
}

template <bool RealPhase>
void collapse_active_measurement_avx512_impl(State& state, const PreparedMeasurement& measurement,
                                             bool branch, double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const __m512i compaction =
        _mm512_load_si512(kMeasurementCompactionPermutations[measurement.pivot].data());
    const double eigenvalue = branch ? -1.0 : 1.0;
    const double base_coefficient = eigenvalue * (RealPhase ? measurement.pauli.even_phase.real()
                                                            : -measurement.pauli.even_phase.imag());
    const __m512d scale = _mm512_set1_pd(kInvSqrt2 / std::sqrt(branch_probability));

    // Apply the selected (I + eP) branch and normalize it. Removing the pivot
    // maps each eight-amplitude block to four consecutive output amplitudes;
    // each input block is in registers before that compacted output is written,
    // so forward traversal cannot overwrite a future source.
    for (uint64_t basis = 0; basis < state.size(); basis += kLanes) {
        const __m512d input_real = _mm512_load_pd(real + basis);
        const __m512d input_imag = _mm512_load_pd(imag + basis);
        const __m512d partner_real = _mm512_permutexvar_pd(permutation, input_real);
        const __m512d partner_imag = _mm512_permutexvar_pd(permutation, input_imag);
        const __m512d coefficient = signed_sine_lanes(basis, measurement.pauli.z, base_coefficient);

        __m512d output_real;
        __m512d output_imag;
        if constexpr (RealPhase) {
            output_real = _mm512_fmadd_pd(coefficient, partner_real, input_real);
            output_imag = _mm512_fmadd_pd(coefficient, partner_imag, input_imag);
        } else {
            output_real = _mm512_fnmadd_pd(coefficient, partner_imag, input_real);
            output_imag = _mm512_fmadd_pd(coefficient, partner_real, input_imag);
        }
        output_real = _mm512_mul_pd(scale, output_real);
        output_imag = _mm512_mul_pd(scale, output_imag);
        const __m256d compact_real =
            _mm512_castpd512_pd256(_mm512_permutexvar_pd(compaction, output_real));
        const __m256d compact_imag =
            _mm512_castpd512_pd256(_mm512_permutexvar_pd(compaction, output_imag));
        _mm256_storeu_pd(real + basis / 2, compact_real);
        _mm256_storeu_pd(imag + basis / 2, compact_imag);
    }
    state.set_active_width(state.active_width() - 1);
}

template <bool RealPhase>
void collapse_active_measurement_high_pivot_avx512_impl(State& state,
                                                        const PreparedMeasurement& measurement,
                                                        bool branch,
                                                        double branch_probability) noexcept {
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    const uint64_t pair_stride = measurement.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const uint64_t lane_xor = measurement.pauli.x & (kLanes - 1);
    const __m512i permutation = _mm512_load_si512(kLanePermutations[lane_xor].data());
    const double eigenvalue = branch ? -1.0 : 1.0;
    const double base_coefficient = eigenvalue * (RealPhase ? measurement.pauli.even_phase.real()
                                                            : -measurement.pauli.even_phase.imag());
    const __m512d scale = _mm512_set1_pd(kInvSqrt2 / std::sqrt(branch_probability));

    // A highest-bit pivot splits each pair period into contiguous source
    // halves. Loading both halves before writing their compacted destination
    // keeps forward traversal safe without scratch storage.
    for (uint64_t block = 0; block < state.size(); block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; offset += kLanes) {
            const uint64_t left = block + offset;
            const uint64_t right_base = (left ^ measurement.pauli.x) & ~(uint64_t{kLanes - 1});
            const __m512d left_real = _mm512_load_pd(real + left);
            const __m512d left_imag = _mm512_load_pd(imag + left);
            const __m512d right_real =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(real + right_base));
            const __m512d right_imag =
                _mm512_permutexvar_pd(permutation, _mm512_load_pd(imag + right_base));
            const __m512d coefficient =
                signed_sine_lanes(left, measurement.pauli.z, base_coefficient);

            __m512d output_real;
            __m512d output_imag;
            if constexpr (RealPhase) {
                output_real = _mm512_fmadd_pd(coefficient, right_real, left_real);
                output_imag = _mm512_fmadd_pd(coefficient, right_imag, left_imag);
            } else {
                output_real = _mm512_fnmadd_pd(coefficient, right_imag, left_real);
                output_imag = _mm512_fmadd_pd(coefficient, right_real, left_imag);
            }
            const uint64_t destination = block / 2 + offset;
            _mm512_store_pd(real + destination, _mm512_mul_pd(scale, output_real));
            _mm512_store_pd(imag + destination, _mm512_mul_pd(scale, output_imag));
        }
    }
    state.set_active_width(state.active_width() - 1);
}

}  // namespace

MeasurementProbabilities active_measurement_probabilities_avx512(
    const State& state, const PreparedMeasurement& measurement,
    ActiveMeasurementKernel kernel) noexcept {
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 4 && kernel != ActiveMeasurementKernel::Scalar &&
           "AVX-512 active measurement requires a profitable vector width");
    if (kernel == ActiveMeasurementKernel::Diagonal) {
        assert(measurement.pauli.is_diagonal() &&
               "AVX-512 diagonal measurement requires a diagonal Pauli");
        return active_diagonal_measurement_probabilities_avx512_impl(state, measurement);
    }
    assert(!measurement.pauli.is_diagonal() &&
           "AVX-512 paired measurement requires a non-diagonal Pauli");
    if (kernel == ActiveMeasurementKernel::HighPivot) {
        assert(measurement.pauli.pairing_bit == (uint64_t{1} << measurement.pivot) &&
               measurement.pauli.pairing_bit >= kLanes &&
               "AVX-512 high-pivot measurement requires the highest X bit");
        if (measurement.pauli.even_phase.real() != 0.0) {
            return active_measurement_probabilities_high_pivot_avx512_impl<true>(state,
                                                                                 measurement);
        }
        return active_measurement_probabilities_high_pivot_avx512_impl<false>(state, measurement);
    }
    assert(kernel == ActiveMeasurementKernel::LanePaired && measurement.pauli.x < kLanes &&
           measurement.pivot < 3 &&
           "AVX-512 lane-paired measurement requires a low-lane Pauli pairing");
    if (measurement.pauli.even_phase.real() != 0.0) {
        return active_measurement_probabilities_avx512_impl<true>(state, measurement);
    }
    return active_measurement_probabilities_avx512_impl<false>(state, measurement);
}

void collapse_active_measurement_avx512(State& state, const PreparedMeasurement& measurement,
                                        ActiveMeasurementKernel kernel, bool branch,
                                        double branch_probability) noexcept {
    assert(state.active_width() == measurement.pauli.active_width &&
           measurement.pauli.active_width >= 4 && kernel != ActiveMeasurementKernel::Scalar &&
           is_finite_robust(branch_probability) && branch_probability > 0.0 &&
           "AVX-512 active measurement requires a positive-probability vector width");
    if (kernel == ActiveMeasurementKernel::Diagonal) {
        assert(measurement.pauli.is_diagonal() &&
               "AVX-512 diagonal measurement requires a diagonal Pauli");
        collapse_active_diagonal_measurement_avx512_impl(state, measurement, branch,
                                                         branch_probability);
        return;
    }
    assert(!measurement.pauli.is_diagonal() &&
           "AVX-512 paired measurement requires a non-diagonal Pauli");
    if (kernel == ActiveMeasurementKernel::HighPivot) {
        assert(measurement.pauli.pairing_bit == (uint64_t{1} << measurement.pivot) &&
               measurement.pauli.pairing_bit >= kLanes &&
               "AVX-512 high-pivot measurement requires the highest X bit");
        if (measurement.pauli.even_phase.real() != 0.0) {
            collapse_active_measurement_high_pivot_avx512_impl<true>(state, measurement, branch,
                                                                     branch_probability);
        } else {
            collapse_active_measurement_high_pivot_avx512_impl<false>(state, measurement, branch,
                                                                      branch_probability);
        }
        return;
    }
    assert(kernel == ActiveMeasurementKernel::LanePaired && measurement.pauli.x < kLanes &&
           measurement.pivot < 3 &&
           "AVX-512 lane-paired measurement requires a low-lane Pauli pairing");
    if (measurement.pauli.even_phase.real() != 0.0) {
        collapse_active_measurement_avx512_impl<true>(state, measurement, branch,
                                                      branch_probability);
    } else {
        collapse_active_measurement_avx512_impl<false>(state, measurement, branch,
                                                       branch_probability);
    }
}

void apply_direct_rotation_avx512(State& state, const PreparedRotation& rotation,
                                  DirectRotationKernel kernel, bool sign) noexcept {
    assert(state.active_width() == rotation.pauli.active_width &&
           "AVX-512 rotation width must match the active state");
    const double sine = sign ? -rotation.sine : rotation.sine;
    switch (kernel) {
        case DirectRotationKernel::Diagonal:
            apply_diagonal_rotation_avx512(state, rotation, sine);
            return;
        case DirectRotationKernel::HighPivot:
            // Prepared Paulis have phases in {+1, -1, +i, -i}; specializing
            // the real and imaginary cases avoids generic complex arithmetic.
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_nondiagonal_rotation_avx512<true>(state, rotation, sine);
            } else {
                apply_nondiagonal_rotation_avx512<false>(state, rotation, sine);
            }
            return;
        case DirectRotationKernel::LanePaired:
            if (rotation.pauli.even_phase.real() != 0.0) {
                apply_lane_paired_rotation_avx512<true>(state, rotation, sine);
            } else {
                apply_lane_paired_rotation_avx512<false>(state, rotation, sine);
            }
            return;
        case DirectRotationKernel::Scalar:
            assert(false && "scalar rotations must not enter the AVX-512 kernel");
            return;
    }
    assert(false && "unknown direct rotation kernel");
}

// Host-specific preparation transposes selector matrices into lane-major
// weights once, keeping both allocation and selector expansion out of shots.
FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(const PreparedFusedRotation& rotation) {
    if (rotation.orbit_rank != 2 || rotation.orbit_pivots[0] < 3 ||
        rotation.orbit_pivots[0] >= rotation.orbit_pivots[1]) {
        return {};
    }

    auto sidecar = std::make_shared<FusedRotationAvx512Sidecar>();
    for (size_t member = 0; member < kDimension; ++member) {
        uint64_t mask = 0;
        if ((member & 1U) != 0) {
            mask ^= rotation.orbit_masks[0];
        }
        if ((member & 2U) != 0) {
            mask ^= rotation.orbit_masks[1];
        }
        const uint64_t lane_xor = mask & (kLanes - 1);
        for (size_t lane = 0; lane < kLanes; ++lane) {
            sidecar->permutations[member].indices[lane] = lane ^ lane_xor;
        }
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

    return FusedRotationSidecar{std::move(sidecar), apply_fused_rotation_avx512};
}

}  // namespace clifft::sampling
