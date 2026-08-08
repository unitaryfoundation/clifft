// AVX-512F+AVX-512DQ fused-rotation kernel. This translation unit is compiled
// with the same explicit AVX2/BMI2/FMA/AVX-512 flags as the SVM AVX-512 path.

#include "clifft/sampling/fused_rotation_simd.h"
#include "clifft/sampling/indexing.h"

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <vector>

namespace clifft::sampling {

namespace {

constexpr size_t kLanes = 8;
constexpr size_t kDimension = 4;
constexpr size_t kMatrixSize = kDimension * kDimension;

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

void apply_fused_rotation_avx512(State& state, const PreparedFusedRotation& rotation,
                                 const void* opaque_sidecar) noexcept {
    const auto& sidecar = *static_cast<const FusedRotationAvx512Sidecar*>(opaque_sidecar);
    assert(rotation.orbit_rank == 2 && rotation.orbit_pivots[0] >= 3 &&
           "AVX-512 fused rotation requires a high-pivot rank-two orbit");
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

}  // namespace

FusedRotationSidecar prepare_fused_rotation_avx512_sidecar(const PreparedFusedRotation& rotation) {
    if (rotation.orbit_rank != 2 || rotation.orbit_pivots[0] < 3) {
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
