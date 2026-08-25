#include "clifft/sampling/interleaved_batch_kernels.h"

#include "clifft/sampling/indexing.h"

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>

namespace clifft::sampling {

namespace {

template <size_t Dimension>
void apply_interleaved_fused_orbits(InterleavedBatchState& state,
                                    const PreparedFusedRotation& rotation) noexcept {
    static_assert(Dimension == 1 || Dimension == 2 || Dimension == 4);
    const uint64_t orbit_count = state.size() / Dimension;
    const size_t matrix_size = Dimension * Dimension;
    assert(rotation.matrices.size() ==
               (size_t{1} << rotation.selector_masks.size()) * matrix_size &&
           "fused rotation matrix table must cover every selector value");
    const size_t lanes = state.active_lanes();
    for (uint64_t packed = 0; packed < orbit_count; ++packed) {
        uint64_t representative = packed;
        if constexpr (Dimension >= 2) {
            representative = insert_zero_bit(representative, rotation.orbit_pivots[0]);
        }
        if constexpr (Dimension == 4) {
            representative = insert_zero_bit(representative, rotation.orbit_pivots[1]);
        }

        size_t selector = 0;
        for (size_t bit = 0; bit < rotation.selector_masks.size(); ++bit) {
            selector |= static_cast<size_t>(
                            std::popcount(representative & rotation.selector_masks[bit]) & 1U)
                        << bit;
        }
        const std::complex<double>* const matrix =
            rotation.matrices.data() + selector * matrix_size;
        std::array<double*, Dimension> real{};
        std::array<double*, Dimension> imag{};
        for (size_t column = 0; column < Dimension; ++column) {
            uint64_t index = representative;
            if constexpr (Dimension >= 2) {
                if ((column & 1U) != 0) {
                    index ^= rotation.orbit_masks[0];
                }
            }
            if constexpr (Dimension == 4) {
                if ((column & 2U) != 0) {
                    index ^= rotation.orbit_masks[1];
                }
            }
            real[column] = state.real_basis(index);
            imag[column] = state.imag_basis(index);
        }

#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
        for (size_t lane = 0; lane < lanes; ++lane) {
            std::array<double, Dimension> input_real{};
            std::array<double, Dimension> input_imag{};
            for (size_t column = 0; column < Dimension; ++column) {
                input_real[column] = real[column][lane];
                input_imag[column] = imag[column][lane];
            }
            for (size_t row = 0; row < Dimension; ++row) {
                double output_real = 0.0;
                double output_imag = 0.0;
                for (size_t column = 0; column < Dimension; ++column) {
                    const std::complex<double> weight = matrix[row * Dimension + column];
                    output_real +=
                        weight.real() * input_real[column] - weight.imag() * input_imag[column];
                    output_imag +=
                        weight.real() * input_imag[column] + weight.imag() * input_real[column];
                }
                real[row][lane] = output_real;
                imag[row][lane] = output_imag;
            }
        }
    }
}

template <size_t Dimension>
void apply_interleaved_dynamic_fused_orbits(
    InterleavedBatchState& state, std::span<const PreparedFusedRotation> variants,
    std::span<const uint8_t> lane_variants) noexcept {
    static_assert(Dimension == 1 || Dimension == 2 || Dimension == 4);
    assert(!variants.empty() && variants.size() <= 4 &&
           "dynamic fused rotation must have one to four variants");
    const PreparedFusedRotation& geometry = variants.front();
    const uint64_t orbit_count = state.size() / Dimension;
    const size_t matrix_size = Dimension * Dimension;
    [[maybe_unused]] const size_t expected_matrices =
        (size_t{1} << geometry.selector_masks.size()) * matrix_size;
    for ([[maybe_unused]] const PreparedFusedRotation& variant : variants) {
        assert(variant.active_width == geometry.active_width &&
               variant.orbit_rank == geometry.orbit_rank &&
               variant.orbit_masks == geometry.orbit_masks &&
               variant.orbit_pivots == geometry.orbit_pivots &&
               variant.selector_masks == geometry.selector_masks &&
               variant.matrices.size() == expected_matrices &&
               "dynamic fused variants must share compiler-prepared geometry");
    }
    const size_t lanes = state.active_lanes();
    for (uint64_t packed = 0; packed < orbit_count; ++packed) {
        uint64_t representative = packed;
        if constexpr (Dimension >= 2) {
            representative = insert_zero_bit(representative, geometry.orbit_pivots[0]);
        }
        if constexpr (Dimension == 4) {
            representative = insert_zero_bit(representative, geometry.orbit_pivots[1]);
        }
        size_t selector = 0;
        for (size_t bit = 0; bit < geometry.selector_masks.size(); ++bit) {
            selector |= static_cast<size_t>(
                            std::popcount(representative & geometry.selector_masks[bit]) & 1U)
                        << bit;
        }
        std::array<const std::complex<double>*, 4> matrices{};
        for (size_t variant = 0; variant < variants.size(); ++variant) {
            matrices[variant] = variants[variant].matrices.data() + selector * matrix_size;
        }
        std::array<double*, Dimension> real{};
        std::array<double*, Dimension> imag{};
        for (size_t column = 0; column < Dimension; ++column) {
            uint64_t index = representative;
            if constexpr (Dimension >= 2) {
                if ((column & 1U) != 0) {
                    index ^= geometry.orbit_masks[0];
                }
            }
            if constexpr (Dimension == 4) {
                if ((column & 2U) != 0) {
                    index ^= geometry.orbit_masks[1];
                }
            }
            real[column] = state.real_basis(index);
            imag[column] = state.imag_basis(index);
        }

#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
        for (size_t lane = 0; lane < lanes; ++lane) {
            assert(lane_variants[lane] < variants.size() &&
                   "dynamic fused lane must select a prepared variant");
            const std::complex<double>* const matrix = matrices[lane_variants[lane]];
            std::array<double, Dimension> input_real{};
            std::array<double, Dimension> input_imag{};
            for (size_t column = 0; column < Dimension; ++column) {
                input_real[column] = real[column][lane];
                input_imag[column] = imag[column][lane];
            }
            for (size_t row = 0; row < Dimension; ++row) {
                double output_real = 0.0;
                double output_imag = 0.0;
                for (size_t column = 0; column < Dimension; ++column) {
                    const std::complex<double> weight = matrix[row * Dimension + column];
                    output_real +=
                        weight.real() * input_real[column] - weight.imag() * input_imag[column];
                    output_imag +=
                        weight.real() * input_imag[column] + weight.imag() * input_real[column];
                }
                real[row][lane] = output_real;
                imag[row][lane] = output_imag;
            }
        }
    }
}

}  // namespace

void prepare_interleaved_rotation_sines(std::span<double> output, double sine,
                                        std::span<const uint8_t> signs) noexcept {
    assert(output.size() >= signs.size() && "signed-sine output must cover every lane");
#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
    for (size_t lane = 0; lane < signs.size(); ++lane) {
        output[lane] = signs[lane] != 0 ? -sine : sine;
    }
}

void apply_interleaved_rotation(InterleavedBatchState& state,
                                const PreparedRotation& rotation,
                                std::span<const double> signed_sines) noexcept {
    assert(state.active_width() == rotation.pauli.active_width &&
           "prepared rotation width must match interleaved state");
    assert(!rotation.pauli.is_identity() && "identity rotations must be removed during planning");
    assert(signed_sines.size() >= state.active_lanes() &&
           "rotation signs must cover every live lane");
    const size_t lanes = state.active_lanes();
    const uint64_t size = state.size();
    if (rotation.pauli.is_diagonal()) {
        for (uint64_t basis = 0; basis < size; ++basis) {
            double* real = state.real_basis(basis);
            double* imag = state.imag_basis(basis);
            const double eigenvalue =
                (std::popcount(basis & rotation.pauli.z) & 1U) != 0 ? -1.0 : 1.0;
#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
            for (size_t lane = 0; lane < lanes; ++lane) {
                const double r = real[lane];
                const double i = imag[lane];
                const double sine = signed_sines[lane] * eigenvalue;
                real[lane] = rotation.cosine * r + sine * i;
                imag[lane] = rotation.cosine * i - sine * r;
            }
        }
        return;
    }

    const uint64_t pair_stride = rotation.pauli.pairing_bit;
    const uint64_t pair_period = pair_stride << 1;
    const bool real_phase = rotation.pauli.even_phase.real() != 0.0;
    const double base_phase = real_phase ? rotation.pauli.even_phase.real()
                                         : rotation.pauli.even_phase.imag();
    for (uint64_t block = 0; block < size; block += pair_period) {
        for (uint64_t offset = 0; offset < pair_stride; ++offset) {
            const uint64_t left = block + offset;
            const uint64_t right = left ^ rotation.pauli.x;
            double* left_real = state.real_basis(left);
            double* left_imag = state.imag_basis(left);
            double* right_real = state.real_basis(right);
            double* right_imag = state.imag_basis(right);
            const bool odd_phase = (std::popcount(left & rotation.pauli.z) & 1U) != 0;
            const double left_phase = odd_phase ? -base_phase : base_phase;
            const double right_phase = real_phase ? left_phase : -left_phase;
#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
            for (size_t lane = 0; lane < lanes; ++lane) {
                const double lr = left_real[lane];
                const double li = left_imag[lane];
                const double rr = right_real[lane];
                const double ri = right_imag[lane];
                const double left_sine = signed_sines[lane] * left_phase;
                const double right_sine = signed_sines[lane] * right_phase;
                if (real_phase) {
                    left_real[lane] = rotation.cosine * lr + right_sine * ri;
                    left_imag[lane] = rotation.cosine * li - right_sine * rr;
                    right_real[lane] = rotation.cosine * rr + left_sine * li;
                    right_imag[lane] = rotation.cosine * ri - left_sine * lr;
                } else {
                    left_real[lane] = rotation.cosine * lr + right_sine * rr;
                    left_imag[lane] = rotation.cosine * li + right_sine * ri;
                    right_real[lane] = rotation.cosine * rr + left_sine * lr;
                    right_imag[lane] = rotation.cosine * ri + left_sine * li;
                }
            }
        }
    }
}

void apply_interleaved_promotion(InterleavedBatchState& state,
                                 const PreparedPromotion& promotion,
                                 std::span<const double> signed_sines) noexcept {
    assert(state.active_width() < state.max_active_width() &&
           "promotion must fit interleaved batch storage");
    assert(signed_sines.size() >= state.active_lanes() &&
           "promotion signs must cover every live lane");
    const size_t lanes = state.active_lanes();
    const uint64_t old_size = state.size();
    for (uint64_t basis = 0; basis < old_size; ++basis) {
        double* real = state.real_basis(basis);
        double* imag = state.imag_basis(basis);
        double* promoted_real = state.real_basis(old_size + basis);
        double* promoted_imag = state.imag_basis(old_size + basis);
#if defined(CLIFFT_USE_OPENMP)
#pragma omp simd
#endif
        for (size_t lane = 0; lane < lanes; ++lane) {
            const double r = real[lane];
            const double i = imag[lane];
            real[lane] = promotion.cosine * r;
            imag[lane] = promotion.cosine * i;
            promoted_real[lane] = signed_sines[lane] * i;
            promoted_imag[lane] = -signed_sines[lane] * r;
        }
    }
    state.set_active_width(state.active_width() + 1);
}

void apply_interleaved_fused_rotation(InterleavedBatchState& state,
                                      const PreparedFusedRotation& rotation) noexcept {
    assert(state.active_width() == rotation.active_width &&
           "fused rotation width must match interleaved state");
    switch (rotation.orbit_rank) {
        case 0:
            apply_interleaved_fused_orbits<1>(state, rotation);
            return;
        case 1:
            apply_interleaved_fused_orbits<2>(state, rotation);
            return;
        case 2:
            apply_interleaved_fused_orbits<4>(state, rotation);
            return;
        default:
            assert(false && "fused rotation orbit rank must be at most two");
            return;
    }
}

void apply_interleaved_dynamic_fused_rotation(
    InterleavedBatchState& state, std::span<const PreparedFusedRotation> variants,
    std::span<const uint8_t> lane_variants) noexcept {
    assert(!variants.empty() && state.active_width() == variants.front().active_width &&
           lane_variants.size() >= state.active_lanes() &&
           "dynamic fused rotation inputs must match interleaved state");
    switch (variants.front().orbit_rank) {
        case 0:
            apply_interleaved_dynamic_fused_orbits<1>(state, variants, lane_variants);
            return;
        case 1:
            apply_interleaved_dynamic_fused_orbits<2>(state, variants, lane_variants);
            return;
        case 2:
            apply_interleaved_dynamic_fused_orbits<4>(state, variants, lane_variants);
            return;
        default:
            assert(false && "dynamic fused rotation orbit rank must be at most two");
            return;
    }
}

}  // namespace clifft::sampling
