#include "clifft/sampling/soa_kernels.h"

#include "test_helpers.h"

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

using clifft::sampling::ActivePauli;
using clifft::sampling::apply_promotion;
using clifft::sampling::apply_rotation;
using clifft::sampling::collapse_measurement;
using clifft::sampling::measurement_probabilities;
using clifft::sampling::MeasurementProbabilities;
using clifft::sampling::prepare_measurement;
using clifft::sampling::prepare_promotion;
using clifft::sampling::prepare_rotation;
using clifft::sampling::PreparedMeasurement;
using clifft::sampling::SoaState;
using clifft::test::check_complex;
using clifft::test::dense_axis_rotation;
using clifft::test::DenseMatrix;

namespace {

constexpr double kTolerance = 2e-11;
constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

std::vector<std::complex<double>> deterministic_state(uint32_t active_width) {
    const uint64_t size = uint64_t{1} << active_width;
    std::vector<std::complex<double>> result(size);
    double norm = 0.0;
    for (uint64_t i = 0; i < size; ++i) {
        const double real = 1.0 + static_cast<double>((3 * i + 1) % 11);
        const double imag = static_cast<double>((5 * i + 2) % 13) - 6.0;
        result[i] = {real, imag};
        norm += std::norm(result[i]);
    }
    const double inv_norm = 1.0 / std::sqrt(norm);
    for (std::complex<double>& value : result) {
        value *= inv_norm;
    }
    return result;
}

void load_state(SoaState& state, const std::vector<std::complex<double>>& values) {
    REQUIRE(values.size() == state.size());
    for (uint64_t i = 0; i < state.size(); ++i) {
        state.real_data()[i] = values[i].real();
        state.imag_data()[i] = values[i].imag();
    }
}

std::vector<std::complex<double>> coefficients(const SoaState& state) {
    std::vector<std::complex<double>> result(state.size());
    for (uint64_t i = 0; i < state.size(); ++i) {
        result[i] = {state.real_data()[i], state.imag_data()[i]};
    }
    return result;
}

std::vector<std::complex<double>> physical_coefficients(const SoaState& state) {
    std::vector<std::complex<double>> result = coefficients(state);
    for (std::complex<double>& value : result) {
        value *= state.global_scalar();
    }
    return result;
}

std::vector<std::complex<double>> apply_matrix(const DenseMatrix& matrix,
                                               const std::vector<std::complex<double>>& input) {
    const uint64_t size = input.size();
    REQUIRE(matrix.size() == size * size);
    std::vector<std::complex<double>> result(size, {0.0, 0.0});
    for (uint64_t row = 0; row < size; ++row) {
        for (uint64_t col = 0; col < size; ++col) {
            result[row] += matrix[row * size + col] * input[col];
        }
    }
    return result;
}

std::vector<std::complex<double>> balanced_rotation(const std::vector<std::complex<double>>& input,
                                                    uint64_t x, uint64_t z, bool sign,
                                                    double half_turns, uint32_t active_width) {
    DenseMatrix matrix = dense_axis_rotation(x, z, sign, half_turns, active_width);
    const double angle = std::numbers::pi * half_turns / 2.0;
    const std::complex<double> balance{std::cos(angle), -std::sin(angle)};
    for (std::complex<double>& value : matrix) {
        value *= balance;
    }
    return apply_matrix(matrix, input);
}

void require_vectors_close(const std::vector<std::complex<double>>& actual,
                           const std::vector<std::complex<double>>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        check_complex(actual[i], expected[i], kTolerance);
    }
}

uint64_t insert_zero_bit(uint64_t packed, uint32_t pivot) {
    const uint64_t lower_mask = (uint64_t{1} << pivot) - 1;
    return (packed & lower_mask) | ((packed & ~lower_mask) << 1);
}

std::complex<double> pauli_phase(uint64_t x, uint64_t z, uint64_t basis) {
    static constexpr std::complex<double> kIPowers[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    const uint32_t exponent =
        std::popcount(x & z) + 2U * (static_cast<uint32_t>(std::popcount(basis & z)) & 1U);
    return kIPowers[exponent & 3U];
}

struct ProjectedBranch {
    double probability = 0.0;
    std::vector<std::complex<double>> normalized;
};

ProjectedBranch dense_project(const std::vector<std::complex<double>>& input, uint64_t x,
                              uint64_t z, bool branch, uint32_t active_width) {
    const DenseMatrix pauli = dense_axis_rotation(x, z, false, 1.0, active_width);
    const std::vector<std::complex<double>> applied = apply_matrix(pauli, input);
    const double eigenvalue = branch ? -1.0 : 1.0;
    ProjectedBranch result;
    result.normalized.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result.normalized[i] = 0.5 * (input[i] + eigenvalue * applied[i]);
        result.probability += std::norm(result.normalized[i]);
    }
    if (result.probability > 0.0) {
        const double inv_norm = 1.0 / std::sqrt(result.probability);
        for (std::complex<double>& value : result.normalized) {
            value *= inv_norm;
        }
    }
    return result;
}

std::vector<std::complex<double>> expand_compacted(
    const std::vector<std::complex<double>>& compacted, const PreparedMeasurement& measurement,
    bool branch) {
    std::vector<std::complex<double>> result(compacted.size() * 2, {0.0, 0.0});
    if (measurement.pauli.is_diagonal()) {
        for (uint64_t packed = 0; packed < compacted.size(); ++packed) {
            const uint64_t without_pivot = insert_zero_bit(packed, measurement.pivot);
            const bool other_parity =
                (std::popcount(without_pivot & measurement.z_without_pivot) & 1U) != 0;
            const bool pivot_value = branch != other_parity;
            const uint64_t source =
                without_pivot | (static_cast<uint64_t>(pivot_value) << measurement.pivot);
            result[source] = compacted[packed];
        }
        return result;
    }

    const double eigenvalue = branch ? -1.0 : 1.0;
    for (uint64_t packed = 0; packed < compacted.size(); ++packed) {
        const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
        const uint64_t source1 = source0 ^ measurement.pauli.x;
        const std::complex<double> phase =
            pauli_phase(measurement.pauli.x, measurement.pauli.z, source0);
        result[source0] = kInvSqrt2 * compacted[packed];
        result[source1] = kInvSqrt2 * eigenvalue * phase * compacted[packed];
    }
    return result;
}

std::vector<std::complex<double>> compact_projected(const ProjectedBranch& projected,
                                                    const PreparedMeasurement& measurement,
                                                    bool branch) {
    std::vector<std::complex<double>> result(measurement.output_size);
    if (measurement.pauli.is_diagonal()) {
        for (uint64_t packed = 0; packed < result.size(); ++packed) {
            const uint64_t without_pivot = insert_zero_bit(packed, measurement.pivot);
            const bool other_parity =
                (std::popcount(without_pivot & measurement.z_without_pivot) & 1U) != 0;
            const bool pivot_value = branch != other_parity;
            const uint64_t source =
                without_pivot | (static_cast<uint64_t>(pivot_value) << measurement.pivot);
            result[packed] = projected.normalized[source];
        }
        return result;
    }

    for (uint64_t packed = 0; packed < result.size(); ++packed) {
        const uint64_t source0 = insert_zero_bit(packed, measurement.pivot);
        result[packed] = projected.normalized[source0] / kInvSqrt2;
    }
    return result;
}

std::vector<uint32_t> valid_measurement_pivots(uint64_t x, uint64_t z, uint32_t active_width) {
    const uint64_t support = x != 0 ? x : z;
    std::vector<uint32_t> result;
    for (uint32_t q = 0; q < active_width; ++q) {
        if ((support & (uint64_t{1} << q)) != 0) {
            result.push_back(q);
        }
    }
    return result;
}

}  // namespace

TEST_CASE("Sampling SoA state owns stable aligned coefficient and scratch planes") {
    const std::complex<double> initial_scalar{0.6, 0.8};
    SoaState state(4, 2, initial_scalar);

    REQUIRE(state.active_width() == 2);
    REQUIRE(state.max_active_width() == 4);
    REQUIRE(state.capacity() == 16);
    REQUIRE(state.size() == 4);
    REQUIRE(state.global_scalar() == initial_scalar);
    REQUIRE(state.real_data()[0] == 1.0);
    for (uint64_t i = 1; i < state.size(); ++i) {
        REQUIRE(state.real_data()[i] == 0.0);
        REQUIRE(state.imag_data()[i] == 0.0);
    }

    REQUIRE(reinterpret_cast<uintptr_t>(state.real_data()) % 64 == 0);
    REQUIRE(reinterpret_cast<uintptr_t>(state.real_data()) %
                clifft::PageAlignedAllocation::kBaseAlignment ==
            0);
    REQUIRE(reinterpret_cast<uintptr_t>(state.imag_data()) % 64 == 0);
    REQUIRE(reinterpret_cast<uintptr_t>(state.scratch_real_data()) % 64 == 0);
    REQUIRE(reinterpret_cast<uintptr_t>(state.scratch_imag_data()) % 64 == 0);

    double* const real = state.real_data();
    double* const imag = state.imag_data();
    double* const scratch_real = state.scratch_real_data();
    double* const scratch_imag = state.scratch_imag_data();
    state.set_active_width(4);
    state.real_data()[3] = 2.0;
    state.imag_data()[2] = -3.0;
    state.set_global_scalar({-1.0, 0.0});
    state.reset();

    REQUIRE(state.active_width() == 2);
    REQUIRE(state.global_scalar() == initial_scalar);
    REQUIRE(state.real_data() == real);
    REQUIRE(state.imag_data() == imag);
    REQUIRE(state.scratch_real_data() == scratch_real);
    REQUIRE(state.scratch_imag_data() == scratch_imag);
    REQUIRE(state.real_data()[0] == 1.0);
    for (uint64_t i = 1; i < state.size(); ++i) {
        REQUIRE(state.real_data()[i] == 0.0);
        REQUIRE(state.imag_data()[i] == 0.0);
    }

    SoaState moved(std::move(state));
    REQUIRE(moved.real_data() == real);
    REQUIRE(moved.imag_data() == imag);
    SoaState assigned(0);
    assigned = std::move(moved);
    REQUIRE(assigned.real_data() == real);
    REQUIRE(assigned.imag_data() == imag);
}

TEST_CASE("Sampling SoA rotations match the existing dense matrix oracle") {
    static constexpr double kAngles[] = {-0.75, -0.25, 0.0, 0.3};
    for (uint32_t active_width = 0; active_width <= 4; ++active_width) {
        const uint64_t mask_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 0; x < mask_limit; ++x) {
            for (uint64_t z = 0; z < mask_limit; ++z) {
                for (bool sign : {false, true}) {
                    for (double half_turns : kAngles) {
                        CAPTURE(active_width, x, z, sign, half_turns);
                        const std::complex<double> scalar{0.3, 0.4};
                        SoaState state(active_width, active_width, scalar);
                        load_state(state, input);
                        double* const real = state.real_data();
                        double* const imag = state.imag_data();

                        apply_rotation(state, prepare_rotation({x, z}, active_width, half_turns),
                                       sign);

                        std::vector<std::complex<double>> expected =
                            balanced_rotation(input, x, z, sign, half_turns, active_width);
                        for (std::complex<double>& value : expected) {
                            value *= scalar;
                        }
                        require_vectors_close(physical_coefficients(state), expected);
                        REQUIRE(state.real_data() == real);
                        REQUIRE(state.imag_data() == imag);
                        REQUIRE(state.active_width() == active_width);
                    }
                }
            }
        }
    }
}

TEST_CASE("Sampling SoA promotion matches a new-axis dense rotation") {
    static constexpr double kAngles[] = {-0.5, 0.0, 0.25, 0.7};
    for (uint32_t active_width = 0; active_width <= 4; ++active_width) {
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (bool sign : {false, true}) {
            for (double half_turns : kAngles) {
                CAPTURE(active_width, sign, half_turns);
                const std::complex<double> scalar{0.3, 0.4};
                SoaState state(active_width + 1, active_width, scalar);
                load_state(state, input);
                double* const real = state.real_data();
                double* const imag = state.imag_data();

                apply_promotion(state, prepare_promotion(half_turns), sign);

                std::vector<std::complex<double>> expanded(input.size() * 2, {0.0, 0.0});
                std::copy(input.begin(), input.end(), expanded.begin());
                std::vector<std::complex<double>> expected = balanced_rotation(
                    expanded, uint64_t{1} << active_width, 0, sign, half_turns, active_width + 1);
                for (std::complex<double>& value : expected) {
                    value *= scalar;
                }
                require_vectors_close(physical_coefficients(state), expected);
                REQUIRE(state.real_data() == real);
                REQUIRE(state.imag_data() == imag);
                REQUIRE(state.active_width() == active_width + 1);
            }
        }
    }
}

TEST_CASE("Sampling SoA measurements match dense projectors for every small Pauli") {
    for (uint32_t active_width = 1; active_width <= 4; ++active_width) {
        const uint64_t mask_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 0; x < mask_limit; ++x) {
            for (uint64_t z = 0; z < mask_limit; ++z) {
                if (x == 0 && z == 0) {
                    continue;
                }
                for (uint32_t pivot : valid_measurement_pivots(x, z, active_width)) {
                    CAPTURE(active_width, x, z, pivot);
                    const PreparedMeasurement measurement =
                        prepare_measurement({x, z}, active_width, pivot);
                    const std::complex<double> scalar{0.3, 0.4};
                    SoaState probability_state(active_width, active_width, scalar);
                    load_state(probability_state, input);
                    const MeasurementProbabilities probabilities =
                        measurement_probabilities(probability_state, measurement);
                    const ProjectedBranch expected_zero =
                        dense_project(input, x, z, false, active_width);
                    const ProjectedBranch expected_one =
                        dense_project(input, x, z, true, active_width);
                    REQUIRE_THAT(probabilities.zero,
                                 Catch::Matchers::WithinAbs(expected_zero.probability, kTolerance));
                    REQUIRE_THAT(probabilities.one,
                                 Catch::Matchers::WithinAbs(expected_one.probability, kTolerance));
                    REQUIRE_THAT(probabilities.total(),
                                 Catch::Matchers::WithinAbs(1.0, kTolerance));

                    for (bool branch : {false, true}) {
                        const ProjectedBranch& expected = branch ? expected_one : expected_zero;
                        REQUIRE(expected.probability > 1e-12);
                        SoaState state(active_width, active_width, scalar);
                        load_state(state, input);
                        double* const real = state.real_data();
                        double* const imag = state.imag_data();
                        double* const scratch_real = state.scratch_real_data();
                        double* const scratch_imag = state.scratch_imag_data();

                        collapse_measurement(state, measurement, branch,
                                             probabilities.for_branch(branch));

                        std::vector<std::complex<double>> expanded =
                            expand_compacted(coefficients(state), measurement, branch);
                        for (std::complex<double>& value : expanded) {
                            value *= scalar;
                        }
                        std::vector<std::complex<double>> expected_physical = expected.normalized;
                        for (std::complex<double>& value : expected_physical) {
                            value *= scalar;
                        }
                        require_vectors_close(expanded, expected_physical);
                        REQUIRE(state.real_data() == real);
                        REQUIRE(state.imag_data() == imag);
                        REQUIRE(state.scratch_real_data() == scratch_real);
                        REQUIRE(state.scratch_imag_data() == scratch_imag);
                        REQUIRE(state.active_width() == active_width - 1);
                        REQUIRE(state.global_scalar() == scalar);
                    }
                }
            }
        }
    }
}

TEST_CASE("Sampling SoA kernels compose across collapse and promotion") {
    constexpr uint32_t kInitialWidth = 2;
    constexpr uint32_t kExpandedWidth = kInitialWidth + 1;
    const std::complex<double> scalar{0.3, 0.4};
    SoaState state(kExpandedWidth, kInitialWidth, scalar);
    std::vector<std::complex<double>> expected = deterministic_state(kInitialWidth);
    load_state(state, expected);
    double* const real = state.real_data();
    double* const imag = state.imag_data();
    double* const scratch_real = state.scratch_real_data();
    double* const scratch_imag = state.scratch_imag_data();

    constexpr double kFirstPromotionAngle = 0.35;
    apply_promotion(state, prepare_promotion(kFirstPromotionAngle), true);
    std::vector<std::complex<double>> expanded(expected.size() * 2, {0.0, 0.0});
    std::copy(expected.begin(), expected.end(), expanded.begin());
    expected = balanced_rotation(expanded, uint64_t{1} << kInitialWidth, 0, true,
                                 kFirstPromotionAngle, kExpandedWidth);

    constexpr ActivePauli kRotation{0b101, 0b110};
    constexpr double kRotationAngle = -0.3;
    apply_rotation(state, prepare_rotation(kRotation, kExpandedWidth, kRotationAngle), true);
    expected =
        balanced_rotation(expected, kRotation.x, kRotation.z, true, kRotationAngle, kExpandedWidth);

    constexpr ActivePauli kMeasurementPauli{0b110, 0b101};
    constexpr uint32_t kPivot = 2;
    constexpr bool kBranch = true;
    const PreparedMeasurement measurement =
        prepare_measurement(kMeasurementPauli, kExpandedWidth, kPivot);
    const MeasurementProbabilities probabilities = measurement_probabilities(state, measurement);
    const ProjectedBranch projected =
        dense_project(expected, kMeasurementPauli.x, kMeasurementPauli.z, kBranch, kExpandedWidth);
    REQUIRE(projected.probability > 1e-12);
    REQUIRE_THAT(probabilities.for_branch(kBranch),
                 Catch::Matchers::WithinAbs(projected.probability, kTolerance));

    collapse_measurement(state, measurement, kBranch, probabilities.for_branch(kBranch));
    expected = compact_projected(projected, measurement, kBranch);
    REQUIRE(state.active_width() == kInitialWidth);

    double stale_tail_norm = 0.0;
    for (uint64_t i = state.size(); i < 2 * state.size(); ++i) {
        stale_tail_norm +=
            std::norm(std::complex<double>{state.real_data()[i], state.imag_data()[i]});
    }
    REQUIRE(stale_tail_norm > 1e-12);

    constexpr double kSecondPromotionAngle = -0.45;
    apply_promotion(state, prepare_promotion(kSecondPromotionAngle), false);
    expanded.assign(expected.size() * 2, {0.0, 0.0});
    std::copy(expected.begin(), expected.end(), expanded.begin());
    expected = balanced_rotation(expanded, uint64_t{1} << kInitialWidth, 0, false,
                                 kSecondPromotionAngle, kExpandedWidth);
    for (std::complex<double>& value : expected) {
        value *= scalar;
    }

    require_vectors_close(physical_coefficients(state), expected);
    REQUIRE(state.real_data() == real);
    REQUIRE(state.imag_data() == imag);
    REQUIRE(state.scratch_real_data() == scratch_real);
    REQUIRE(state.scratch_imag_data() == scratch_imag);
    REQUIRE(state.global_scalar() == scalar);
}

TEST_CASE("Sampling SoA preparation and transitions reject malformed inputs") {
    REQUIRE(prepare_rotation({0b101, 0}, 3, 0.25).pauli.pair_selector == 0b100);
    REQUIRE_THROWS_AS(SoaState(clifft::kDenseActiveWidthLimit), std::invalid_argument);
    REQUIRE_THROWS_AS(SoaState(1, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(SoaState(1, 0, {clifft::test::opaque_nan(), 0.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_rotation({2, 0}, 1, 0.25), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_rotation({1, 0}, 1, clifft::test::opaque_nan()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_promotion(clifft::test::opaque_nan()), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({}, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({1, 0}, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({1, 0}, 2, 1), std::invalid_argument);

    SoaState state(1, 1);
    const auto wrong_width = prepare_rotation({1, 0}, 2, 0.25);
    REQUIRE_THROWS_AS(apply_rotation(state, wrong_width, false), std::invalid_argument);
    REQUIRE_THROWS_AS(apply_promotion(state, prepare_promotion(0.25), false), std::out_of_range);

    const PreparedMeasurement measurement = prepare_measurement({1, 0}, 1, 0);
    REQUIRE_THROWS_AS(collapse_measurement(state, measurement, false, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(collapse_measurement(state, measurement, false, clifft::test::opaque_nan()),
                      std::invalid_argument);
}
