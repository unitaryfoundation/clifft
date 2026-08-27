#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/kernels.h"
#include "clifft/util/runtime_isa.h"

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

using clifft::internal::RuntimeIsa;
using clifft::sampling::activate_zero_coordinate;
using clifft::sampling::active_measurement_probabilities_neon;
using clifft::sampling::ActiveMeasurementKernel;
using clifft::sampling::ActivePauli;
using clifft::sampling::apply_instrument_no_fire;
using clifft::sampling::apply_new_x_instrument_no_fire;
using clifft::sampling::apply_promotion;
using clifft::sampling::apply_rotation;
using clifft::sampling::collapse_active_measurement_neon;
using clifft::sampling::collapse_instrument_source;
using clifft::sampling::collapse_measurement;
using clifft::sampling::collapse_new_x_instrument_source;
using clifft::sampling::DirectRotationKernel;
using clifft::sampling::ExecutorBackend;
using clifft::sampling::expectation_value;
using clifft::sampling::measurement_probabilities;
using clifft::sampling::MeasurementProbabilities;
using clifft::sampling::NewXInstrumentKernel;
using clifft::sampling::prepare_measurement;
using clifft::sampling::prepare_pauli;
using clifft::sampling::prepare_promotion;
using clifft::sampling::prepare_rotation;
using clifft::sampling::PreparedMeasurement;
using clifft::sampling::resolve_active_measurement_kernel;
using clifft::sampling::resolve_direct_rotation_kernel;
using clifft::sampling::resolve_executor_backend;
using clifft::sampling::resolve_new_x_instrument_kernel;
using clifft::sampling::State;
using clifft::test::check_complex;
using clifft::test::dense_axis_rotation;
using clifft::test::dense_matvec;
using clifft::test::DenseMatrix;

#if defined(CLIFFT_TESTS_HAVE_X86_KERNELS)
using clifft::sampling::active_measurement_probabilities_avx2;
using clifft::sampling::active_measurement_probabilities_avx512;
using clifft::sampling::apply_new_x_instrument_no_fire_avx2;
using clifft::sampling::collapse_active_measurement_avx2;
using clifft::sampling::collapse_active_measurement_avx512;
#endif

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

void load_state(State& state, const std::vector<std::complex<double>>& values) {
    REQUIRE(values.size() == state.size());
    for (uint64_t i = 0; i < state.size(); ++i) {
        state.real_data()[i] = values[i].real();
        state.imag_data()[i] = values[i].imag();
    }
}

std::vector<std::complex<double>> coefficients(const State& state) {
    std::vector<std::complex<double>> result(state.size());
    for (uint64_t i = 0; i < state.size(); ++i) {
        result[i] = {state.real_data()[i], state.imag_data()[i]};
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
    return dense_matvec(matrix, input);
}

void require_vectors_close(const std::vector<std::complex<double>>& actual,
                           const std::vector<std::complex<double>>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        check_complex(actual[i], expected[i], kTolerance);
    }
}

// Keep the projector and compaction oracle independent of the kernel's index
// helpers so an indexing defect cannot make both actual and expected agree.
uint64_t expand_index_without_pivot(uint64_t packed, uint32_t pivot) {
    const uint64_t lower_mask = (uint64_t{1} << pivot) - 1;
    return (packed & lower_mask) | ((packed >> pivot) << (pivot + 1));
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
    const std::vector<std::complex<double>> applied = dense_matvec(pauli, input);
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
            const uint64_t without_pivot = expand_index_without_pivot(packed, measurement.pivot);
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
        const uint64_t source0 = expand_index_without_pivot(packed, measurement.pivot);
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
            const uint64_t without_pivot = expand_index_without_pivot(packed, measurement.pivot);
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
        const uint64_t source0 = expand_index_without_pivot(packed, measurement.pivot);
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

TEST_CASE("Sampling kernel state owns stable aligned storage") {
    State state(4, 2);

    REQUIRE(state.active_width() == 2);
    REQUIRE(state.max_active_width() == 4);
    REQUIRE(state.capacity() == 16);
    REQUIRE(state.size() == 4);
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
    state.reset();

    REQUIRE(state.active_width() == 2);
    REQUIRE(state.real_data() == real);
    REQUIRE(state.imag_data() == imag);
    REQUIRE(state.scratch_real_data() == scratch_real);
    REQUIRE(state.scratch_imag_data() == scratch_imag);
    REQUIRE(state.real_data()[0] == 1.0);
    for (uint64_t i = 1; i < state.size(); ++i) {
        REQUIRE(state.real_data()[i] == 0.0);
        REQUIRE(state.imag_data()[i] == 0.0);
    }

    State moved(std::move(state));
    REQUIRE(moved.real_data() == real);
    REQUIRE(moved.imag_data() == imag);
    State assigned(0);
    assigned = std::move(moved);
    REQUIRE(assigned.real_data() == real);
    REQUIRE(assigned.imag_data() == imag);
}

TEST_CASE("Sampling state grows only at a continuation boundary and preserves live data") {
    State state(1, 1);
    const std::vector<std::complex<double>> input = deterministic_state(1);
    load_state(state, input);

    state.ensure_capacity(3);

    REQUIRE(state.active_width() == 1);
    REQUIRE(state.initial_active_width() == 1);
    REQUIRE(state.max_active_width() == 3);
    REQUIRE(state.capacity() == 8);
    require_vectors_close(coefficients(state), input);
    REQUIRE(reinterpret_cast<uintptr_t>(state.real_data()) %
                clifft::PageAlignedAllocation::kBaseAlignment ==
            0);

    state.reset();
    REQUIRE(state.active_width() == 1);
    REQUIRE(state.real_data()[0] == 1.0);
    REQUIRE(state.real_data()[1] == 0.0);
}

TEST_CASE("Sampling kernels rotations match the existing dense matrix oracle") {
    static constexpr double kAngles[] = {-0.75, -0.25, 0.0, 0.3};
    for (uint32_t active_width = 1; active_width <= 4; ++active_width) {
        const uint64_t mask_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 0; x < mask_limit; ++x) {
            for (uint64_t z = 0; z < mask_limit; ++z) {
                if (x == 0 && z == 0) {
                    continue;
                }
                for (bool sign : {false, true}) {
                    for (double half_turns : kAngles) {
                        CAPTURE(active_width, x, z, sign, half_turns);
                        State state(active_width, active_width);
                        load_state(state, input);
                        double* const real = state.real_data();
                        double* const imag = state.imag_data();

                        apply_rotation(state, prepare_rotation({x, z}, active_width, half_turns),
                                       sign);

                        std::vector<std::complex<double>> expected =
                            balanced_rotation(input, x, z, sign, half_turns, active_width);
                        require_vectors_close(coefficients(state), expected);
                        REQUIRE(state.real_data() == real);
                        REQUIRE(state.imag_data() == imag);
                        REQUIRE(state.active_width() == active_width);
                    }
                }
            }
        }
    }
}

TEST_CASE("Direct rotation SIMD selection preserves scalar boundaries") {
    const auto select = [](ActivePauli pauli, uint32_t active_width,
                           ExecutorBackend backend = ExecutorBackend::Avx512) {
        return resolve_direct_rotation_kernel(prepare_rotation(pauli, active_width, 0.3), backend);
    };

    REQUIRE(select({0, 0b11}, 2) == DirectRotationKernel::Scalar);
    REQUIRE(select({0, 0b101}, 3) == DirectRotationKernel::Diagonal);
    REQUIRE(select({0b10, 0b01}, 2) == DirectRotationKernel::Scalar);
    REQUIRE(select({0b001, 0b110}, 3) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b010, 0b101}, 3) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b100, 0b011}, 3) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b1000, 0b0111}, 4) == DirectRotationKernel::HighPivot);
    REQUIRE(select({0b10000, 0b01111}, 5) == DirectRotationKernel::Scalar);
    REQUIRE(select({0b100000, 0b011111}, 6) == DirectRotationKernel::HighPivot);

    REQUIRE(select({0, 0b1}, 1, ExecutorBackend::Avx2) == DirectRotationKernel::Scalar);
    REQUIRE(select({0, 0b11}, 2, ExecutorBackend::Avx2) == DirectRotationKernel::Diagonal);
    REQUIRE(select({0b1, 0b0}, 1, ExecutorBackend::Avx2) == DirectRotationKernel::Scalar);
    REQUIRE(select({0b01, 0b10}, 2, ExecutorBackend::Avx2) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b100, 0b011}, 3, ExecutorBackend::Avx2) == DirectRotationKernel::HighPivot);
    REQUIRE(select({0b10000, 0b01111}, 5, ExecutorBackend::Avx2) ==
            DirectRotationKernel::HighPivot);
    REQUIRE(select({0b01, 0b10}, 2, ExecutorBackend::Scalar) == DirectRotationKernel::Scalar);

    REQUIRE(select({0, 0b11}, 2, ExecutorBackend::Neon) == DirectRotationKernel::Scalar);
    REQUIRE(select({0, 0b101}, 3, ExecutorBackend::Neon) == DirectRotationKernel::Diagonal);
    REQUIRE(select({0b100, 0b011}, 3, ExecutorBackend::Neon) == DirectRotationKernel::Scalar);
    REQUIRE(select({0b1000, 0b0111}, 4, ExecutorBackend::Neon) == DirectRotationKernel::HighPivot);
    REQUIRE(select({0b001, 0b110}, 3, ExecutorBackend::Neon) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b0001, 0b1110}, 4, ExecutorBackend::Neon) == DirectRotationKernel::LanePaired);
    REQUIRE(select({0b000001, 0b111110}, 6, ExecutorBackend::Neon) ==
            DirectRotationKernel::LanePaired);
}

TEST_CASE("Sampling executor backend follows the resolved process ISA") {
    REQUIRE(resolve_executor_backend(RuntimeIsa::Scalar) == ExecutorBackend::Scalar);
    REQUIRE(resolve_executor_backend(RuntimeIsa::Neon) == ExecutorBackend::Neon);
    REQUIRE(resolve_executor_backend(RuntimeIsa::Avx2) == ExecutorBackend::Avx2);
    REQUIRE(resolve_executor_backend(RuntimeIsa::Avx512) == ExecutorBackend::Avx512);
    REQUIRE_THROWS(resolve_executor_backend(RuntimeIsa::TrapUnknown));
}

TEST_CASE("Active measurement SIMD selection preserves scalar boundaries") {
    const auto select = [](ActivePauli pauli, uint32_t active_width, uint32_t pivot,
                           ExecutorBackend backend = ExecutorBackend::Avx512) {
        return resolve_active_measurement_kernel(prepare_measurement(pauli, active_width, pivot),
                                                 backend);
    };

    REQUIRE(select({0, 0b101}, 3, 0) == ActiveMeasurementKernel::Scalar);
    REQUIRE(select({0b01, 0b10}, 2, 0) == ActiveMeasurementKernel::Scalar);
    REQUIRE(select({0b001, 0b110}, 3, 0) == ActiveMeasurementKernel::Scalar);
    REQUIRE(select({0b001, 0b1110}, 4, 0) == ActiveMeasurementKernel::LanePaired);
    REQUIRE(select({0b111, 0b101000}, 6, 2) == ActiveMeasurementKernel::LanePaired);
    REQUIRE(select({0b1000, 0b0111}, 4, 3) == ActiveMeasurementKernel::HighPivot);
    REQUIRE(select({0b01, 0b10}, 2, 0, ExecutorBackend::Avx2) ==
            ActiveMeasurementKernel::LanePaired);
    REQUIRE(select({0b01, 0b110}, 3, 0, ExecutorBackend::Avx2) ==
            ActiveMeasurementKernel::LanePaired);
    REQUIRE(select({0b10, 0b101}, 3, 1, ExecutorBackend::Avx2) ==
            ActiveMeasurementKernel::LanePaired);
    REQUIRE(select({0b100, 0b011}, 3, 2, ExecutorBackend::Avx2) ==
            ActiveMeasurementKernel::HighPivot);
    REQUIRE(select({0b01, 0b110}, 3, 0, ExecutorBackend::Scalar) ==
            ActiveMeasurementKernel::Scalar);
    REQUIRE(select({0, 0b10000}, 5, 4, ExecutorBackend::Neon) == ActiveMeasurementKernel::Scalar);
    REQUIRE(select({0, 0b100000}, 6, 5, ExecutorBackend::Neon) ==
            ActiveMeasurementKernel::Diagonal);
    REQUIRE(select({0b100000, 0b011111}, 6, 5, ExecutorBackend::Neon) ==
            ActiveMeasurementKernel::Scalar);
}

TEST_CASE("New X instrument SIMD selection preserves scalar boundaries") {
    REQUIRE(resolve_new_x_instrument_kernel(1, ExecutorBackend::Avx2) ==
            NewXInstrumentKernel::Scalar);
    REQUIRE(resolve_new_x_instrument_kernel(2, ExecutorBackend::Avx2) ==
            NewXInstrumentKernel::Vectorized);
    REQUIRE(resolve_new_x_instrument_kernel(8, ExecutorBackend::Avx2) ==
            NewXInstrumentKernel::Vectorized);
    REQUIRE(resolve_new_x_instrument_kernel(8, ExecutorBackend::Scalar) ==
            NewXInstrumentKernel::Scalar);
    REQUIRE(resolve_new_x_instrument_kernel(8, ExecutorBackend::Avx512) ==
            NewXInstrumentKernel::Vectorized);
}

TEST_CASE("Sampling kernel expectation values match the existing dense matrix oracle") {
    for (uint32_t active_width = 0; active_width <= 4; ++active_width) {
        const uint64_t mask_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 0; x < mask_limit; ++x) {
            for (uint64_t z = 0; z < mask_limit; ++z) {
                CAPTURE(active_width, x, z);
                State state(active_width, active_width);
                load_state(state, input);

                const std::vector<std::complex<double>> applied =
                    dense_matvec(dense_axis_rotation(x, z, false, 1.0, active_width), input);
                std::complex<double> expected{0.0, 0.0};
                for (size_t i = 0; i < input.size(); ++i) {
                    expected += std::conj(input[i]) * applied[i];
                }

                REQUIRE_THAT(expectation_value(state, prepare_pauli({x, z}, active_width)),
                             Catch::Matchers::WithinAbs(expected.real(), kTolerance));
                REQUIRE_THAT(expected.imag(), Catch::Matchers::WithinAbs(0.0, kTolerance));
            }
        }
    }
}

TEST_CASE("Sampling kernels promotion matches a new-axis dense rotation") {
    static constexpr double kAngles[] = {-0.5, 0.0, 0.25, 0.7};
    for (uint32_t active_width = 0; active_width <= 4; ++active_width) {
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (bool sign : {false, true}) {
            for (double half_turns : kAngles) {
                CAPTURE(active_width, sign, half_turns);
                State state(active_width + 1, active_width);
                load_state(state, input);
                double* const real = state.real_data();
                double* const imag = state.imag_data();

                apply_promotion(state, prepare_promotion(half_turns), sign);

                std::vector<std::complex<double>> expanded(input.size() * 2, {0.0, 0.0});
                std::copy(input.begin(), input.end(), expanded.begin());
                std::vector<std::complex<double>> expected = balanced_rotation(
                    expanded, uint64_t{1} << active_width, 0, sign, half_turns, active_width + 1);
                require_vectors_close(coefficients(state), expected);
                REQUIRE(state.real_data() == real);
                REQUIRE(state.imag_data() == imag);
                REQUIRE(state.active_width() == active_width + 1);
            }
        }
    }
}

TEST_CASE("Sampling kernels measurements match dense projectors for every small Pauli") {
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
                    State probability_state(active_width, active_width);
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
                        State state(active_width, active_width);
                        load_state(state, input);
                        double* const real = state.real_data();
                        double* const imag = state.imag_data();
                        double* const scratch_real = state.scratch_real_data();
                        double* const scratch_imag = state.scratch_imag_data();

                        collapse_measurement(state, measurement, branch,
                                             probabilities.for_branch(branch));

                        std::vector<std::complex<double>> expanded =
                            expand_compacted(coefficients(state), measurement, branch);
                        require_vectors_close(expanded, expected.normalized);
                        REQUIRE(state.real_data() == real);
                        REQUIRE(state.imag_data() == imag);
                        REQUIRE(state.scratch_real_data() == scratch_real);
                        REQUIRE(state.scratch_imag_data() == scratch_imag);
                        REQUIRE(state.active_width() == active_width - 1);
                    }
                }
            }
        }
    }
}

#if defined(CLIFFT_TESTS_HAVE_X86_KERNELS)
TEST_CASE("Active measurement SIMD matches scalar Pauli compaction") {
    const RuntimeIsa runtime_isa = clifft::internal::runtime_isa();
    if (runtime_isa != RuntimeIsa::Avx2 && runtime_isa != RuntimeIsa::Avx512) {
        return;
    }
    const ExecutorBackend backend =
        runtime_isa == RuntimeIsa::Avx512 ? ExecutorBackend::Avx512 : ExecutorBackend::Avx2;

    const uint64_t vector_lanes = runtime_isa == RuntimeIsa::Avx512 ? 8 : 4;
    const uint32_t lane_index_bits = runtime_isa == RuntimeIsa::Avx512 ? 3 : 2;
    const uint32_t min_profitable_width = runtime_isa == RuntimeIsa::Avx512 ? 4 : 2;
    for (uint32_t active_width = min_profitable_width; active_width <= 6; ++active_width) {
        const uint64_t z_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 1; x < z_limit; ++x) {
            for (uint64_t z = 0; z < z_limit; ++z) {
                for (uint32_t pivot : valid_measurement_pivots(x, z, active_width)) {
                    CAPTURE(active_width, x, z, pivot);
                    const PreparedMeasurement measurement =
                        prepare_measurement({x, z}, active_width, pivot);
                    const ActiveMeasurementKernel selected =
                        resolve_active_measurement_kernel(measurement, backend);
                    const uint64_t pivot_bit = uint64_t{1} << pivot;
                    ActiveMeasurementKernel expected_kernel = ActiveMeasurementKernel::Scalar;
                    if (std::bit_floor(x) == pivot_bit && pivot_bit >= vector_lanes) {
                        expected_kernel = ActiveMeasurementKernel::HighPivot;
                    } else if (x < vector_lanes && pivot < lane_index_bits) {
                        expected_kernel = ActiveMeasurementKernel::LanePaired;
                    }
                    REQUIRE(selected == expected_kernel);
                    if (expected_kernel == ActiveMeasurementKernel::Scalar) {
                        continue;
                    }

                    State scalar_probability_state(active_width, active_width);
                    load_state(scalar_probability_state, input);
                    const MeasurementProbabilities expected =
                        measurement_probabilities(scalar_probability_state, measurement);

                    State vector_probability_state(active_width, active_width);
                    load_state(vector_probability_state, input);
                    const MeasurementProbabilities actual =
                        runtime_isa == RuntimeIsa::Avx512
                            ? active_measurement_probabilities_avx512(vector_probability_state,
                                                                      measurement, selected)
                            : active_measurement_probabilities_avx2(vector_probability_state,
                                                                    measurement, selected);
                    REQUIRE_THAT(actual.zero,
                                 Catch::Matchers::WithinAbs(expected.zero, kTolerance));
                    REQUIRE_THAT(actual.one, Catch::Matchers::WithinAbs(expected.one, kTolerance));

                    for (bool branch : {false, true}) {
                        State scalar_state(active_width, active_width);
                        load_state(scalar_state, input);
                        collapse_measurement(scalar_state, measurement, branch,
                                             expected.for_branch(branch));

                        State vector_state(active_width, active_width);
                        load_state(vector_state, input);
                        if (runtime_isa == RuntimeIsa::Avx512) {
                            collapse_active_measurement_avx512(vector_state, measurement, selected,
                                                               branch, actual.for_branch(branch));
                        } else {
                            collapse_active_measurement_avx2(vector_state, measurement, selected,
                                                             branch, actual.for_branch(branch));
                        }

                        require_vectors_close(coefficients(vector_state),
                                              coefficients(scalar_state));
                        REQUIRE(vector_state.active_width() == active_width - 1);
                    }
                }
            }
        }
    }
}

TEST_CASE("Diagonal active measurement SIMD matches scalar compaction") {
    const RuntimeIsa runtime_isa = clifft::internal::runtime_isa();
    if (runtime_isa != RuntimeIsa::Avx2 && runtime_isa != RuntimeIsa::Avx512) {
        return;
    }
    const ExecutorBackend backend =
        runtime_isa == RuntimeIsa::Avx512 ? ExecutorBackend::Avx512 : ExecutorBackend::Avx2;
    const uint32_t min_profitable_width = runtime_isa == RuntimeIsa::Avx512 ? 4 : 2;
    for (uint32_t active_width = min_profitable_width; active_width <= 6; ++active_width) {
        const uint64_t z_limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t z = 1; z < z_limit; ++z) {
            for (uint32_t pivot : valid_measurement_pivots(0, z, active_width)) {
                CAPTURE(active_width, z, pivot);
                const PreparedMeasurement measurement =
                    prepare_measurement({0, z}, active_width, pivot);
                const uint64_t pivot_bit = uint64_t{1} << pivot;
                const bool has_lower_z = (z & (pivot_bit - 1)) != 0;
                const ActiveMeasurementKernel selected =
                    resolve_active_measurement_kernel(measurement, backend);
                REQUIRE(selected == (has_lower_z ? ActiveMeasurementKernel::Scalar
                                                 : ActiveMeasurementKernel::Diagonal));
                if (has_lower_z) {
                    continue;
                }

                State scalar_probability_state(active_width, active_width);
                load_state(scalar_probability_state, input);
                const MeasurementProbabilities expected =
                    measurement_probabilities(scalar_probability_state, measurement);

                State vector_probability_state(active_width, active_width);
                load_state(vector_probability_state, input);
                const MeasurementProbabilities actual =
                    runtime_isa == RuntimeIsa::Avx512
                        ? active_measurement_probabilities_avx512(vector_probability_state,
                                                                  measurement, selected)
                        : active_measurement_probabilities_avx2(vector_probability_state,
                                                                measurement, selected);
                REQUIRE_THAT(actual.zero, Catch::Matchers::WithinAbs(expected.zero, kTolerance));
                REQUIRE_THAT(actual.one, Catch::Matchers::WithinAbs(expected.one, kTolerance));

                for (bool branch : {false, true}) {
                    State scalar_state(active_width, active_width);
                    load_state(scalar_state, input);
                    collapse_measurement(scalar_state, measurement, branch,
                                         expected.for_branch(branch));

                    State vector_state(active_width, active_width);
                    load_state(vector_state, input);
                    if (runtime_isa == RuntimeIsa::Avx512) {
                        collapse_active_measurement_avx512(vector_state, measurement, selected,
                                                           branch, actual.for_branch(branch));
                    } else {
                        collapse_active_measurement_avx2(vector_state, measurement, selected,
                                                         branch, actual.for_branch(branch));
                    }
                    require_vectors_close(coefficients(vector_state), coefficients(scalar_state));
                }
            }
        }
    }
}
#endif

#if defined(CLIFFT_TESTS_HAVE_APPLE_NEON)
TEST_CASE("Apple NEON diagonal measurement matches scalar compaction") {
    constexpr uint32_t kActiveWidth = 6;
    const uint64_t z_limit = uint64_t{1} << kActiveWidth;
    const std::vector<std::complex<double>> input = deterministic_state(kActiveWidth);
    for (uint64_t z = 1; z < z_limit; ++z) {
        for (uint32_t pivot : valid_measurement_pivots(0, z, kActiveWidth)) {
            const uint64_t pivot_bit = uint64_t{1} << pivot;
            if ((z & (pivot_bit - 1)) != 0) {
                continue;
            }
            CAPTURE(z, pivot);
            const PreparedMeasurement measurement =
                prepare_measurement({0, z}, kActiveWidth, pivot);
            State scalar_probability_state(kActiveWidth, kActiveWidth);
            load_state(scalar_probability_state, input);
            const MeasurementProbabilities expected =
                measurement_probabilities(scalar_probability_state, measurement);

            State vector_probability_state(kActiveWidth, kActiveWidth);
            load_state(vector_probability_state, input);
            const MeasurementProbabilities actual = active_measurement_probabilities_neon(
                vector_probability_state, measurement, ActiveMeasurementKernel::Diagonal);
            REQUIRE_THAT(actual.zero, Catch::Matchers::WithinAbs(expected.zero, kTolerance));
            REQUIRE_THAT(actual.one, Catch::Matchers::WithinAbs(expected.one, kTolerance));

            for (bool branch : {false, true}) {
                State scalar_state(kActiveWidth, kActiveWidth);
                load_state(scalar_state, input);
                collapse_measurement(scalar_state, measurement, branch,
                                     expected.for_branch(branch));

                State vector_state(kActiveWidth, kActiveWidth);
                load_state(vector_state, input);
                collapse_active_measurement_neon(vector_state, measurement,
                                                 ActiveMeasurementKernel::Diagonal, branch,
                                                 actual.for_branch(branch));
                require_vectors_close(coefficients(vector_state), coefficients(scalar_state));
            }
        }
    }
}
#endif

TEST_CASE("Sampling instrument kernels match dense projectors without compacting") {
    constexpr double kFactorZero = 0.8;
    constexpr double kFactorOne = 0.3;
    for (uint32_t active_width = 1; active_width <= 3; ++active_width) {
        const uint64_t limit = uint64_t{1} << active_width;
        const std::vector<std::complex<double>> input = deterministic_state(active_width);
        for (uint64_t x = 0; x < limit; ++x) {
            for (uint64_t z = 0; z < limit; ++z) {
                if (x == 0 && z == 0) {
                    continue;
                }
                CAPTURE(active_width, x, z);
                const uint64_t support = x != 0 ? x : z;
                const PreparedMeasurement measurement = prepare_measurement(
                    {x, z}, active_width, static_cast<uint32_t>(std::countr_zero(support)));
                const ProjectedBranch projected_zero =
                    dense_project(input, x, z, false, active_width);
                const ProjectedBranch projected_one =
                    dense_project(input, x, z, true, active_width);

                const double no_fire_probability =
                    kFactorZero * kFactorZero * projected_zero.probability +
                    kFactorOne * kFactorOne * projected_one.probability;
                std::vector<std::complex<double>> filtered(input.size());
                for (size_t i = 0; i < filtered.size(); ++i) {
                    filtered[i] = (kFactorZero * std::sqrt(projected_zero.probability) *
                                       projected_zero.normalized[i] +
                                   kFactorOne * std::sqrt(projected_one.probability) *
                                       projected_one.normalized[i]) /
                                  std::sqrt(no_fire_probability);
                }

                State no_fire(active_width, active_width);
                load_state(no_fire, input);
                apply_instrument_no_fire(no_fire, measurement.pauli, kFactorZero, kFactorOne,
                                         no_fire_probability);
                require_vectors_close(coefficients(no_fire), filtered);
                REQUIRE(no_fire.active_width() == active_width);

                for (bool branch : {false, true}) {
                    const ProjectedBranch& projected = branch ? projected_one : projected_zero;
                    State collapsed(active_width, active_width);
                    load_state(collapsed, input);
                    collapse_instrument_source(collapsed, measurement.pauli, branch,
                                               projected.probability);
                    require_vectors_close(coefficients(collapsed), projected.normalized);
                    REQUIRE(collapsed.active_width() == active_width);
                }
            }
        }
    }
}

TEST_CASE("Sampling instrument activation adds a clean coordinate in existing storage") {
    State state(3, 2);
    const std::vector<std::complex<double>> input = deterministic_state(2);
    load_state(state, input);
    double* const real = state.real_data();
    double* const imag = state.imag_data();

    activate_zero_coordinate(state);

    REQUIRE(state.active_width() == 3);
    REQUIRE(state.real_data() == real);
    REQUIRE(state.imag_data() == imag);
    for (uint64_t i = 0; i < input.size(); ++i) {
        REQUIRE(state.real_data()[i] == input[i].real());
        REQUIRE(state.imag_data()[i] == input[i].imag());
        REQUIRE(state.real_data()[input.size() + i] == 0.0);
        REQUIRE(state.imag_data()[input.size() + i] == 0.0);
    }
}

TEST_CASE("Sampling new X instrument activation matches the generic widened source") {
    for (uint32_t initial_width : {0U, 3U}) {
        CAPTURE(initial_width);
        const uint32_t expanded_width = initial_width + 1;
        const ActivePauli new_x{uint64_t{1} << initial_width, 0};
        const PreparedMeasurement measurement =
            prepare_measurement(new_x, expanded_width, initial_width);
        const std::vector<std::complex<double>> input = deterministic_state(initial_width);

        State population_oracle(expanded_width, initial_width);
        load_state(population_oracle, input);
        activate_zero_coordinate(population_oracle);
        const MeasurementProbabilities expected_populations =
            measurement_probabilities(population_oracle, measurement);

        constexpr MeasurementProbabilities kExactPopulations{0.5, 0.5};
        REQUIRE_THAT(kExactPopulations.zero,
                     Catch::Matchers::WithinAbs(expected_populations.zero, kTolerance));
        REQUIRE_THAT(kExactPopulations.one,
                     Catch::Matchers::WithinAbs(expected_populations.one, kTolerance));

        for (const auto& [factor_zero, factor_one] :
             {std::pair{0.8, 0.3}, std::pair{0.3, 0.8}, std::pair{0.6, 0.6}}) {
            CAPTURE(factor_zero, factor_one);
            const double expected_no_fire_probability =
                factor_zero * factor_zero * expected_populations.zero +
                factor_one * factor_one * expected_populations.one;
            const double actual_no_fire_probability =
                factor_zero * factor_zero * kExactPopulations.zero +
                factor_one * factor_one * kExactPopulations.one;

            State no_fire_oracle(expanded_width, initial_width);
            load_state(no_fire_oracle, input);
            activate_zero_coordinate(no_fire_oracle);
            apply_instrument_no_fire(no_fire_oracle, measurement.pauli, factor_zero, factor_one,
                                     expected_no_fire_probability);

            State no_fire_actual(expanded_width, initial_width);
            load_state(no_fire_actual, input);
            apply_new_x_instrument_no_fire(no_fire_actual, factor_zero, factor_one,
                                           actual_no_fire_probability);
            require_vectors_close(coefficients(no_fire_actual), coefficients(no_fire_oracle));
            REQUIRE(no_fire_actual.active_width() == expanded_width);
            if (factor_zero == factor_one) {
                for (uint64_t basis = 0; basis < input.size(); ++basis) {
                    REQUIRE(no_fire_actual.real_data()[input.size() + basis] == 0.0);
                    REQUIRE(no_fire_actual.imag_data()[input.size() + basis] == 0.0);
                }
            }
        }

        for (bool branch : {false, true}) {
            CAPTURE(branch);
            State collapse_oracle(expanded_width, initial_width);
            load_state(collapse_oracle, input);
            activate_zero_coordinate(collapse_oracle);
            collapse_instrument_source(collapse_oracle, measurement.pauli, branch,
                                       expected_populations.for_branch(branch));

            State collapse_actual(expanded_width, initial_width);
            load_state(collapse_actual, input);
            collapse_new_x_instrument_source(collapse_actual, branch,
                                             kExactPopulations.for_branch(branch));
            require_vectors_close(coefficients(collapse_actual), coefficients(collapse_oracle));
            REQUIRE(collapse_actual.active_width() == expanded_width);
        }
    }
}

#if defined(CLIFFT_TESTS_HAVE_X86_KERNELS)
TEST_CASE("Sampling AVX2 new X instrument activation matches scalar") {
    const RuntimeIsa runtime_isa = clifft::internal::runtime_isa();
    if (runtime_isa != RuntimeIsa::Avx2 && runtime_isa != RuntimeIsa::Avx512) {
        return;
    }

    for (uint32_t initial_width = 2; initial_width <= 8; ++initial_width) {
        CAPTURE(initial_width);
        const std::vector<std::complex<double>> input = deterministic_state(initial_width);
        for (const auto& [factor_zero, factor_one] :
             {std::pair{0.8, 0.3}, std::pair{0.3, 0.8}, std::pair{0.6, 0.6}, std::pair{1.0, 0.0},
              std::pair{0.0, 1.0}, std::pair{1.0, 1.0}}) {
            CAPTURE(factor_zero, factor_one);
            const double no_fire_probability =
                0.5 * factor_zero * factor_zero + 0.5 * factor_one * factor_one;

            State expected(initial_width + 1, initial_width);
            load_state(expected, input);
            apply_new_x_instrument_no_fire(expected, factor_zero, factor_one, no_fire_probability);

            State actual(initial_width + 1, initial_width);
            load_state(actual, input);
            apply_new_x_instrument_no_fire_avx2(actual, factor_zero, factor_one,
                                                no_fire_probability);

            require_vectors_close(coefficients(actual), coefficients(expected));
            REQUIRE(actual.active_width() == initial_width + 1);
        }
    }
}
#endif

TEST_CASE("Sampling kernels compose across collapse and promotion") {
    constexpr uint32_t kInitialWidth = 2;
    constexpr uint32_t kExpandedWidth = kInitialWidth + 1;
    State state(kExpandedWidth, kInitialWidth);
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
    require_vectors_close(coefficients(state), expected);
    REQUIRE(state.real_data() == real);
    REQUIRE(state.imag_data() == imag);
    REQUIRE(state.scratch_real_data() == scratch_real);
    REQUIRE(state.scratch_imag_data() == scratch_imag);
}

TEST_CASE("Sampling kernel preparation precomputes non-diagonal pairing metadata") {
    REQUIRE(prepare_rotation({0b101, 0}, 3, 0.25).pauli.pairing_bit == 0b100);
}

TEST_CASE("Sampling kernel preparation rejects malformed inputs") {
    REQUIRE_THROWS_AS(State(clifft::kDenseActiveWidthLimit), std::invalid_argument);
    REQUIRE_THROWS_AS(State(1, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_rotation({}, 1, 0.25), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_rotation({2, 0}, 1, 0.25), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_rotation({1, 0}, 1, clifft::test::opaque_nan()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_promotion(clifft::test::opaque_nan()), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({}, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({1, 0}, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(prepare_measurement({1, 0}, 2, 1), std::invalid_argument);
}
