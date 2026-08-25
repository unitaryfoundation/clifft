#include "clifft/sampling/interleaved_batch_kernels.h"
#include "clifft/sampling/kernels.h"
#include "clifft/util/numeric.h"
#include "clifft/util/page_allocation.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

using clifft::sampling::ActivePauli;
using clifft::sampling::apply_fused_rotation;
using clifft::sampling::apply_interleaved_dynamic_fused_rotation;
using clifft::sampling::apply_interleaved_fused_rotation;
using clifft::sampling::apply_interleaved_promotion;
using clifft::sampling::apply_interleaved_rotation;
using clifft::sampling::apply_promotion;
using clifft::sampling::apply_rotation;
using clifft::sampling::InterleavedBatchState;
using clifft::sampling::prepare_interleaved_rotation_sines;
using clifft::sampling::prepare_promotion;
using clifft::sampling::prepare_rotation;
using clifft::sampling::State;
using clifft::test::check_complex;

namespace {

constexpr double kTolerance = 2e-11;

std::vector<std::complex<double>> lane_state(uint32_t active_width, uint32_t lane) {
    const uint64_t size = uint64_t{1} << active_width;
    std::vector<std::complex<double>> result(size);
    double norm = 0.0;
    for (uint64_t basis = 0; basis < size; ++basis) {
        const double real =
            1.0 + static_cast<double>((3 * basis + 5 * lane + 1) % 17);
        const double imag =
            static_cast<double>((7 * basis + 2 * lane + 3) % 19) - 9.0;
        result[basis] = {real, imag};
        norm += std::norm(result[basis]);
    }
    const double inv_norm = 1.0 / std::sqrt(norm);
    for (std::complex<double>& value : result) {
        value *= inv_norm;
    }
    return result;
}

void load_state(State& state, const std::vector<std::complex<double>>& values) {
    REQUIRE(values.size() == state.size());
    for (uint64_t basis = 0; basis < state.size(); ++basis) {
        state.real_data()[basis] = values[basis].real();
        state.imag_data()[basis] = values[basis].imag();
    }
}

void load_batch(InterleavedBatchState& state,
                const std::vector<std::vector<std::complex<double>>>& values) {
    REQUIRE(values.size() == state.active_lanes());
    for (uint32_t lane = 0; lane < state.active_lanes(); ++lane) {
        REQUIRE(values[lane].size() == state.size());
        for (uint64_t basis = 0; basis < state.size(); ++basis) {
            state.real_basis(basis)[lane] = values[lane][basis].real();
            state.imag_basis(basis)[lane] = values[lane][basis].imag();
        }
    }
}

void require_lane_matches(const InterleavedBatchState& batch, uint32_t lane,
                          const State& expected) {
    REQUIRE(batch.size() == expected.size());
    for (uint64_t basis = 0; basis < batch.size(); ++basis) {
        CAPTURE(lane, basis);
        check_complex({batch.real_basis(basis)[lane], batch.imag_basis(basis)[lane]},
                      {expected.real_data()[basis], expected.imag_data()[basis]}, kTolerance);
    }
}

}  // namespace

TEST_CASE("Interleaved batch state retains aligned storage across resets") {
    InterleavedBatchState state(5, 2, 65);

    REQUIRE(state.active_width() == 2);
    REQUIRE(state.max_active_width() == 5);
    REQUIRE(state.lane_capacity() == 65);
    REQUIRE(state.lane_pitch() == 72);
    REQUIRE(state.active_lanes() == 65);
    REQUIRE(state.capacity() == 32);
    REQUIRE(reinterpret_cast<uintptr_t>(state.real_basis(0)) %
                clifft::PageAlignedAllocation::kBaseAlignment ==
            0);
    for (uint32_t lane = 0; lane < state.active_lanes(); ++lane) {
        REQUIRE(state.real_basis(0)[lane] == 1.0);
        REQUIRE(state.imag_basis(0)[lane] == 0.0);
        for (uint64_t basis = 1; basis < state.size(); ++basis) {
            REQUIRE(state.real_basis(basis)[lane] == 0.0);
            REQUIRE(state.imag_basis(basis)[lane] == 0.0);
        }
    }

    state.set_active_width(5);
    state.reset(17);
    REQUIRE(state.active_width() == 2);
    REQUIRE(state.active_lanes() == 17);
    REQUIRE(state.real_basis(0)[16] == 1.0);
    REQUIRE(state.real_basis(0)[17] == 0.0);

    REQUIRE_THROWS_AS(InterleavedBatchState(1, 0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(InterleavedBatchState(1, 2, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(InterleavedBatchState(clifft::kDenseActiveWidthLimit, 0, 1),
                      std::invalid_argument);
}

TEST_CASE("Interleaved batch rotations match independent scalar lanes") {
    constexpr uint32_t kLanes = 17;
    constexpr double kHalfTurns = 0.137;
    std::vector<uint8_t> signs(kLanes);
    std::vector<double> signed_sines(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        signs[lane] = static_cast<uint8_t>((lane % 3) == 1);
    }

    for (uint32_t active_width = 1; active_width <= 5; ++active_width) {
        CAPTURE(active_width);
        const uint64_t mask = (uint64_t{1} << active_width) - 1;
        const std::vector<ActivePauli> paulis = {
            {0, 1}, {1, 0}, {1, 1}, {mask, mask >> 1}};
        std::vector<std::vector<std::complex<double>>> inputs;
        inputs.reserve(kLanes);
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            inputs.push_back(lane_state(active_width, lane));
        }

        for (ActivePauli pauli : paulis) {
            CAPTURE(pauli.x, pauli.z);
            InterleavedBatchState batch(active_width, active_width, kLanes);
            load_batch(batch, inputs);
            const auto rotation =
                prepare_rotation(pauli, active_width, kHalfTurns);
            prepare_interleaved_rotation_sines(signed_sines, rotation.sine, signs);
            apply_interleaved_rotation(batch, rotation, signed_sines);

            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                State expected(active_width, active_width);
                load_state(expected, inputs[lane]);
                apply_rotation(expected, rotation, signs[lane] != 0);
                require_lane_matches(batch, lane, expected);
            }
        }
    }
}

TEST_CASE("Interleaved batch promotion matches independent scalar lanes") {
    constexpr uint32_t kLanes = 17;
    std::vector<uint8_t> signs(kLanes);
    std::vector<double> signed_sines(kLanes);
    std::vector<std::vector<std::complex<double>>> inputs;
    inputs.reserve(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        signs[lane] = static_cast<uint8_t>((lane & 1U) != 0);
        inputs.push_back(lane_state(4, lane));
    }

    InterleavedBatchState batch(5, 4, kLanes);
    load_batch(batch, inputs);
    const auto promotion = prepare_promotion(-0.283);
    prepare_interleaved_rotation_sines(signed_sines, promotion.sine, signs);
    apply_interleaved_promotion(batch, promotion, signed_sines);
    REQUIRE(batch.active_width() == 5);

    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        State expected(5, 4);
        load_state(expected, inputs[lane]);
        apply_promotion(expected, promotion, signs[lane] != 0);
        require_lane_matches(batch, lane, expected);
    }
}

TEST_CASE("Interleaved fused rotations match independent scalar lanes") {
    constexpr uint32_t kLanes = 17;
    std::vector<std::vector<std::complex<double>>> inputs;
    inputs.reserve(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        inputs.push_back(lane_state(4, lane));
    }

    clifft::sampling::PreparedFusedRotation rotation;
    rotation.active_width = 4;
    rotation.orbit_rank = 2;
    rotation.orbit_masks = {1, 2};
    rotation.orbit_pivots = {0, 1};
    rotation.selector_masks = {4};
    rotation.matrices.resize(32);
    for (size_t variant = 0; variant < 2; ++variant) {
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                const double scale = 1.0 / static_cast<double>(1 + row + column + variant);
                rotation.matrices[variant * 16 + row * 4 + column] = {
                    (row == column ? 0.7 : 0.03) + 0.01 * static_cast<double>(variant),
                    scale * 0.02 * static_cast<double>(static_cast<int>(row) -
                                                       static_cast<int>(column))};
            }
        }
    }

    InterleavedBatchState batch(4, 4, kLanes);
    load_batch(batch, inputs);
    apply_interleaved_fused_rotation(batch, rotation);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        State expected(4, 4);
        load_state(expected, inputs[lane]);
        apply_fused_rotation(expected, rotation);
        require_lane_matches(batch, lane, expected);
    }
}

TEST_CASE("Interleaved dynamic fused rotations match independent scalar lanes") {
    constexpr uint32_t kLanes = 17;
    std::vector<std::vector<std::complex<double>>> inputs;
    inputs.reserve(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        inputs.push_back(lane_state(4, lane));
    }

    clifft::sampling::PreparedFusedRotation first;
    first.active_width = 4;
    first.orbit_rank = 1;
    first.orbit_masks = {2, 0};
    first.orbit_pivots = {1, 0};
    first.selector_masks = {4};
    first.matrices.resize(8);
    for (size_t index = 0; index < first.matrices.size(); ++index) {
        first.matrices[index] = {0.1 + 0.02 * static_cast<double>(index),
                                 -0.03 * static_cast<double>(index % 3)};
    }
    std::vector<clifft::sampling::PreparedFusedRotation> variants(4, first);
    for (size_t variant = 1; variant < variants.size(); ++variant) {
        for (std::complex<double>& value : variants[variant].matrices) {
            value += std::complex<double>{0.01 * static_cast<double>(variant),
                                          0.02 * static_cast<double>(variant)};
        }
    }
    std::vector<uint8_t> lane_variants(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        lane_variants[lane] = static_cast<uint8_t>((lane * 3) % variants.size());
    }

    InterleavedBatchState batch(4, 4, kLanes);
    load_batch(batch, inputs);
    apply_interleaved_dynamic_fused_rotation(batch, variants, lane_variants);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        State expected(4, 4);
        load_state(expected, inputs[lane]);
        apply_fused_rotation(expected, variants[lane_variants[lane]]);
        require_lane_matches(batch, lane, expected);
    }
}
