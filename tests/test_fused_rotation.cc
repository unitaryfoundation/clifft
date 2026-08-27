#include "clifft/sampling/executor.h"
#include "clifft/sampling/fused_rotation.h"
#include "clifft/sampling/kernel_dispatch.h"
#include "clifft/sampling/kernels.h"
#include "clifft/util/runtime_isa.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::apply_direct_rotation_neon;
using clifft::sampling::apply_direct_rotation_neon_parallel;
using clifft::sampling::apply_fused_rotation;
using clifft::sampling::apply_rotation;
using clifft::sampling::DirectRotationKernel;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::index;
using clifft::sampling::PlannedAction;
using clifft::sampling::prepare_dynamic_fused_rotation_run;
using clifft::sampling::prepare_fused_rotation_neon_sidecar;
using clifft::sampling::prepare_fused_rotation_run;
using clifft::sampling::prepare_rotation;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::State;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;

SamplingPlan rotation_plan(uint32_t active_width, std::span<const RotateActivePauli> rotations) {
    SamplingPlan plan;
    plan.num_qubits = active_width;
    plan.initial_active_width = active_width;
    plan.peak_active_width = active_width;
    plan.symbols = {SymbolKind::Presampled};
    for (const RotateActivePauli& rotation : rotations) {
        plan.actions.push_back(PlannedAction{active_width, active_width, rotation});
    }
    return plan;
}

void require_matches_scalar(const SamplingPlan& plan, uint8_t presampled_value,
                            size_t expected_action_count) {
    const ExecutablePlan executable(plan);
    REQUIRE(executable.num_actions() == expected_action_count);
    Executor executor(executable);
    executor.run_shot(std::array<uint8_t, 1>{presampled_value});

    State expected(plan.peak_active_width, plan.initial_active_width);
    for (const PlannedAction& planned : plan.actions) {
        const auto& rotation = std::get<RotateActivePauli>(planned.action);
        bool sign = rotation.sign.constant();
        for (SymbolId term : rotation.sign.terms()) {
            REQUIRE(index(term) == 0);
            sign ^= presampled_value != 0;
        }
        apply_rotation(expected,
                       prepare_rotation(rotation.pauli, planned.active_before, rotation.half_turns),
                       sign);
    }

    REQUIRE(executor.state().size() == expected.size());
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        CAPTURE(presampled_value, basis);
        REQUIRE_THAT(executor.state().real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
        REQUIRE_THAT(executor.state().imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
    }
}

}  // namespace

TEST_CASE("Executable rotation inspection includes prepared weights") {
    const std::array first_rotation = {
        RotateActivePauli{{0b1010, 0b0010}, 0.25, AffineBool(false)},
    };
    const std::array second_rotation = {
        RotateActivePauli{{0b1010, 0b0010}, 0.5, AffineBool(false)},
    };

    const ExecutablePlan first(rotation_plan(4, first_rotation));
    const ExecutablePlan second(rotation_plan(4, second_rotation));
    const std::string inspection = first.inspect_action(0);

    REQUIRE(inspection.find(" pair=3") != std::string::npos);
    REQUIRE(inspection.find(" cos=") != std::string::npos);
    REQUIRE(inspection.find(" sin=") != std::string::npos);
    REQUIRE(inspection != second.inspect_action(0));
}

TEST_CASE("Fused rotation stops at a dynamic sign") {
    const std::array rotations = {
        RotateActivePauli{{0b01, 0b00}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b10, 0b01}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b11, 0b11}, 0.4, AffineBool(false)},
        RotateActivePauli{{0b00, 0b10}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b01, 0b10}, -0.1, AffineBool(false)},
        RotateActivePauli{{0b10, 0b11}, 0.35, AffineBool(true)},
        RotateActivePauli{{0b11, 0b01}, -0.45, AffineBool(false)},
    };
    const SamplingPlan plan = rotation_plan(2, rotations);
    require_matches_scalar(plan, 0, 3);
    require_matches_scalar(plan, 1, 3);
}

TEST_CASE("Executable plan preserves optional provenance across fusion") {
    const std::array rotations = {
        RotateActivePauli{{0b01, 0b00}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b10, 0b01}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b11, 0b11}, 0.4, AffineBool(false)},
    };

    const SamplingPlan ordinary = rotation_plan(2, rotations);
    const ExecutablePlan ordinary_executable(ordinary);
    REQUIRE(ordinary_executable.num_actions() == 1);
    REQUIRE_FALSE(ordinary_executable.action_plan_range(0).has_value());

    SamplingPlan inspected = rotation_plan(2, rotations);
    inspected.source_map.emplace();
    for (uint32_t line = 1; line <= rotations.size(); ++line) {
        inspected.source_map->append(std::span<const uint32_t>(&line, 1));
    }
    const ExecutablePlan executable(inspected);
    REQUIRE(executable.num_actions() == 1);
    REQUIRE(executable.action_plan_range(0) == ExecutablePlan::PlanActionRange{0, 3});
    REQUIRE(executable.inspect_action(0) == "FUSED_ROTATION descriptor=0");
    REQUIRE(executable.inspect().find("plans=[0,3) FUSED_ROTATION descriptor=0") !=
            std::string::npos);
    REQUIRE_THROWS_AS(executable.action_plan_range(1), std::out_of_range);
}

TEST_CASE("Fused rotation leaves a below threshold run unfused") {
    const std::array rotations = {
        RotateActivePauli{{0b01, 0b10}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b10, 0b01}, -0.3, AffineBool(true)},
    };
    require_matches_scalar(rotation_plan(2, rotations), 0, 2);
}

TEST_CASE("Fused rotation falls back for a rank three run") {
    const std::array rotations = {
        RotateActivePauli{{0b001, 0b010}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b010, 0b100}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b100, 0b001}, 0.4, AffineBool(false)},
    };
    require_matches_scalar(rotation_plan(3, rotations), 0, 3);
}

#if defined(CLIFFT_TESTS_HAVE_APPLE_NEON)
TEST_CASE("Apple NEON fused rotation matches scalar") {
    const std::array rotations = {
        RotateActivePauli{{0b000110, 0b001011}, 0.17, AffineBool(false)},
        RotateActivePauli{{0b110001, 0b010101}, -0.23, AffineBool(true)},
        RotateActivePauli{{0b110111, 0b100011}, 0.31, AffineBool(false)},
        RotateActivePauli{{0b000110, 0b011001}, -0.11, AffineBool(true)},
        RotateActivePauli{{0b110001, 0b101010}, 0.29, AffineBool(false)},
    };
    const SamplingPlan plan = rotation_plan(6, rotations);
    const auto run = prepare_fused_rotation_run(plan.actions);
    REQUIRE(run.action_count == rotations.size());
    REQUIRE(run.rotation.has_value());

    const auto sidecar = prepare_fused_rotation_neon_sidecar(*run.rotation);
    REQUIRE(sidecar.storage != nullptr);
    REQUIRE(sidecar.kernel != nullptr);
    REQUIRE(sidecar.parallel_kernel != nullptr);

    State expected(6, 6);
    State actual(6, 6);
    State parallel(6, 6);
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        const double real = static_cast<double>(basis + 1) / 100.0;
        const double imag = -static_cast<double>((basis * 7) % 19 + 1) / 80.0;
        expected.real_data()[basis] = real;
        expected.imag_data()[basis] = imag;
        actual.real_data()[basis] = real;
        actual.imag_data()[basis] = imag;
        parallel.real_data()[basis] = real;
        parallel.imag_data()[basis] = imag;
    }

    apply_fused_rotation(expected, *run.rotation);
    sidecar.kernel(actual, *run.rotation, sidecar.storage.get());
    sidecar.parallel_kernel(parallel, *run.rotation, sidecar.storage.get(), 4, 0);
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        CAPTURE(basis);
        REQUIRE_THAT(actual.real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 2e-12));
        REQUIRE_THAT(actual.imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 2e-12));
        REQUIRE_THAT(parallel.real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 2e-12));
        REQUIRE_THAT(parallel.imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 2e-12));
    }
}

TEST_CASE("Apple NEON fused rotation rejects a lane pivot") {
    const std::array rotations = {
        RotateActivePauli{{0b000001, 0b001010}, 0.17, AffineBool(false)},
        RotateActivePauli{{0b010100, 0b000101}, -0.23, AffineBool(true)},
        RotateActivePauli{{0b010101, 0b100010}, 0.31, AffineBool(false)},
    };
    const SamplingPlan plan = rotation_plan(6, rotations);
    const auto run = prepare_fused_rotation_run(plan.actions);
    REQUIRE(run.rotation.has_value());
    REQUIRE(prepare_fused_rotation_neon_sidecar(*run.rotation).storage == nullptr);
}

TEST_CASE("Apple NEON direct rotation serial and parallel match scalar") {
    constexpr uint32_t kActiveWidth = 10;
    const std::array cases = {
        std::pair{ActivePauli{0, 0b1010101011}, DirectRotationKernel::Diagonal},
        std::pair{ActivePauli{0b0000000001, 0b1010101010}, DirectRotationKernel::LanePaired},
        std::pair{ActivePauli{0b1000100101, 0b0111011011}, DirectRotationKernel::HighPivot},
    };
    for (const auto& [pauli, kernel] : cases) {
        for (bool sign : {false, true}) {
            CAPTURE(pauli.x, pauli.z, sign);
            State expected(kActiveWidth, kActiveWidth);
            State serial(kActiveWidth, kActiveWidth);
            State parallel(kActiveWidth, kActiveWidth);
            for (uint64_t basis = 0; basis < expected.size(); ++basis) {
                const double real = static_cast<double>((basis * 13) % 29 + 1) / 100.0;
                const double imag = -static_cast<double>((basis * 7) % 23 + 1) / 90.0;
                expected.real_data()[basis] = real;
                expected.imag_data()[basis] = imag;
                serial.real_data()[basis] = real;
                serial.imag_data()[basis] = imag;
                parallel.real_data()[basis] = real;
                parallel.imag_data()[basis] = imag;
            }
            const auto rotation = prepare_rotation(pauli, kActiveWidth, 0.271);
            apply_rotation(expected, rotation, sign);
            apply_direct_rotation_neon(serial, rotation, kernel, sign);
            apply_direct_rotation_neon_parallel(parallel, rotation, kernel, sign, 4, 0);
            for (uint64_t basis = 0; basis < expected.size(); ++basis) {
                CAPTURE(basis);
                REQUIRE_THAT(serial.real_data()[basis],
                             Catch::Matchers::WithinAbs(expected.real_data()[basis], 2e-12));
                REQUIRE_THAT(serial.imag_data()[basis],
                             Catch::Matchers::WithinAbs(expected.imag_data()[basis], 2e-12));
                REQUIRE_THAT(parallel.real_data()[basis],
                             Catch::Matchers::WithinAbs(expected.real_data()[basis], 2e-12));
                REQUIRE_THAT(parallel.imag_data()[basis],
                             Catch::Matchers::WithinAbs(expected.imag_data()[basis], 2e-12));
            }
        }
    }
}
#endif

TEST_CASE("Dynamic fused rotation matches scalar across affine sign values") {
    const SymbolId first_sign{0};
    const SymbolId second_sign{1};
    const std::array rotations = {
        RotateActivePauli{{0b001000, 0b000101}, 0.17, AffineBool::symbol(first_sign)},
        RotateActivePauli{{0b100000, 0b010010}, -0.23, AffineBool::symbol(second_sign)},
        RotateActivePauli{{0b101000, 0b100001}, 0.31, AffineBool(false, {first_sign, second_sign})},
        RotateActivePauli{{0b001000, 0b001110}, -0.11, AffineBool(true, {first_sign})},
        RotateActivePauli{{0b100000, 0b011001}, 0.29, AffineBool(true, {second_sign})},
        RotateActivePauli{{0b101000, 0b100100}, -0.37, AffineBool(true, {first_sign, second_sign})},
        RotateActivePauli{{0b001000, 0b010101}, 0.13, AffineBool(false)},
        RotateActivePauli{{0b100000, 0b101010}, -0.19, AffineBool::symbol(first_sign)},
    };

    SamplingPlan plan;
    plan.num_qubits = 6;
    plan.initial_active_width = 6;
    plan.peak_active_width = 6;
    plan.symbols = {SymbolKind::Presampled, SymbolKind::Presampled};
    for (const RotateActivePauli& rotation : rotations) {
        plan.actions.push_back(PlannedAction{6, 6, rotation});
    }

    const auto prepared_run = prepare_dynamic_fused_rotation_run(plan.actions);
    REQUIRE(prepared_run.action_count == rotations.size());
    REQUIRE(prepared_run.rotation.has_value());
    REQUIRE(prepared_run.rotation->sign_basis.size() == 2);
    REQUIRE(prepared_run.rotation->variants.size() == 4);

    const ExecutablePlan executable(plan);
    // Dynamic-sign fusion is deliberately AVX-512-only; other ISAs retain the
    // sequential rotations even though constant-sign fusion supports AVX2.
    const size_t expected_action_count =
        clifft::internal::runtime_isa() == clifft::internal::RuntimeIsa::Avx512 ? 1
                                                                                : rotations.size();
    REQUIRE(executable.num_actions() == expected_action_count);
    for (uint8_t first_value : {uint8_t{0}, uint8_t{1}}) {
        for (uint8_t second_value : {uint8_t{0}, uint8_t{1}}) {
            const std::array values = {first_value, second_value};
            Executor executor(executable);
            executor.run_shot(values);

#if defined(CLIFFT_TESTS_HAVE_OPENMP)
            Executor parallel_executor(executable, 0, 4, 0);
            parallel_executor.run_shot(values);
            REQUIRE(std::ranges::equal(parallel_executor.state().real(), executor.state().real()));
            REQUIRE(std::ranges::equal(parallel_executor.state().imag(), executor.state().imag()));
#endif

            State expected(6, 6);
            for (const RotateActivePauli& rotation : rotations) {
                bool sign = rotation.sign.constant();
                for (SymbolId term : rotation.sign.terms()) {
                    sign ^= values[index(term)] != 0;
                }
                apply_rotation(expected, prepare_rotation(rotation.pauli, 6, rotation.half_turns),
                               sign);
            }

            uint32_t variant = 0;
            for (size_t i = 0; i < prepared_run.rotation->sign_basis.size(); ++i) {
                const AffineBool& expression = prepared_run.rotation->sign_basis[i];
                bool value = expression.constant();
                for (SymbolId term : expression.terms()) {
                    value ^= values[index(term)] != 0;
                }
                variant |= static_cast<uint32_t>(value) << i;
            }
            State prepared_state(6, 6);
            apply_fused_rotation(prepared_state, prepared_run.rotation->variants[variant]);

            for (uint64_t basis = 0; basis < expected.size(); ++basis) {
                CAPTURE(first_value, second_value, basis);
                REQUIRE_THAT(executor.state().real_data()[basis],
                             Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
                REQUIRE_THAT(executor.state().imag_data()[basis],
                             Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
                REQUIRE_THAT(prepared_state.real_data()[basis],
                             Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
                REQUIRE_THAT(prepared_state.imag_data()[basis],
                             Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
            }
        }
    }
}

TEST_CASE("Dynamic fused rotation enforces bounded selection") {
    const SymbolId first_sign{0};
    const SymbolId second_sign{1};
    constexpr std::array<uint64_t, 3> kXMasks = {0b001000, 0b100000, 0b101000};
    const std::array signs = {AffineBool::symbol(first_sign), AffineBool::symbol(second_sign),
                              AffineBool(false, {first_sign, second_sign})};
    std::vector<PlannedAction> actions;
    for (size_t i = 0; i < 8; ++i) {
        actions.push_back(PlannedAction{
            6, 6, RotateActivePauli{{kXMasks[i % 3], uint64_t{1} << (i % 6)}, 0.2, signs[i % 3]}});
    }

    REQUIRE(prepare_dynamic_fused_rotation_run(std::span(actions).first(7)).rotation ==
            std::nullopt);

    std::vector<PlannedAction> rank_three = actions;
    std::get<RotateActivePauli>(rank_three[2].action).pauli.x = 0b010000;
    REQUIRE(prepare_dynamic_fused_rotation_run(rank_three).rotation == std::nullopt);

    std::vector<PlannedAction> sign_rank_three = actions;
    std::get<RotateActivePauli>(sign_rank_three[2].action).sign = AffineBool::symbol(SymbolId{2});
    REQUIRE(prepare_dynamic_fused_rotation_run(sign_rank_three).rotation == std::nullopt);

    std::vector<PlannedAction> low_pivots = actions;
    for (PlannedAction& planned : low_pivots) {
        auto& rotation = std::get<RotateActivePauli>(planned.action);
        rotation.pauli.x >>= 3;
    }
    REQUIRE(prepare_dynamic_fused_rotation_run(low_pivots).rotation == std::nullopt);
}

TEST_CASE("Dynamic fused rotation stops at execution barriers") {
    const SymbolId first_sign{0};
    const SymbolId second_sign{1};
    constexpr std::array<uint64_t, 3> kXMasks = {0b001000, 0b100000, 0b101000};
    const std::array signs = {AffineBool::symbol(first_sign), AffineBool::symbol(second_sign),
                              AffineBool(false, {first_sign, second_sign})};
    std::vector<PlannedAction> actions;
    for (size_t i = 0; i < 8; ++i) {
        actions.push_back(PlannedAction{
            6, 6, RotateActivePauli{{kXMasks[i % 3], uint64_t{1} << (i % 6)}, 0.2, signs[i % 3]}});
    }

    std::vector<PlannedAction> width_barrier = actions;
    width_barrier.push_back(PlannedAction{
        5, 5, RotateActivePauli{{0b001000, 0b000101}, 0.25, AffineBool::symbol(first_sign)}});
    const auto width_run = prepare_dynamic_fused_rotation_run(width_barrier);
    REQUIRE(width_run.action_count == actions.size());
    REQUIRE(width_run.rotation.has_value());
}

TEST_CASE("Direct rotation SIMD matches scalar across eligible shapes") {
    const std::array rotations = {
        RotateActivePauli{{0, 0b101011}, -0.3, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b001011, 0b000011}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b011001, 0b010101}, -0.4, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b000101, 0b000110}, 0.35, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b001111, 0b001001}, -0.15, AffineBool::symbol(SymbolId{0})},
    };
    const SamplingPlan plan = rotation_plan(6, rotations);
    require_matches_scalar(plan, 0, rotations.size());
    require_matches_scalar(plan, 1, rotations.size());
}

TEST_CASE("Direct rotation SIMD matches scalar for every lane permutation") {
    std::array<RotateActivePauli, 16> rotations;
    for (uint64_t lane_xor = 0; lane_xor < 8; ++lane_xor) {
        const uint64_t x = 0b100000 | lane_xor;
        rotations[2 * lane_xor] =
            RotateActivePauli{{x, (~x) & 0b111111}, 0.2, AffineBool::symbol(SymbolId{0})};
        rotations[2 * lane_xor + 1] =
            RotateActivePauli{{x, x & (~x + 1)}, -0.3, AffineBool::symbol(SymbolId{0})};
    }
    const SamplingPlan plan = rotation_plan(6, rotations);
    require_matches_scalar(plan, 0, rotations.size());
    require_matches_scalar(plan, 1, rotations.size());
}

TEST_CASE("Direct rotation SIMD matches scalar for every lane-paired mask") {
    std::array<RotateActivePauli, 14> rotations;
    for (uint64_t x = 1; x < 8; ++x) {
        rotations[2 * (x - 1)] =
            RotateActivePauli{{x, (~x) & 0b111111}, 0.2, AffineBool::symbol(SymbolId{0})};
        rotations[2 * (x - 1) + 1] =
            RotateActivePauli{{x, x & (~x + 1)}, -0.3, AffineBool::symbol(SymbolId{0})};
    }
    const SamplingPlan plan = rotation_plan(6, rotations);
    require_matches_scalar(plan, 0, rotations.size());
    require_matches_scalar(plan, 1, rotations.size());
}

TEST_CASE("Direct rotation SIMD matches scalar for imaginary lane pairs with high Z") {
    const std::array rotations = {
        RotateActivePauli{{0b001, 0b111001}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b110, 0b101010}, -0.3, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b101, 0b011100}, 0.45, AffineBool::symbol(SymbolId{0})},
    };
    const SamplingPlan plan = rotation_plan(6, rotations);
    require_matches_scalar(plan, 0, rotations.size());
    require_matches_scalar(plan, 1, rotations.size());
}

TEST_CASE("Direct rotation SIMD matches scalar with intermediate high bits") {
    const std::array rotations = {
        RotateActivePauli{{0b101010, 0b001110}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b111000, 0b100101}, -0.3, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b110011, 0b010001}, 0.4, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0, 0b101000}, -0.25, AffineBool::symbol(SymbolId{0})},
    };
    const SamplingPlan plan = rotation_plan(6, rotations);
    require_matches_scalar(plan, 0, rotations.size());
    require_matches_scalar(plan, 1, rotations.size());
}

TEST_CASE("Direct rotation SIMD matches scalar at vector boundaries") {
    const std::array avx2_diagonal = {
        RotateActivePauli{{0, 0b11}, 0.3, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(2, avx2_diagonal), 0, avx2_diagonal.size());
    require_matches_scalar(rotation_plan(2, avx2_diagonal), 1, avx2_diagonal.size());

    const std::array avx2_lane_paired = {
        RotateActivePauli{{0b01, 0b10}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b10, 0b01}, -0.3, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(2, avx2_lane_paired), 0, avx2_lane_paired.size());
    require_matches_scalar(rotation_plan(2, avx2_lane_paired), 1, avx2_lane_paired.size());

    const std::array diagonal = {
        RotateActivePauli{{0, 0b111}, 0.3, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(3, diagonal), 0, diagonal.size());
    require_matches_scalar(rotation_plan(3, diagonal), 1, diagonal.size());

    const std::array lane_paired = {
        RotateActivePauli{{0b001, 0b110}, 0.2, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b010, 0b101}, -0.3, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b111, 0b011}, 0.4, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(3, lane_paired), 0, lane_paired.size());
    require_matches_scalar(rotation_plan(3, lane_paired), 1, lane_paired.size());

    const std::array paired = {
        RotateActivePauli{{0b1101, 0b0011}, -0.25, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b1011, 0b0101}, 0.4, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(4, paired), 0, paired.size());
    require_matches_scalar(rotation_plan(4, paired), 1, paired.size());

    const std::array pivot_four = {
        RotateActivePauli{{0b110000, 0b101011}, 0.35, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b010000, 0b011101}, -0.2, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(6, pivot_four), 0, pivot_four.size());
    require_matches_scalar(rotation_plan(6, pivot_four), 1, pivot_four.size());
}
