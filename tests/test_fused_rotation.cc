#include "clifft/sampling/executor.h"
#include "clifft/sampling/kernels.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace {

using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::apply_rotation;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::index;
using clifft::sampling::PlannedAction;
using clifft::sampling::prepare_rotation;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::State;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;

SamplingPlan rotation_plan(uint32_t active_width, std::span<const RotateActivePauli> rotations) {
    SamplingPlan plan;
    plan.num_qubits = active_width;
    plan.initial_active_width = active_width;
    plan.max_active_width = active_width;
    plan.symbols = {SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt}};
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

    State expected(plan.max_active_width, plan.initial_active_width, plan.global_weight);
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
    REQUIRE_THAT(executor.state().global_scalar().real(),
                 Catch::Matchers::WithinAbs(expected.global_scalar().real(), 1e-12));
    REQUIRE_THAT(executor.state().global_scalar().imag(),
                 Catch::Matchers::WithinAbs(expected.global_scalar().imag(), 1e-12));
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        CAPTURE(presampled_value, basis);
        REQUIRE_THAT(executor.state().real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
        REQUIRE_THAT(executor.state().imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
    }
}

}  // namespace

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

TEST_CASE("Fused rotation preserves a signed identity barrier") {
    const std::array rotations = {
        RotateActivePauli{{0b01, 0b00}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b10, 0b01}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b11, 0b11}, 0.4, AffineBool(false)},
        RotateActivePauli{{0b00, 0b00}, 0.5, AffineBool(true)},
        RotateActivePauli{{0b01, 0b10}, -0.1, AffineBool(false)},
        RotateActivePauli{{0b10, 0b11}, 0.35, AffineBool(true)},
        RotateActivePauli{{0b11, 0b01}, -0.45, AffineBool(false)},
    };
    require_matches_scalar(rotation_plan(2, rotations), 0, 3);
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

TEST_CASE("Direct rotation SIMD matches scalar across eligible shapes") {
    const std::array rotations = {
        RotateActivePauli{{0, 0}, 0.125, AffineBool::symbol(SymbolId{0})},
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
    const std::array diagonal = {
        RotateActivePauli{{0, 0b111}, 0.3, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(3, diagonal), 0, diagonal.size());
    require_matches_scalar(rotation_plan(3, diagonal), 1, diagonal.size());

    const std::array paired = {
        RotateActivePauli{{0b1101, 0b0011}, -0.25, AffineBool::symbol(SymbolId{0})},
        RotateActivePauli{{0b1011, 0b0101}, 0.4, AffineBool::symbol(SymbolId{0})},
    };
    require_matches_scalar(rotation_plan(4, paired), 0, paired.size());
    require_matches_scalar(rotation_plan(4, paired), 1, paired.size());
}
