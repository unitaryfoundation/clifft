#include "clifft/sampling/planner.h"

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <variant>

using clifft::HirModule;
using clifft::MeasRecordIdx;
using clifft::NoiseSite;
using clifft::sampling::AffineBool;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::plan_sampling;
using clifft::test::X;
using clifft::test::Z;

namespace {

template <typename T>
const T& action_as(const SamplingPlan& plan, size_t index) {
    return std::get<T>(plan.actions.at(index).action);
}

}  // namespace

TEST_CASE("Sampling planner preserves empty module metadata") {
    HirModule hir(3, 0);
    hir.global_weight = {0.25, -0.5};

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.num_qubits == 3);
    REQUIRE(plan.global_weight == std::complex<double>{0.25, -0.5});
    REQUIRE(plan.initial_active_width == 0);
    REQUIRE(plan.max_active_width == 0);
    REQUIRE(plan.actions.empty());
    REQUIRE(plan.symbols.empty());
}

TEST_CASE("Sampling planner promotes rotations and keeps later active support") {
    HirModule hir(1, 2);
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_phase_rotation(hir, X(0), 0, false, 0.5);

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.max_active_width == 1);
    REQUIRE(plan.actions.size() == 2);
    const auto& promotion = action_as<PromoteDormantRotation>(plan, 0);
    REQUIRE(promotion.dormant_pivot == 0);
    REQUIRE(promotion.half_turns == 0.25);
    REQUIRE(promotion.sign == AffineBool(false));
    const auto& rotation = action_as<RotateActivePauli>(plan, 1);
    REQUIRE(rotation.pauli.x == 1);
    REQUIRE(rotation.pauli.z == 0);
    REQUIRE(rotation.half_turns == 0.5);
}

TEST_CASE("Sampling planner emits direct multi-coordinate active Paulis") {
    HirModule hir(2, 3);
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_tgate(hir, X(0) | X(1), 0, false);
    clifft::test::append_phase_rotation(hir, X(1), 0, false, 0.5);

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.max_active_width == 2);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[1].action));
    const auto& rotation = action_as<RotateActivePauli>(plan, 2);
    REQUIRE(rotation.pauli.x == 3);
    REQUIRE(rotation.pauli.z == 0);
}

TEST_CASE("Sampling planner preserves high physical Pauli coordinates") {
    HirModule hir(129, 2);
    hir.append_tgate(false, [](clifft::MutablePauliMaskView slot) {
        slot.x().bit_set(128, true);
    });
    hir.append_phase_rotation(0.5, [](clifft::MutablePauliMaskView slot) {
        slot.x().bit_set(128, true);
    });

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.max_active_width == 1);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    const auto& rotation = action_as<RotateActivePauli>(plan, 1);
    REQUIRE(rotation.pauli.x == 1);
    REQUIRE(rotation.pauli.z == 0);
}

TEST_CASE("Sampling planner records repeat dormant measurements consistently") {
    HirModule hir(1, 2);
    hir.num_measurements = 2;
    clifft::test::append_measure(hir, X(0), 0, false, MeasRecordIdx{0});
    clifft::test::append_measure(hir, X(0), 0, false, MeasRecordIdx{1});

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 2);
    REQUIRE(plan.symbols.size() == 1);
    const auto& first = action_as<MeasureDormantRandom>(plan, 0);
    REQUIRE(first.branch == SymbolId{0});
    REQUIRE(first.outcome == AffineBool::symbol(SymbolId{0}));
    const auto& second = action_as<RecordClassical>(plan, 1);
    REQUIRE(second.outcome == AffineBool::symbol(SymbolId{0}));
}

TEST_CASE("Sampling planner correlates arbitrary repeated dormant measurements") {
    for (uint64_t x = 1; x < 4; ++x) {
        for (uint64_t z = 0; z < 4; ++z) {
            CAPTURE(x, z);
            HirModule hir(2, 2);
            hir.num_measurements = 2;
            clifft::test::append_measure(hir, x, z, true, MeasRecordIdx{0});
            clifft::test::append_measure(hir, x, z, true, MeasRecordIdx{1});

            const SamplingPlan plan = plan_sampling(hir);

            const auto& first = action_as<MeasureDormantRandom>(plan, 0);
            const auto& repeated = action_as<RecordClassical>(plan, 1);
            REQUIRE(repeated.outcome == first.outcome);
        }
    }
}

TEST_CASE("Sampling planner accepts traced rotation and measurement HIR") {
    const clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nT 0\nM 0\n"));

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 2);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    REQUIRE(std::holds_alternative<MeasureActivePauli>(plan.actions[1].action));
    REQUIRE(plan.symbols.size() == 1);
}

TEST_CASE("Sampling planner collapses active measurements and propagates branches") {
    HirModule hir(1, 3);
    hir.num_measurements = 2;
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_measure(hir, 0, Z(0), false, MeasRecordIdx{0});
    clifft::test::append_measure(hir, 0, Z(0), false, MeasRecordIdx{1});

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 3);
    const auto& measurement = action_as<MeasureActivePauli>(plan, 1);
    REQUIRE(measurement.pauli.x == 0);
    REQUIRE(measurement.pauli.z == 1);
    REQUIRE(measurement.active_pivot == 0);
    REQUIRE(measurement.branch == SymbolId{0});
    REQUIRE(measurement.outcome == AffineBool::symbol(SymbolId{0}));
    REQUIRE(plan.actions[1].active_before == 1);
    REQUIRE(plan.actions[1].active_after == 0);
    const auto& repeated = action_as<RecordClassical>(plan, 2);
    REQUIRE(repeated.outcome == AffineBool::symbol(SymbolId{0}));
}

TEST_CASE("Sampling planner correlates arbitrary repeated active measurements") {
    for (uint64_t x = 0; x < 4; ++x) {
        for (uint64_t z = 0; z < 4; ++z) {
            if (x == 0 && z == 0) {
                continue;
            }
            CAPTURE(x, z);
            HirModule hir(2, 4);
            hir.num_measurements = 2;
            clifft::test::append_tgate(hir, X(0), 0, false);
            clifft::test::append_tgate(hir, X(1), 0, false);
            clifft::test::append_measure(hir, x, z, true, MeasRecordIdx{0});
            clifft::test::append_measure(hir, x, z, true, MeasRecordIdx{1});

            const SamplingPlan plan = plan_sampling(hir);

            const auto& first = action_as<MeasureActivePauli>(plan, 2);
            const auto& repeated = action_as<RecordClassical>(plan, 3);
            REQUIRE(repeated.outcome == first.outcome);
        }
    }
}

TEST_CASE("Sampling planner keeps geometric Pauli signs") {
    HirModule hir(1, 2);
    clifft::test::append_tgate(hir, X(0), Z(0), false);
    clifft::test::append_phase_rotation(hir, X(0), 0, false, 0.5);

    const SamplingPlan plan = plan_sampling(hir);

    const auto& rotation = action_as<RotateActivePauli>(plan, 1);
    REQUIRE(rotation.pauli.x == 1);
    REQUIRE(rotation.pauli.z == 1);
    REQUIRE(rotation.sign == AffineBool(true));
}

TEST_CASE("Sampling planner retains balanced rotation global factors") {
    HirModule hir(1, 2);
    hir.global_weight = {std::cos(-std::numbers::pi / 4.0),
                         std::sin(-std::numbers::pi / 4.0)};
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 0.5);
    clifft::test::append_tgate(hir, 0, Z(0), false);

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 2);
    REQUIRE(action_as<RotateActivePauli>(plan, 0).pauli.is_identity());
    REQUIRE(action_as<RotateActivePauli>(plan, 1).pauli.is_identity());
    const std::complex<double> expected{std::cos(std::numbers::pi / 8.0),
                                        std::sin(std::numbers::pi / 8.0)};
    REQUIRE_THAT(plan.global_weight.real(),
                 Catch::Matchers::WithinAbs(expected.real(), 1e-12));
    REQUIRE_THAT(plan.global_weight.imag(),
                 Catch::Matchers::WithinAbs(expected.imag(), 1e-12));
}

TEST_CASE("Sampling planner rejects unsupported operations explicitly") {
    HirModule hir(1, 0);
    hir.noise_sites.push_back(NoiseSite{});
    hir.append_noise(clifft::NoiseSiteIdx{0});

    REQUIRE_THROWS_WITH(plan_sampling(hir),
                        "sampling planner does not support HIR operation NOISE at index 0");
}

TEST_CASE("Sampling planner reports the dense active width limit") {
    HirModule hir(60, 60);
    for (uint32_t q = 0; q < 60; ++q) {
        clifft::test::append_tgate(hir, X(q), 0, false);
    }

    REQUIRE_THROWS_WITH(
        plan_sampling(hir),
        "sampling planner active width would reach 60, but the dense-state limit is 60");
}

TEST_CASE("Sampling planner output is deterministic") {
    HirModule hir(2, 4);
    hir.num_measurements = 2;
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_phase_rotation(hir, X(0) | X(1), Z(1), true, -0.25);
    clifft::test::append_measure(hir, X(1), Z(0), false, MeasRecordIdx{0});
    clifft::test::append_measure(hir, X(1), Z(0), true, MeasRecordIdx{1});

    const SamplingPlan first = plan_sampling(hir);
    const SamplingPlan second = plan_sampling(hir);

    REQUIRE(first.inspect() == second.inspect());
}
