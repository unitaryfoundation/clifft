#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

using clifft::sampling::AffineBool;
using clifft::sampling::classify_measurement_branch;
using clifft::sampling::DefineSymbol;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentSiteId;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::MeasurementBranchKind;
using clifft::sampling::MeasurementProbabilities;
using clifft::sampling::PlannedAction;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordSlot;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;

namespace {

SamplingPlan active_then_dormant_plan(double promotion_half_turns) {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.max_active_width = 1;
    plan.num_visible_records = 2;
    plan.symbols = {
        SymbolInfo{SymbolKind::Branch, 1, std::nullopt},
        SymbolInfo{SymbolKind::Branch, 2, std::nullopt},
    };
    plan.actions = {
        PlannedAction{0, 1, PromoteDormantRotation{promotion_half_turns, AffineBool(false)}},
        PlannedAction{1, 0,
                      MeasureActivePauli{{0, 1}, 0, SymbolId{0},
                                         AffineBool::symbol(SymbolId{0}), RecordSlot{0}}},
        PlannedAction{0, 0,
                      MeasureDormantRandom{1, SymbolId{1}, AffineBool::symbol(SymbolId{1}),
                                           RecordSlot{1}}},
    };
    return plan;
}

void require_matches_legacy(std::string_view circuit_text, uint32_t shots, uint64_t seed) {
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit_text));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    Executor executor(executable, seed);

    const clifft::CompiledModule legacy_program = clifft::lower(hir);
    clifft::SchrodingerState legacy({.peak_rank = legacy_program.peak_rank,
                                     .num_measurements = legacy_program.total_meas_slots,
                                     .num_qubits = legacy_program.num_qubits,
                                     .seed = seed});

    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (shot > 0) {
            legacy.reset();
        }
        clifft::execute(legacy_program, legacy);
        executor.run_shot();

        CAPTURE(circuit_text, shot);
        REQUIRE(executor.visible_records().size() == legacy_program.num_measurements);
        REQUIRE(std::ranges::equal(executor.visible_records(),
                                   std::span<const uint8_t>(legacy.meas_record).first(
                                       legacy_program.num_measurements)));
    }
}

}  // namespace

TEST_CASE("Sampling executor classifies measurement dust without drawing") {
    const auto zero = classify_measurement_branch(MeasurementProbabilities{1.0, 0.0});
    REQUIRE(zero.kind == MeasurementBranchKind::Zero);
    REQUIRE_FALSE(zero.clamped_dust);

    const auto dusty_zero = classify_measurement_branch(MeasurementProbabilities{1.0, 1e-30});
    REQUIRE(dusty_zero.kind == MeasurementBranchKind::Zero);
    REQUIRE(dusty_zero.clamped_dust);

    const auto dusty_one = classify_measurement_branch(MeasurementProbabilities{1e-30, 1.0});
    REQUIRE(dusty_one.kind == MeasurementBranchKind::One);
    REQUIRE(dusty_one.clamped_dust);

    const auto random = classify_measurement_branch(MeasurementProbabilities{0.25, 0.75});
    REQUIRE(random.kind == MeasurementBranchKind::Random);
}

TEST_CASE("Sampling executor evaluates presampled and derived affine symbols") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt},
        SymbolInfo{SymbolKind::Derived, 0, std::nullopt},
    };
    plan.actions = {
        PlannedAction{0, 0,
                      DefineSymbol{SymbolId{1},
                                   AffineBool(true, std::vector<SymbolId>{SymbolId{0}})}},
        PlannedAction{0, 0,
                      RecordClassical{AffineBool::symbol(SymbolId{1}), RecordSlot{0}}},
        PlannedAction{0, 0,
                      RecordClassical{AffineBool::symbol(SymbolId{0}), RecordSlot{1}}},
    };

    const ExecutablePlan executable(plan);
    Executor executor(executable, 42);
    executor.run_shot(std::array<uint8_t, 1>{0});
    REQUIRE(executor.visible_records()[0] == 1);
    REQUIRE(executor.hidden_records()[0] == 0);
    REQUIRE(executor.symbols()[1] == 1);

    const uint8_t* const visible = executor.visible_records().data();
    const uint8_t* const hidden = executor.hidden_records().data();
    const uint8_t* const symbols = executor.symbols().data();
    const double* const real = executor.state().real_data();
    executor.run_shot(std::array<uint8_t, 1>{1});
    REQUIRE(executor.visible_records()[0] == 0);
    REQUIRE(executor.hidden_records()[0] == 1);
    REQUIRE(executor.symbols()[1] == 0);
    REQUIRE(executor.visible_records().data() == visible);
    REQUIRE(executor.hidden_records().data() == hidden);
    REQUIRE(executor.symbols().data() == symbols);
    REQUIRE(executor.state().real_data() == real);
}

TEST_CASE("Sampling executor skips RNG for deterministic active measurements") {
    const ExecutablePlan executable(active_then_dormant_plan(0.0));
    Executor executor(executable, 456);
    clifft::Xoshiro256PlusPlus expected_rng(456);
    const bool expected_dormant_branch = expected_rng.next_double() >= 0.5;
    const bool branch_if_active_had_drawn = expected_rng.next_double() >= 0.5;
    REQUIRE(expected_dormant_branch != branch_if_active_had_drawn);

    executor.run_shot();

    REQUIRE(executor.visible_records()[0] == 0);
    REQUIRE(executor.visible_records()[1] == expected_dormant_branch);
    REQUIRE(executor.dust_clamps() == 0);
}

TEST_CASE("Sampling executor clamps positive active measurement dust") {
    const ExecutablePlan executable(active_then_dormant_plan(1e-10));
    Executor executor(executable, 456);
    clifft::Xoshiro256PlusPlus expected_rng(456);
    const bool expected_dormant_branch = expected_rng.next_double() >= 0.5;
    const bool branch_if_active_had_drawn = expected_rng.next_double() >= 0.5;
    REQUIRE(expected_dormant_branch != branch_if_active_had_drawn);

    executor.run_shot();

    REQUIRE(executor.visible_records()[0] == 0);
    REQUIRE(executor.visible_records()[1] == expected_dormant_branch);
    REQUIRE(executor.dust_clamps() == 1);
}

TEST_CASE("Sampling executor retains exact identity rotation scalars") {
    clifft::HirModule hir(1, 1);
    hir.append_tgate(false, [](clifft::MutablePauliMaskView slot) {
        slot.z().bit_set(0, true);
        slot.set_sign(true);
    });
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    Executor executor(executable);

    executor.run_shot();

    const std::complex<double> expected{std::cos(std::numbers::pi / 4.0),
                                        std::sin(std::numbers::pi / 4.0)};
    REQUIRE_THAT(executor.state().global_scalar().real(),
                 Catch::Matchers::WithinAbs(expected.real(), 1e-12));
    REQUIRE_THAT(executor.state().global_scalar().imag(),
                 Catch::Matchers::WithinAbs(expected.imag(), 1e-12));
}

TEST_CASE("Sampling executor rejects instrument boundaries before dispatch") {
    SamplingPlan plan;
    plan.num_instrument_sites = 1;
    plan.actions = {PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}}}};

    REQUIRE_THROWS_WITH(ExecutablePlan(plan),
                        "sampling executable does not yet support instrument boundary site 0");
}

TEST_CASE("Sampling executor matches legacy records for supported circuits") {
    require_matches_legacy("M 0\n", 32, 1234);
    require_matches_legacy("H 0\nM 0\nM 0\n", 64, 2345);
    require_matches_legacy("H 0\nT 0\nM 0\n", 64, 3456);
    require_matches_legacy("H 0\nT 0\nM 0\nM 0\n", 64, 3567);
    require_matches_legacy("H 0\nT_DAG 0\nS 0\nM 0\n", 64, 4567);
    require_matches_legacy("H 0\nH 1\nT 0\nT 1\nCX 0 1\nM 0 1\n", 64, 5678);
}
