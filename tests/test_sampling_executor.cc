#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/svm/svm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
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
using clifft::sampling::ReplayResult;
using clifft::sampling::RotateActivePauli;
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
                      MeasureActivePauli{
                          {0, 1}, 0, SymbolId{0}, AffineBool::symbol(SymbolId{0}), RecordSlot{0}}},
        PlannedAction{
            0, 0,
            MeasureDormantRandom{1, SymbolId{1}, AffineBool::symbol(SymbolId{1}), RecordSlot{1}}},
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
        REQUIRE(std::ranges::equal(
            executor.visible_records(),
            std::span<const uint8_t>(legacy.meas_record).first(legacy_program.num_measurements)));
    }
}

void require_replay_matches_legacy(std::string_view circuit_text,
                                   std::span<const uint8_t> forced_records, size_t num_records) {
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit_text));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    REQUIRE(executable.num_hidden_records() == 0);
    REQUIRE(executable.num_visible_records() > 0);
    REQUIRE(forced_records.size() == num_records * executable.num_visible_records());

    const clifft::CompiledModule legacy_program = clifft::lower(hir);
    const std::vector<double> legacy =
        clifft::record_probabilities(legacy_program, forced_records, num_records);
    REQUIRE(legacy.size() == num_records);

    Executor executor(executable);
    const size_t stride = executable.num_visible_records();
    for (size_t i = 0; i < num_records; ++i) {
        const std::span<const uint8_t> record = forced_records.subspan(i * stride, stride);
        const ReplayResult replay = executor.replay_shot(record);
        CAPTURE(circuit_text, i, record);
        if (legacy[i] == clifft::kUnreachableLogProb) {
            REQUIRE_FALSE(replay.reachable);
        } else {
            REQUIRE(replay.reachable);
            REQUIRE_THAT(replay.log_probability, Catch::Matchers::WithinAbs(legacy[i], 1e-12));
            REQUIRE(std::ranges::equal(executor.visible_records(), record));
        }
    }
}

SamplingPlan plan_from(std::string_view circuit_text) {
    return clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit_text)));
}

bool has_multi_coordinate_active_measurement(const SamplingPlan& plan) {
    return std::ranges::any_of(plan.actions, [](const PlannedAction& action) {
        const auto* measurement = std::get_if<MeasureActivePauli>(&action.action);
        return measurement != nullptr &&
               std::popcount(measurement->pauli.x | measurement->pauli.z) > 1;
    });
}

bool has_arbitrary_angle_active_rotation(const SamplingPlan& plan, double half_turns) {
    return std::ranges::any_of(plan.actions, [half_turns](const PlannedAction& action) {
        if (const auto* rotation = std::get_if<RotateActivePauli>(&action.action)) {
            return rotation->half_turns == half_turns;
        }
        return false;
    });
}

}  // namespace

TEST_CASE("Sampling executor classifies measurement dust without drawing") {
    const auto zero = classify_measurement_branch(MeasurementProbabilities{1.0, 0.0});
    REQUIRE(zero.kind == MeasurementBranchKind::DeterministicZero);
    REQUIRE_FALSE(zero.clamped_dust);

    const auto dusty_zero = classify_measurement_branch(MeasurementProbabilities{1.0, 1e-30});
    REQUIRE(dusty_zero.kind == MeasurementBranchKind::DeterministicZero);
    REQUIRE(dusty_zero.clamped_dust);

    const auto dusty_one = classify_measurement_branch(MeasurementProbabilities{1e-30, 1.0});
    REQUIRE(dusty_one.kind == MeasurementBranchKind::DeterministicOne);
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
        PlannedAction{
            0, 0, DefineSymbol{SymbolId{1}, AffineBool(true, std::vector<SymbolId>{SymbolId{0}})}},
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(SymbolId{1}), RecordSlot{0}}},
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(SymbolId{0}), RecordSlot{1}}},
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

TEST_CASE("Sampling executor applies sampled symbols to later state actions") {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.max_active_width = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
    plan.actions = {
        PlannedAction{
            0, 0,
            MeasureDormantRandom{0, SymbolId{0}, AffineBool::symbol(SymbolId{0}), RecordSlot{0}}},
        PlannedAction{0, 1, PromoteDormantRotation{0.5, AffineBool::symbol(SymbolId{0})}},
        PlannedAction{
            1, 1,
            clifft::sampling::RotateActivePauli{{0, 1}, 0.5, AffineBool::symbol(SymbolId{0})}},
    };

    const ExecutablePlan executable(plan);
    Executor executor(executable, 42);
    bool saw_zero = false;
    bool saw_one = false;
    for (uint32_t shot = 0; shot < 64; ++shot) {
        executor.run_shot();
        const bool branch = executor.visible_records()[0] != 0;
        saw_zero |= !branch;
        saw_one |= branch;
        const double expected_imag = branch ? 0.5 : -0.5;
        for (uint64_t basis = 0; basis < executor.state().size(); ++basis) {
            CAPTURE(shot, branch, basis);
            REQUIRE_THAT(executor.state().real_data()[basis],
                         Catch::Matchers::WithinAbs(0.5, 1e-12));
            REQUIRE_THAT(executor.state().imag_data()[basis],
                         Catch::Matchers::WithinAbs(expected_imag, 1e-12));
        }
    }
    REQUIRE(saw_zero);
    REQUIRE(saw_one);
}

TEST_CASE("Sampling replay inverts affine records and preserves branch dependencies") {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.max_active_width = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
    plan.actions = {
        PlannedAction{
            0, 0,
            MeasureDormantRandom{0, SymbolId{0}, AffineBool(true, {SymbolId{0}}), RecordSlot{0}}},
        PlannedAction{0, 1, PromoteDormantRotation{0.5, AffineBool::symbol(SymbolId{0})}},
        PlannedAction{1, 1, RotateActivePauli{{0, 1}, 0.5, AffineBool::symbol(SymbolId{0})}},
    };

    const ExecutablePlan executable(plan);
    Executor executor(executable);
    for (uint8_t forced_record : {uint8_t{0}, uint8_t{1}}) {
        const ReplayResult replay = executor.replay_shot(std::array<uint8_t, 1>{forced_record});
        const bool branch = forced_record == 0;
        CAPTURE(forced_record, branch);
        REQUIRE(replay.reachable);
        REQUIRE_THAT(replay.log_probability, Catch::Matchers::WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(executor.visible_records()[0] == forced_record);
        REQUIRE(executor.symbols()[0] == branch);
        const double expected_imag = branch ? 0.5 : -0.5;
        for (uint64_t basis = 0; basis < executor.state().size(); ++basis) {
            REQUIRE_THAT(executor.state().real_data()[basis],
                         Catch::Matchers::WithinAbs(0.5, 1e-12));
            REQUIRE_THAT(executor.state().imag_data()[basis],
                         Catch::Matchers::WithinAbs(expected_imag, 1e-12));
        }
    }
    for (uint32_t shot = 0; shot < 16; ++shot) {
        executor.run_shot();
        REQUIRE(executor.visible_records()[0] == (executor.symbols()[0] ^ 1U));
    }
}

TEST_CASE("Sampling replay checks all records conditional on presampled symbols") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt}};
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(SymbolId{0}), RecordSlot{0}}},
        PlannedAction{0, 0, RecordClassical{AffineBool(true), RecordSlot{1}}},
    };

    const ExecutablePlan executable(plan);
    Executor executor(executable);
    const ReplayResult matching =
        executor.replay_shot(std::array<uint8_t, 2>{1, 1}, std::array<uint8_t, 1>{1});
    REQUIRE(matching.reachable);
    REQUIRE(matching.log_probability == 0.0);
    REQUIRE(executor.visible_records()[0] == 1);
    REQUIRE(executor.hidden_records()[0] == 1);

    const ReplayResult mismatching =
        executor.replay_shot(std::array<uint8_t, 2>{0, 1}, std::array<uint8_t, 1>{1});
    REQUIRE_FALSE(mismatching.reachable);
    REQUIRE(executor.visible_records()[0] == 1);
    // The hidden write follows the inconsistent visible record and is skipped.
    REQUIRE(executor.hidden_records()[0] == 0);
}

TEST_CASE("Sampling replay applies active measurement dust policy") {
    const ExecutablePlan dusty(active_then_dormant_plan(1e-10));
    Executor survivor(dusty);
    const ReplayResult survivor_result = survivor.replay_shot(std::array<uint8_t, 2>{0, 1});
    REQUIRE(survivor_result.reachable);
    REQUIRE_THAT(survivor_result.log_probability, Catch::Matchers::WithinAbs(std::log(0.5), 1e-15));
    REQUIRE(survivor.dust_clamps() == 1);

    Executor dust_branch(dusty);
    const ReplayResult dust_result = dust_branch.replay_shot(std::array<uint8_t, 2>{1, 0});
    REQUIRE_FALSE(dust_result.reachable);
    REQUIRE(dust_branch.dust_clamps() == 1);

    const ExecutablePlan exact(active_then_dormant_plan(0.0));
    Executor impossible_exact(exact);
    const ReplayResult exact_result = impossible_exact.replay_shot(std::array<uint8_t, 2>{1, 0});
    REQUIRE_FALSE(exact_result.reachable);
    REQUIRE(impossible_exact.dust_clamps() == 0);
}

TEST_CASE("Sampling replay accumulates active and dormant log probabilities") {
    SamplingPlan plan = active_then_dormant_plan(0.5);
    std::get<MeasureActivePauli>(plan.actions[1].action).outcome ^= true;
    const ExecutablePlan executable(plan);
    Executor executor(executable);
    const ReplayResult replay = executor.replay_shot(std::array<uint8_t, 2>{0, 1});
    REQUIRE(replay.reachable);
    REQUIRE_THAT(replay.log_probability, Catch::Matchers::WithinAbs(std::log(0.25), 1e-15));
    REQUIRE(executor.symbols()[0] == 1);
    REQUIRE(executor.visible_records()[0] == 0);
}

TEST_CASE("Sampling replay does not advance measurement RNG") {
    const ExecutablePlan executable(active_then_dormant_plan(0.0));
    Executor executor(executable, 456);
    const ReplayResult replay = executor.replay_shot(std::array<uint8_t, 2>{0, 1});
    REQUIRE(replay.reachable);

    clifft::Xoshiro256PlusPlus expected_rng(456);
    const bool expected_dormant_branch = expected_rng.next_double() >= 0.5;
    executor.run_shot();
    REQUIRE(executor.visible_records()[1] == expected_dormant_branch);
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

    constexpr std::string_view kMultiCoordinateX = "H 0\nH 1\nT 0\nT 1\nMPP X0*X1\n";
    REQUIRE(has_multi_coordinate_active_measurement(plan_from(kMultiCoordinateX)));
    require_matches_legacy(kMultiCoordinateX, 64, 6789);

    constexpr std::string_view kMultiCoordinateYZ = "H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1\n";
    REQUIRE(has_multi_coordinate_active_measurement(plan_from(kMultiCoordinateYZ)));
    require_matches_legacy(kMultiCoordinateYZ, 64, 7890);

    constexpr std::string_view kArbitraryRotation = "H 0\nT 0\nR_Z(0.3) 0\nM 0\nM 0\n";
    REQUIRE(has_arbitrary_angle_active_rotation(plan_from(kArbitraryRotation), 0.3));
    require_matches_legacy(kArbitraryRotation, 64, 8901);
}

TEST_CASE("Sampling replay matches legacy record probabilities") {
    require_replay_matches_legacy("H 0\nM 0\nM 0\n", std::array<uint8_t, 4>{0, 0, 0, 1}, 2);
    require_replay_matches_legacy("H 0\nM 0\nM 0\n", std::array<uint8_t, 4>{1, 0, 1, 1}, 2);
    require_replay_matches_legacy("H 0\nT 0\nR_Z(0.3) 0\nM 0\n", std::array<uint8_t, 2>{0, 1}, 2);
    require_replay_matches_legacy("H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1\n",
                                  std::array<uint8_t, 2>{0, 1}, 2);
    require_replay_matches_legacy("H 0\nCX 0 1\nM 0 1\n",
                                  std::array<uint8_t, 8>{0, 0, 0, 1, 1, 0, 1, 1}, 4);
}

TEST_CASE("Sampling replay probabilities normalize for a small active circuit") {
    constexpr std::string_view kCircuit = "H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1\nM 0\n";
    const ExecutablePlan executable(plan_from(kCircuit));
    REQUIRE(executable.num_visible_records() == 2);
    Executor executor(executable);

    double total = 0.0;
    uint32_t reachable = 0;
    for (uint8_t first : {uint8_t{0}, uint8_t{1}}) {
        for (uint8_t second : {uint8_t{0}, uint8_t{1}}) {
            const ReplayResult replay = executor.replay_shot(std::array<uint8_t, 2>{first, second});
            if (replay.reachable) {
                total += std::exp(replay.log_probability);
                ++reachable;
            }
        }
    }
    REQUIRE(reachable >= 2);
    REQUIRE_THAT(total, Catch::Matchers::WithinAbs(1.0, 1e-12));
}
