#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include "instrument_test_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

using clifft::HirModule;
using clifft::MeasRecordIdx;
using clifft::NoiseSite;
using clifft::sampling::AffineBool;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::ApplyReadoutNoise;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentMode;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::plan_sampling;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordParity;
using clifft::sampling::RecordSlot;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingPlanOptions;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteDetector;
using clifft::sampling::WriteExpectationValue;
using clifft::sampling::WriteObservable;
using clifft::test::X;
using clifft::test::Z;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

template <typename T>
const T& action_as(const SamplingPlan& plan, size_t index) {
    return std::get<T>(plan.actions.at(index).action);
}

uint64_t fnv1a64(std::string_view text) {
    uint64_t digest = 14695981039346656037ULL;
    for (unsigned char byte : text) {
        digest ^= byte;
        digest *= 1099511628211ULL;
    }
    return digest;
}

SamplingPlanOptions source_map_options() {
    SamplingPlanOptions options;
    options.retain_source_map = true;
    return options;
}

}  // namespace

TEST_CASE("Sampling planner preserves empty module metadata") {
    HirModule hir(3, 0);

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.num_qubits == 3);
    REQUIRE(plan.initial_active_width == 0);
    REQUIRE(plan.peak_active_width == 0);
    REQUIRE(plan.actions.empty());
    REQUIRE(plan.symbols.empty());
}

TEST_CASE("Sampling planner promotes rotations and keeps later active support") {
    HirModule hir(1, 2);
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_phase_rotation(hir, X(0), 0, false, 0.5);

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.peak_active_width == 1);
    REQUIRE(plan.actions.size() == 2);
    const auto& promotion = action_as<PromoteDormantRotation>(plan, 0);
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

    REQUIRE(plan.peak_active_width == 2);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[1].action));
    const auto& rotation = action_as<RotateActivePauli>(plan, 2);
    REQUIRE(rotation.pauli.x == 3);
    REQUIRE(rotation.pauli.z == 0);
}

TEST_CASE("Sampling planner preserves high physical Pauli coordinates") {
    HirModule hir(129, 2);
    hir.append_tgate(false, [](clifft::MutablePauliMaskView slot) { slot.x().bit_set(128, true); });
    hir.append_phase_rotation(
        0.5, [](clifft::MutablePauliMaskView slot) { slot.x().bit_set(128, true); });

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.peak_active_width == 1);
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
    REQUIRE(plan.symbols.size() == 2);
    REQUIRE(plan.symbols[1] == SymbolKind::Unused);
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

TEST_CASE("Sampling planner correlates mixed active and dormant measurements") {
    HirModule hir(2, 3);
    hir.num_measurements = 2;
    clifft::test::append_tgate(hir, X(0), 0, false);
    clifft::test::append_measure(hir, X(0) | X(1), 0, false, MeasRecordIdx{0});
    clifft::test::append_measure(hir, X(0) | X(1), 0, false, MeasRecordIdx{1});

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 3);
    const auto& first = action_as<MeasureDormantRandom>(plan, 1);
    REQUIRE(plan.actions[1].active_before == 1);
    REQUIRE(plan.actions[1].active_after == 1);
    REQUIRE(first.branch == SymbolId{0});
    REQUIRE(first.outcome == AffineBool::symbol(SymbolId{0}));
    const auto& repeated = action_as<RecordClassical>(plan, 2);
    REQUIRE(repeated.outcome == first.outcome);
}

TEST_CASE("Sampling planner accepts traced rotation and measurement HIR") {
    const clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nT 0\nM 0\n"));

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.actions.size() == 2);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    REQUIRE(std::holds_alternative<MeasureActivePauli>(plan.actions[1].action));
    REQUIRE(plan.symbols.size() == 1);
}

TEST_CASE("Sampling planner retains source provenance only when requested") {
    const HirModule hir = clifft::trace(clifft::parse("H 0\nT 0\nM 0\nDETECTOR rec[-1]\n"));

    const SamplingPlan ordinary = plan_sampling(hir);
    REQUIRE_FALSE(ordinary.source_map.has_value());

    const SamplingPlan inspected = plan_sampling(hir, source_map_options());
    REQUIRE(inspected.source_map.has_value());
    REQUIRE(inspected.source_map->size() == inspected.actions.size());
    REQUIRE(std::ranges::equal(inspected.source_map->lines_for(0), std::array<uint32_t, 1>{2}));
    REQUIRE(std::ranges::equal(inspected.source_map->lines_for(1), std::array<uint32_t, 1>{3}));
    REQUIRE(std::ranges::equal(inspected.source_map->lines_for(2), std::array<uint32_t, 1>{4}));
    REQUIRE(inspected.inspect_action(0).find("PROMOTE_DORMANT") != std::string::npos);
    REQUIRE_THROWS_AS(inspected.inspect_action(inspected.actions.size()), std::out_of_range);
}

TEST_CASE("Sampling planner requires complete requested source provenance") {
    HirModule hir = clifft::trace(clifft::parse("T 0\nM 0\n"));
    hir.source_map.clear();

    REQUIRE_NOTHROW(plan_sampling(hir));
    REQUIRE_THROWS_AS(plan_sampling(hir, source_map_options()), std::invalid_argument);
}

TEST_CASE("Sampling planner combines source lines for one observable action") {
    const HirModule hir = clifft::trace(
        clifft::parse("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"));

    const SamplingPlan plan = plan_sampling(hir, source_map_options());

    REQUIRE(plan.actions.size() == 3);
    REQUIRE(plan.source_map.has_value());
    REQUIRE(std::holds_alternative<WriteObservable>(plan.actions.back().action));
    REQUIRE(std::ranges::equal(plan.source_map->lines_for(2), std::array<uint32_t, 2>{2, 4}));
}

TEST_CASE("Sampling planner maps both instrument actions to their source line") {
    const clifft::InstrumentTraceOptions options = clifft::test::source_dependent_jump_options();
    const HirModule hir = clifft::trace(clifft::parse("LEVEL_TRANSITION[jump] 0\n"), &options);

    const SamplingPlan plan = plan_sampling(hir, source_map_options());

    REQUIRE(plan.actions.size() == 2);
    REQUIRE(plan.source_map.has_value());
    REQUIRE(std::holds_alternative<ApplyInstrument>(plan.actions[0].action));
    REQUIRE(std::holds_alternative<InstrumentBoundary>(plan.actions[1].action));
    for (size_t action = 0; action < plan.actions.size(); ++action) {
        REQUIRE(std::ranges::equal(plan.source_map->lines_for(action), std::array<uint32_t, 1>{1}));
    }
}

TEST_CASE("Sampling planner fixes instrument source handling before execution") {
    auto plan_for = [](std::string_view circuit, bool neglect) {
        const clifft::InstrumentTraceOptions options =
            clifft::test::source_dependent_jump_options(neglect);
        return plan_sampling(clifft::trace(clifft::parse(circuit), &options));
    };
    auto instrument = [](const SamplingPlan& plan) -> const ApplyInstrument& {
        for (const auto& action : plan.actions) {
            if (const auto* result = std::get_if<ApplyInstrument>(&action.action)) {
                return *result;
            }
        }
        throw std::logic_error("test plan omitted its instrument action");
    };

    const SamplingPlan classical = plan_for("LEVEL_TRANSITION[jump] 0", false);
    REQUIRE(instrument(classical).mode == InstrumentMode::Classical);
    REQUIRE(std::holds_alternative<InstrumentBoundary>(classical.actions.back().action));

    const SamplingPlan activated = plan_for("H 0\nT 0\nH 1\nLEVEL_TRANSITION[jump] 1\nM 1", false);
    const ApplyInstrument& activated_instrument = instrument(activated);
    REQUIRE(activated_instrument.mode == InstrumentMode::Activate);
    REQUIRE(activated_instrument.source.x == 2);
    REQUIRE(activated_instrument.source.z == 0);
    REQUIRE(activated_instrument.sign == AffineBool(false));
    REQUIRE(activated_instrument.destination_flip.has_value());
    REQUIRE(activated.peak_active_width == 2);
    const auto& after_activation = action_as<MeasureActivePauli>(activated, 3);
    REQUIRE(after_activation.outcome ==
            (AffineBool::symbol(*activated_instrument.destination_flip) ^
             AffineBool::symbol(after_activation.branch)));

    const SamplingPlan active = plan_for("H 0\nT 0\nLEVEL_TRANSITION[jump] 0", false);
    REQUIRE(instrument(active).mode == InstrumentMode::Active);

    const SamplingPlan neglected = plan_for("H 0\nLEVEL_TRANSITION[jump] 0", true);
    REQUIRE(instrument(neglected).mode == InstrumentMode::DormantTrap);
    REQUIRE(neglected.peak_active_width == 0);
}

TEST_CASE("Sampling planner keeps prefix symbols stable across continuation suffixes") {
    const clifft::InstrumentTraceOptions options = clifft::test::source_dependent_jump_options();
    const auto make = [&](std::string_view suffix) {
        const std::string circuit = "H 0\nM 0\nLEVEL_TRANSITION[jump] 0\n" + std::string(suffix);
        return plan_sampling(clifft::trace(clifft::parse(circuit), &options));
    };
    const SamplingPlan short_plan = make("");
    const SamplingPlan longer_plan = make("X_ERROR(0.1) 0");

    const auto& short_measurement = action_as<MeasureDormantRandom>(short_plan, 0);
    const auto& longer_measurement = action_as<MeasureDormantRandom>(longer_plan, 0);
    REQUIRE(short_measurement.branch == longer_measurement.branch);

    const auto find_instrument = [](const SamplingPlan& plan) -> const ApplyInstrument& {
        for (const auto& action : plan.actions) {
            if (const auto* instrument = std::get_if<ApplyInstrument>(&action.action)) {
                return *instrument;
            }
        }
        throw std::logic_error("test plan omitted its instrument action");
    };
    REQUIRE(find_instrument(short_plan).destination_flip ==
            find_instrument(longer_plan).destination_flip);
    REQUIRE(longer_plan.symbols.size() == short_plan.symbols.size() + 1);
}

TEST_CASE("Sampling planner preserves mixed symbol order at instrument boundaries") {
    const clifft::InstrumentTraceOptions options = clifft::test::source_dependent_jump_options();
    const SamplingPlan plan = plan_sampling(clifft::trace(clifft::parse(R"(
        X_ERROR(0.1) 0
        H 0
        M 0
        READOUT_NOISE(0.1, 0.2) rec[-1]
        LEVEL_TRANSITION[jump] 0
        X_ERROR(0.3) 0
    )"),
                                                          &options));

    REQUIRE(plan.symbols.size() == 5);
    REQUIRE(plan.symbols[0] == SymbolKind::Presampled);
    REQUIRE(plan.symbols[1] == SymbolKind::Branch);
    REQUIRE(plan.symbols[2] == SymbolKind::Readout);
    REQUIRE(plan.symbols[3] == SymbolKind::Instrument);
    REQUIRE(plan.symbols[4] == SymbolKind::Presampled);

    const auto boundary = std::ranges::find_if(plan.actions, [](const auto& action) {
        return std::holds_alternative<InstrumentBoundary>(action.action);
    });
    REQUIRE(boundary != plan.actions.end());
    const InstrumentBoundary& instrument = std::get<InstrumentBoundary>(boundary->action);
    REQUIRE(instrument.next_noise_site == 1);
    REQUIRE(instrument.symbol_prefix_size == 4);
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
            if (first.pauli.x != 0) {
                REQUIRE(first.active_pivot == 63U - std::countl_zero(first.pauli.x));
            }
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

TEST_CASE("Sampling planner omits identity rotations") {
    HirModule hir(1, 2);
    clifft::test::append_phase_rotation(hir, 0, Z(0), false, 0.5);
    clifft::test::append_tgate(hir, 0, Z(0), false);

    const SamplingPlan plan = plan_sampling(hir);

    CHECK(plan.actions.empty());
}

TEST_CASE("Sampling planner omits signed identity rotations") {
    SECTION("constant sign") {
        HirModule hir(1, 1);
        clifft::test::append_tgate(hir, 0, Z(0), true);

        const SamplingPlan plan = plan_sampling(hir);

        CHECK(plan.actions.empty());
    }

    SECTION("symbolic sign") {
        const HirModule hir = clifft::trace(clifft::parse("X_ERROR(0.1) 0\nR_Z(0.3) 0"));

        const SamplingPlan plan = plan_sampling(hir);

        REQUIRE(plan.symbols.size() == 1);
        CHECK(plan.actions.empty());
    }
}

TEST_CASE("Sampling planner classifies active and dormant expectation probes") {
    const SamplingPlan plan = plan_sampling(clifft::trace(clifft::parse(R"(
        H 0
        T 0
        EXP_VAL X0
        EXP_VAL X1
        EXP_VAL Z1
    )")));

    REQUIRE(plan.num_exp_vals == 3);
    REQUIRE(plan.actions.size() == 4);
    REQUIRE(plan.final_tableau.has_value());
    const auto& active = action_as<WriteExpectationValue>(plan, 1);
    REQUIRE(active.active.has_value());
    REQUIRE_FALSE(active.active->projection.is_identity());
    REQUIRE(active.exp_val == clifft::sampling::ExpValSlot{0});
    const auto& dormant_x = action_as<WriteExpectationValue>(plan, 2);
    REQUIRE_FALSE(dormant_x.active.has_value());
    const auto& dormant_z = action_as<WriteExpectationValue>(plan, 3);
    REQUIRE(dormant_z.active.has_value());
    REQUIRE(dormant_z.active->projection.is_identity());
}

TEST_CASE("Sampling planner propagates stochastic signs into expectation probes") {
    const SamplingPlan noise =
        plan_sampling(clifft::trace(clifft::parse("X_ERROR(1) 0\nEXP_VAL Z0")));
    REQUIRE_FALSE(noise.final_tableau.has_value());
    const auto& noise_probe = action_as<WriteExpectationValue>(noise, 0);
    REQUIRE(noise_probe.active.has_value());
    REQUIRE(noise_probe.active->projection.is_identity());
    REQUIRE(noise_probe.active->sign == AffineBool::symbol(SymbolId{0}));

    const SamplingPlan feedback =
        plan_sampling(clifft::trace(clifft::parse("H 0\nM 0\nCX rec[-1] 1\nEXP_VAL Z1")));
    const auto& measurement = action_as<MeasureDormantRandom>(feedback, 0);
    const auto& feedback_probe = action_as<WriteExpectationValue>(feedback, 1);
    REQUIRE(feedback_probe.active.has_value());
    REQUIRE(feedback_probe.active->projection.is_identity());
    REQUIRE(feedback_probe.active->sign == AffineBool::symbol(measurement.branch));
}

TEST_CASE("Sampling planner eliminates Pauli noise and feedback into record expressions") {
    const HirModule hir = clifft::trace(clifft::parse(R"(
        X_ERROR(1) 0
        M 0
        CX rec[-1] 1
        M 1
    )"));

    const SamplingPlan plan = plan_sampling(hir);

    REQUIRE(plan.presampled_noise_sites.size() == 1);
    REQUIRE(plan.presampled_noise_sites[0].outcomes.size() == 1);
    const SymbolId noise = plan.presampled_noise_sites[0].outcomes[0].symbol;
    REQUIRE(plan.symbols[static_cast<uint32_t>(noise)] == SymbolKind::Presampled);
    REQUIRE(plan.actions.size() == 2);
    REQUIRE(action_as<RecordClassical>(plan, 0).outcome == AffineBool::symbol(noise));
    REQUIRE(action_as<RecordClassical>(plan, 1).outcome == AffineBool::symbol(noise));
}

TEST_CASE("Sampling planner carries symbolic frame across coordinate changes") {
    const SamplingPlan plan = plan_sampling(clifft::trace(clifft::parse(R"(
        H 0
        T 0
        X_ERROR(1) 1
        MPP X0*Z1
        CX rec[-1] 2
        H 2
        T 2
        MPP Y0*X2
        M 0 1 2
    )")));

    REQUIRE(plan.actions.size() == 7);
    REQUIRE(plan.symbols.size() == 6);
    const SymbolId noise = plan.presampled_noise_sites.at(0).outcomes.at(0).symbol;
    const auto& first_measurement = action_as<MeasureActivePauli>(plan, 1);
    REQUIRE(first_measurement.outcome ==
            (AffineBool::symbol(noise) ^ AffineBool::symbol(first_measurement.branch)));

    const auto& mixed_measurement = action_as<MeasureDormantRandom>(plan, 3);
    REQUIRE(mixed_measurement.outcome ==
            (AffineBool::symbol(noise) ^ AffineBool::symbol(mixed_measurement.branch) ^ true));

    const auto& first_final_measurement = action_as<MeasureDormantRandom>(plan, 4);
    REQUIRE(first_final_measurement.outcome ==
            (AffineBool::symbol(mixed_measurement.branch) ^
             AffineBool::symbol(first_final_measurement.branch)));
    REQUIRE(action_as<RecordClassical>(plan, 5).outcome == AffineBool::symbol(noise));

    const auto& last_measurement = action_as<MeasureActivePauli>(plan, 6);
    REQUIRE(last_measurement.outcome == (AffineBool::symbol(first_final_measurement.branch) ^
                                         AffineBool::symbol(last_measurement.branch)));
}

TEST_CASE("Sampling planner carries corrected records into syndrome outputs") {
    const HirModule hir = clifft::trace(clifft::parse(R"(
        M 0
        READOUT_NOISE(0.1, 0.2) rec[-1]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    const std::array<uint8_t, 1> expected_detectors{1};
    const std::array<uint8_t, 1> expected_observables{1};

    const SamplingPlan plan =
        plan_sampling(hir, {postselection, expected_detectors, expected_observables});

    REQUIRE(plan.num_detectors == 1);
    REQUIRE(plan.num_observables == 1);
    REQUIRE(plan.actions.size() == 4);
    const auto& readout = action_as<ApplyReadoutNoise>(plan, 1);
    REQUIRE(plan.symbols[static_cast<uint32_t>(readout.flip)] == SymbolKind::Readout);
    REQUIRE(readout.source == AffineBool(false));
    REQUIRE(readout.prob_zero_to_one == 0.1);
    REQUIRE(readout.prob_one_to_zero == 0.2);
    const auto& detector = action_as<WriteDetector>(plan, 2);
    REQUIRE(detector.postselected);
    const auto& detector_parity = detector.outcome;
    REQUIRE(detector_parity.constant());
    REQUIRE(detector_parity.records() == std::vector<RecordSlot>{RecordSlot{0}});
    const auto& observable = action_as<WriteObservable>(plan, 3);
    REQUIRE(std::holds_alternative<RecordParity>(observable.outcome));
    const auto& observable_parity = std::get<RecordParity>(observable.outcome);
    REQUIRE(observable_parity.constant());
    REQUIRE(observable_parity.records() == std::vector<RecordSlot>{RecordSlot{0}});
}

TEST_CASE("Sampling planner falls back for historical observable records") {
    const SamplingPlan plan = plan_sampling(clifft::trace(clifft::parse(R"(
        X 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-1]
        READOUT_NOISE(1) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-1]
    )")));

    REQUIRE(plan.actions.size() == 5);
    const auto& readout = action_as<ApplyReadoutNoise>(plan, 1);
    const auto& before = action_as<WriteObservable>(plan, 2);
    const auto& after = action_as<WriteObservable>(plan, 3);
    const auto& straddled = action_as<WriteObservable>(plan, 4);
    REQUIRE(std::get<AffineBool>(before.outcome) == AffineBool(true));
    REQUIRE(std::holds_alternative<RecordParity>(after.outcome));
    REQUIRE(std::get<RecordParity>(after.outcome).records() ==
            std::vector<RecordSlot>{RecordSlot{0}});
    REQUIRE(std::get<AffineBool>(straddled.outcome) == AffineBool::symbol(readout.flip));
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

TEST_CASE("Sampling planner does not activate absorbed Pauli rotations") {
    for (const char* source : {
             "R_ZZ(1.0) 0 1",
             "R_PAULI(-1.0) X0*Y1",
             "R_ZZ(0.5) 0 1",
             "R_PAULI(-0.5) X0*Y1",
             "SPP X0*Y1",
             "R_Z(0.3) 0\nR_Z(0.7) 0",
         }) {
        CAPTURE(source);
        HirModule hir = clifft::trace(clifft::parse(source));
        auto passes = clifft::default_hir_pass_manager();
        passes.run(hir);
        const SamplingPlan plan = plan_sampling(hir);

        CHECK(hir.ops.empty());
        CHECK(plan.initial_active_width == 0);
        CHECK(plan.peak_active_width == 0);
        CHECK(plan.actions.empty());
    }
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

TEST_CASE("Sampling planner target QEC plan characterization") {
    clifft::Circuit circuit =
        clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/target_qec.stim");
    HirModule hir = clifft::trace(circuit);
    auto passes = clifft::default_hir_pass_manager();
    passes.run(hir);

    const SamplingPlan plan = plan_sampling(hir);
    REQUIRE(plan.actions.size() == 91);
    REQUIRE(plan.symbols.size() == 2061);
    // The inspection includes every action field and affine expression, making
    // the production fixture a compact end-to-end planner characterization.
    const std::string inspection = plan.inspect();
    INFO(inspection);
    // After verifying that a reported inspection change is intentional, update
    // this digest to the new value shown by the failed assertion.
    REQUIRE(fnv1a64(inspection) == 0xad863e67839d8f4fULL);
}
