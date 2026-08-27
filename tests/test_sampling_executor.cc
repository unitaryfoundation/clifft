#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/util/intra_shot_parallel.h"
#include "clifft/util/noise_sampling.h"
#include "clifft/util/shot_seed.h"
#include "clifft/util/xoshiro.h"

#include "instrument_test_helpers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using clifft::sampling::ActiveExpectation;
using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::apply_rotation;
using clifft::sampling::ApplyInstrument;
using clifft::sampling::DefineSymbol;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::ExpValSlot;
using clifft::sampling::ForcedTraceOut;
using clifft::sampling::index;
using clifft::sampling::InstrumentBoundary;
using clifft::sampling::InstrumentDistribution;
using clifft::sampling::InstrumentMode;
using clifft::sampling::InstrumentSiteId;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::MeasureDormantRandom;
using clifft::sampling::PlannedAction;
using clifft::sampling::prepare_rotation;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::record_log_probabilities;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordSlot;
using clifft::sampling::ReplayResult;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::sample_records;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingPlanOptions;
using clifft::sampling::State;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteExpectationValue;

namespace {

SamplingPlan active_then_dormant_plan(double promotion_half_turns) {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.peak_active_width = 1;
    plan.num_visible_records = 2;
    plan.symbols = {SymbolKind::Branch, SymbolKind::Branch};
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

SamplingPlan dormant_trap_plan() {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.symbols = {SymbolKind::Unused};
    plan.instrument_distributions = {InstrumentDistribution{{1.0, 1.0}, {}}};
    plan.actions = {
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
    };
    return plan;
}

SamplingPlan plan_from(std::string_view circuit_text) {
    return clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit_text)));
}

SamplingPlan categorical_noise_plan() {
    SamplingPlan plan;
    plan.symbols = {SymbolKind::Presampled, SymbolKind::Presampled, SymbolKind::Presampled,
                    SymbolKind::Presampled};
    plan.presampled_noise_sites = {
        PresampledNoiseSite{0.05, {PresampledNoiseOutcome{SymbolId{0}, 0.05}}},
        PresampledNoiseSite{
            0.3,
            {PresampledNoiseOutcome{SymbolId{1}, 0.1}, PresampledNoiseOutcome{SymbolId{2}, 0.2}}},
        PresampledNoiseSite{0.0, {}},
        PresampledNoiseSite{0.4, {PresampledNoiseOutcome{SymbolId{3}, 0.4}}},
    };
    return plan;
}

std::vector<double> noise_hazards(const SamplingPlan& plan) {
    std::vector<double> result;
    result.reserve(plan.presampled_noise_sites.size());
    double cumulative = 0.0;
    for (const PresampledNoiseSite& site : plan.presampled_noise_sites) {
        double probability = 0.0;
        for (const PresampledNoiseOutcome& outcome : site.outcomes) {
            probability += outcome.probability;
        }
        cumulative += clifft::bernoulli_hazard(probability);
        result.push_back(cumulative);
    }
    return result;
}

std::vector<uint8_t> sample_reference_noise(const SamplingPlan& plan,
                                            std::span<const double> hazards,
                                            clifft::Xoshiro256PlusPlus& rng) {
    std::vector<uint8_t> result(plan.symbols.size(), 0);
    uint32_t first_candidate = 0;
    while (first_candidate < plan.presampled_noise_sites.size()) {
        const double current_hazard = first_candidate == 0 ? 0.0 : hazards[first_candidate - 1];
        if (current_hazard >= hazards.back()) {
            break;
        }
        const uint32_t site_index =
            clifft::sample_next_noise_site(hazards, first_candidate, rng.next_double());
        if (site_index == clifft::kNoNoiseSite) {
            break;
        }
        const PresampledNoiseSite& site = plan.presampled_noise_sites[site_index];
        REQUIRE_FALSE(site.outcomes.empty());
        size_t outcome_index = 0;
        if (site.outcomes.size() > 1) {
            double total_probability = 0.0;
            for (const PresampledNoiseOutcome& outcome : site.outcomes) {
                total_probability += outcome.probability;
            }
            const double channel_draw = rng.next_double() * total_probability;
            double cumulative = 0.0;
            for (size_t i = 0; i < site.outcomes.size(); ++i) {
                cumulative += site.outcomes[i].probability;
                if (channel_draw < cumulative) {
                    outcome_index = i;
                    break;
                }
            }
        }
        result[index(site.outcomes[outcome_index].symbol)] = 1;
        first_candidate = site_index + 1;
    }
    return result;
}

}  // namespace

TEST_CASE("Noise hazard sampler skips silent sites") {
    const double first = clifft::bernoulli_hazard(0.25);
    const std::array<double, 3> hazards{
        first,
        first,
        first + clifft::bernoulli_hazard(0.5),
    };

    REQUIRE(clifft::sample_next_noise_site(hazards, 0, 0.0) == 0);
    REQUIRE(clifft::sample_next_noise_site(hazards, 1, 0.0) == 2);
    REQUIRE(clifft::sample_next_noise_site(hazards, 0, 0.3) == 2);
    REQUIRE(clifft::sample_next_noise_site(hazards, 0, 0.9) == clifft::kNoNoiseSite);
}

TEST_CASE("Sampling executor skips silent noise sites deterministically") {
    constexpr uint64_t seed = 0x123456789abcdef0ULL;
    const SamplingPlan plan = categorical_noise_plan();
    const std::vector<double> hazards = noise_hazards(plan);
    const ExecutablePlan executable(plan);
    Executor executor(executable, seed);
    clifft::Xoshiro256PlusPlus reference_rng(seed);

    for (uint32_t shot = 0; shot < 128; ++shot) {
        const std::vector<uint8_t> expected = sample_reference_noise(plan, hazards, reference_rng);
        executor.run_shot();
        CAPTURE(shot, expected, executor.symbols());
        REQUIRE(std::ranges::equal(executor.symbols(), expected));
    }

    // Alternating supplied and sampled inputs catches stale one bits without
    // requiring every presampled symbol to be cleared on every shot.
    const std::array<uint8_t, 4> all_ones{1, 1, 1, 1};
    executor.run_shot(all_ones);
    REQUIRE(std::ranges::equal(executor.symbols(), all_ones));

    const std::vector<uint8_t> expected = sample_reference_noise(plan, hazards, reference_rng);
    executor.run_shot();
    REQUIRE(std::ranges::equal(executor.symbols(), expected));

    const std::array<uint8_t, 4> all_zeroes{0, 0, 0, 0};
    executor.run_shot(all_zeroes);
    REQUIRE(std::ranges::equal(executor.symbols(), all_zeroes));
}

TEST_CASE("Sampling executor does not draw for empty noise sites") {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.num_visible_records = 1;
    plan.presampled_noise_sites = {PresampledNoiseSite{0.0, {}}};
    plan.symbols = {SymbolKind::Branch};
    plan.actions = {PlannedAction{
        0, 0,
        MeasureDormantRandom{0, SymbolId{0}, AffineBool::symbol(SymbolId{0}), RecordSlot{0}}}};

    constexpr uint64_t seed = 1234;
    const ExecutablePlan executable(plan);
    Executor executor(executable, seed);
    clifft::Xoshiro256PlusPlus reference_rng(seed);
    for (uint32_t shot = 0; shot < 32; ++shot) {
        const uint8_t expected = static_cast<uint8_t>(reference_rng.next_double() >= 0.5);
        executor.run_shot();
        CAPTURE(shot);
        REQUIRE(executor.visible_records()[0] == expected);
    }
}

TEST_CASE("Sampling executor evaluates presampled and derived affine symbols") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.symbols = {SymbolKind::Presampled, SymbolKind::Derived};
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
    plan.peak_active_width = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolKind::Branch};
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

TEST_CASE("Sampling executor fuses constant rotation orbits") {
    auto require_matches_scalar = [](uint32_t active_width,
                                     std::span<const RotateActivePauli> rotations) {
        SamplingPlan plan;
        plan.num_qubits = active_width;
        plan.initial_active_width = active_width;
        plan.peak_active_width = active_width;
        plan.symbols = {SymbolKind::Presampled};
        for (uint32_t axis = 0; axis < active_width; ++axis) {
            plan.actions.push_back(
                PlannedAction{active_width, active_width,
                              RotateActivePauli{ActivePauli{uint64_t{1} << axis, 0}, 0.5,
                                                AffineBool::symbol(SymbolId{0})}});
        }
        for (const RotateActivePauli& rotation : rotations) {
            plan.actions.push_back(PlannedAction{active_width, active_width, rotation});
        }

        const ExecutablePlan executable(plan);
        REQUIRE(executable.num_actions() == active_width + 1);
        Executor executor(executable);
        executor.run_shot(std::array<uint8_t, 1>{0});

        State expected(active_width, active_width);
        for (uint32_t axis = 0; axis < active_width; ++axis) {
            apply_rotation(expected,
                           prepare_rotation(ActivePauli{uint64_t{1} << axis, 0}, active_width, 0.5),
                           false);
        }
        for (const RotateActivePauli& rotation : rotations) {
            apply_rotation(expected,
                           prepare_rotation(rotation.pauli, active_width, rotation.half_turns),
                           rotation.sign.constant());
        }

        REQUIRE(executor.state().size() == expected.size());
        for (uint64_t basis = 0; basis < expected.size(); ++basis) {
            CAPTURE(active_width, basis);
            REQUIRE_THAT(executor.state().real_data()[basis],
                         Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
            REQUIRE_THAT(executor.state().imag_data()[basis],
                         Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
        }
    };

    const std::array<RotateActivePauli, 4> rank_one = {
        RotateActivePauli{{0b101, 0b100}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b101, 0b101}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b000, 0b110}, 0.4, AffineBool(false)},
        RotateActivePauli{{0b101, 0b111}, 0.1, AffineBool(false)},
    };
    require_matches_scalar(3, rank_one);

    const std::array<RotateActivePauli, 3> rank_zero = {
        RotateActivePauli{{0b000, 0b001}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b000, 0b010}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b000, 0b111}, 0.4, AffineBool(false)},
    };
    require_matches_scalar(3, rank_zero);

    const std::array<RotateActivePauli, 5> rank_two = {
        RotateActivePauli{{0b00101, 0b00100}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b11010, 0b01000}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b11111, 0b11101}, 0.4, AffineBool(false)},
        RotateActivePauli{{0b00101, 0b11110}, 0.1, AffineBool(false)},
        RotateActivePauli{{0b11010, 0b10111}, -0.2, AffineBool(true)},
    };
    require_matches_scalar(5, rank_two);

    const std::array<RotateActivePauli, 5> max_selectors = {
        RotateActivePauli{{0, 0b00001}, 0.25, AffineBool(false)},
        RotateActivePauli{{0, 0b00010}, -0.3, AffineBool(true)},
        RotateActivePauli{{0, 0b00100}, 0.4, AffineBool(false)},
        RotateActivePauli{{0, 0b01000}, 0.1, AffineBool(false)},
        RotateActivePauli{{0, 0b10000}, -0.2, AffineBool(true)},
    };
    require_matches_scalar(5, max_selectors);

    const std::array<RotateActivePauli, 5> rank_two_high_pivots = {
        RotateActivePauli{{0b001011, 0b100101}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b101101, 0b011010}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b100110, 0b110001}, 0.4, AffineBool(false)},
        RotateActivePauli{{0b001011, 0b001111}, 0.1, AffineBool(false)},
        RotateActivePauli{{0b101101, 0b101100}, -0.2, AffineBool(true)},
    };
    require_matches_scalar(6, rank_two_high_pivots);

    SamplingPlan wide_selector_plan;
    wide_selector_plan.num_qubits = 6;
    wide_selector_plan.initial_active_width = 6;
    wide_selector_plan.peak_active_width = 6;
    for (uint32_t axis = 0; axis < 6; ++axis) {
        wide_selector_plan.actions.push_back(PlannedAction{
            6, 6, RotateActivePauli{{0, uint64_t{1} << axis}, 0.25, AffineBool(false)}});
    }
    REQUIRE(ExecutablePlan(wide_selector_plan).num_actions() == 6);
}

TEST_CASE("Sampling replay inverts affine records and preserves branch dependencies") {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.peak_active_width = 1;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolKind::Branch};
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

TEST_CASE("Sampling replay evaluates record-dependent expectations") {
    const ExecutablePlan executable(plan_from("H 0\nM 0\nCX rec[-1] 1\nEXP_VAL Z1\n"));
    Executor executor(executable);

    for (uint8_t forced_record : {uint8_t{0}, uint8_t{1}}) {
        const ReplayResult replay = executor.replay_shot(std::array{forced_record});
        CAPTURE(forced_record);
        REQUIRE(replay.reachable);
        REQUIRE_THAT(replay.log_probability, Catch::Matchers::WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(executor.exp_vals()[0] == (forced_record == 0 ? 1.0 : -1.0));
    }
}

TEST_CASE("Sampling basis probabilities reject nonunitary plans") {
    const ExecutablePlan measured(plan_from("M 0\n"));
    REQUIRE_FALSE(measured.supports_final_state_queries());
    REQUIRE_THROWS_AS(
        clifft::sampling::basis_probabilities(measured, std::array<uint64_t, 1>{0}, 1, 1),
        std::invalid_argument);

    const ExecutablePlan with_probe(plan_from("H 0\nEXP_VAL X0\n"));
    REQUIRE(with_probe.supports_final_state_queries());
    const std::vector<double> probabilities =
        clifft::sampling::basis_probabilities(with_probe, std::array<uint64_t, 2>{0, 1}, 2, 1);
    REQUIRE_THAT(probabilities[0], Catch::Matchers::WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(probabilities[1], Catch::Matchers::WithinAbs(0.5, 1e-12));
}

TEST_CASE("Sampling basis probabilities conjugate a complex Clifford frame") {
    // The frame H*S*H mixes the two active coordinates with weights (1 +- i)/2
    // while the T rotation leaves them relatively imaginary. The query expands
    // the state through the inverse frame, so dropping the conjugate of that
    // expansion inverts the interference and swaps the outcome weights.
    const double p_low = (2.0 - std::numbers::sqrt2) / 4.0;
    const double p_high = (2.0 + std::numbers::sqrt2) / 4.0;

    SECTION("every dormant column is a pivot") {
        const ExecutablePlan executable(plan_from("H 0\nS 0\nT 0\nH 0\n"));
        const std::vector<double> probabilities =
            clifft::sampling::basis_probabilities(executable, std::array<uint64_t, 2>{0, 1}, 2, 1);
        REQUIRE_THAT(probabilities[0], Catch::Matchers::WithinAbs(p_low, 1e-12));
        REQUIRE_THAT(probabilities[1], Catch::Matchers::WithinAbs(p_high, 1e-12));
    }

    SECTION("a free dormant column disables the gray-code walk") {
        // The trailing CX correlates the dormant qubit with the active
        // coordinate, so the dormant column has no X pivot and the query takes
        // the explicit per-index walk instead of the gray-code path.
        const ExecutablePlan executable(plan_from("H 0\nS 0\nT 0\nH 0\nCX 0 1\n"));
        const std::vector<double> probabilities =
            clifft::sampling::basis_probabilities(executable, std::array<uint64_t, 2>{0, 3}, 2, 1);
        REQUIRE_THAT(probabilities[0], Catch::Matchers::WithinAbs(p_low, 1e-12));
        REQUIRE_THAT(probabilities[1], Catch::Matchers::WithinAbs(p_high, 1e-12));
    }
}

TEST_CASE("Sampling statevectors reject nonunitary and oversized plans") {
    const ExecutablePlan measured(plan_from("H 0\nM 0\n"));
    REQUIRE_THROWS_AS(clifft::sampling::get_statevector(measured), std::invalid_argument);

    const ExecutablePlan oversized(plan_from("H 10\n"));
    REQUIRE_THROWS_AS(clifft::sampling::get_statevector(oversized), std::runtime_error);

    const ExecutablePlan with_probe(plan_from("H 0\nEXP_VAL X0\n"));
    const std::vector<std::complex<double>> statevector =
        clifft::sampling::get_statevector(with_probe);
    REQUIRE(statevector.size() == 2);
    const double expected = 1.0 / std::numbers::sqrt2;
    REQUIRE_THAT(statevector[0].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
    REQUIRE_THAT(statevector[1].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
}

TEST_CASE("Sampling replay checks all records conditional on presampled symbols") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.num_hidden_records = 1;
    plan.symbols = {SymbolKind::Presampled};
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
    // Only the executed prefix is meaningful after an unreachable replay.
    REQUIRE(executor.visible_records()[0] == 1);
}

TEST_CASE("Sampling expression registers reset true symbols between shots") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolKind::Presampled};
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(SymbolId{0}), RecordSlot{0}}},
    };

    const ExecutablePlan executable(plan);
    Executor executor(executable);
    executor.run_shot(std::array<uint8_t, 1>{1});
    REQUIRE(executor.visible_records()[0] == 1);
    executor.run_shot(std::array<uint8_t, 1>{0});
    REQUIRE(executor.visible_records()[0] == 0);
    executor.run_shot(std::array<uint8_t, 1>{1});
    REQUIRE(executor.visible_records()[0] == 1);
}

TEST_CASE("Sampling expression registers preserve noisy postselection") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        X_ERROR(0.5) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection,
                                              .expected_detectors = {},
                                              .expected_observables = {}}));

    const clifft::sampling::SamplingSurvivorResult result =
        clifft::sampling::sample_survivors(executable, 2000, uint64_t{280}, true);
    REQUIRE(result.passed_shots > 900);
    REQUIRE(result.passed_shots < 1100);
    REQUIRE(std::ranges::all_of(result.measurements, [](uint8_t value) { return value == 0; }));
    REQUIRE(std::ranges::all_of(result.detectors, [](uint8_t value) { return value == 0; }));
    REQUIRE(std::ranges::all_of(result.observables, [](uint8_t value) { return value == 0; }));
}

TEST_CASE("Sampling replay applies active measurement dust policy") {
    SECTION("dust on the one branch") {
        const ExecutablePlan dusty(active_then_dormant_plan(1e-10));
        Executor survivor(dusty);
        const ReplayResult survivor_result = survivor.replay_shot(std::array<uint8_t, 2>{0, 1});
        REQUIRE(survivor_result.reachable);
        REQUIRE_THAT(survivor_result.log_probability,
                     Catch::Matchers::WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(survivor.dust_clamps() == 1);

        Executor dust_branch(dusty);
        const ReplayResult dust_result = dust_branch.replay_shot(std::array<uint8_t, 2>{1, 0});
        REQUIRE_FALSE(dust_result.reachable);
        REQUIRE(dust_branch.dust_clamps() == 1);

        const ExecutablePlan exact(active_then_dormant_plan(0.0));
        Executor impossible_exact(exact);
        const ReplayResult exact_result =
            impossible_exact.replay_shot(std::array<uint8_t, 2>{1, 0});
        REQUIRE_FALSE(exact_result.reachable);
        REQUIRE(impossible_exact.dust_clamps() == 0);
    }

    SECTION("dust on the zero branch") {
        const ExecutablePlan dusty(active_then_dormant_plan(1.0 - 1e-10));
        Executor survivor(dusty);
        const ReplayResult survivor_result = survivor.replay_shot(std::array<uint8_t, 2>{1, 1});
        REQUIRE(survivor_result.reachable);
        REQUIRE_THAT(survivor_result.log_probability,
                     Catch::Matchers::WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(survivor.dust_clamps() == 1);

        Executor dust_branch(dusty);
        const ReplayResult dust_result = dust_branch.replay_shot(std::array<uint8_t, 2>{0, 0});
        REQUIRE_FALSE(dust_result.reachable);
        REQUIRE(dust_branch.dust_clamps() == 1);
    }
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

TEST_CASE("Sampling executor clamps active measurement dust to branch one") {
    const ExecutablePlan executable(active_then_dormant_plan(1.0 - 1e-10));
    Executor executor(executable, 456);
    clifft::Xoshiro256PlusPlus expected_rng(456);
    const bool expected_dormant_branch = expected_rng.next_double() >= 0.5;
    const bool branch_if_active_had_drawn = expected_rng.next_double() >= 0.5;
    REQUIRE(expected_dormant_branch != branch_if_active_had_drawn);

    executor.run_shot();

    REQUIRE(executor.visible_records()[0] == 1);
    REQUIRE(executor.visible_records()[1] == expected_dormant_branch);
    REQUIRE(executor.dust_clamps() == 1);
}

TEST_CASE("Sampling executor receives no identity rotation actions") {
    clifft::HirModule hir(1, 1);
    hir.append_tgate(false, [](clifft::MutablePauliMaskView slot) {
        slot.z().bit_set(0, true);
        slot.set_sign(true);
    });
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    REQUIRE(executable.num_actions() == 0);
    Executor executor(executable);

    executor.run_shot();
}

TEST_CASE("Sampling identity rotation elision preserves active rotation fusion") {
    const SamplingPlan plan = plan_from(R"(
        R_X(0.1) 0
        R_Z(0.2) 1
        R_Z(0.2) 0
        R_Z(0.3) 2
        R_Y(0.3) 0
        R_X(0.4) 0
    )");

    REQUIRE(plan.actions.size() == 4);
    REQUIRE(std::holds_alternative<PromoteDormantRotation>(plan.actions[0].action));
    for (size_t i = 1; i < plan.actions.size(); ++i) {
        REQUIRE(std::holds_alternative<RotateActivePauli>(plan.actions[i].action));
    }

    const ExecutablePlan executable(plan);
    REQUIRE(executable.num_actions() == 2);
    CHECK(executable.inspect_action(1) == "FUSED_ROTATION descriptor=0");
}

TEST_CASE("Sampling executor prepares instrument boundaries before dispatch") {
    const clifft::InstrumentTraceOptions options = clifft::test::source_dependent_jump_options();
    const clifft::HirModule hir =
        clifft::trace(clifft::parse("LEVEL_TRANSITION[jump] 0"), &options);

    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));

    REQUIRE(executable.has_instruments());
    REQUIRE(executable.num_instrument_sites() == 1);
}

TEST_CASE("Sampling executable selects new X instrument activation") {
    const clifft::InstrumentTraceOptions options = clifft::test::source_dependent_jump_options();
    const auto compile = [&](std::string_view circuit) {
        const clifft::HirModule hir = clifft::trace(clifft::parse(circuit), &options);
        return ExecutablePlan(clifft::sampling::plan_sampling(hir));
    };

    const ExecutablePlan activating = compile("H 0\nT 0\nH 1\nLEVEL_TRANSITION[jump] 1\nM 1");
    REQUIRE(activating.num_new_x_instrument_activations() == 1);

    const ExecutablePlan already_active = compile("H 0\nT 0\nLEVEL_TRANSITION[jump] 0\nM 0");
    REQUIRE(already_active.num_new_x_instrument_activations() == 0);
}

TEST_CASE("Sampling executable preserves generic instrument activation") {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.initial_active_width = 1;
    plan.peak_active_width = 2;
    plan.symbols = {SymbolKind::Instrument};
    plan.instrument_distributions = {InstrumentDistribution{{}, {}}};
    plan.actions = {
        PlannedAction{1, 2,
                      ApplyInstrument{InstrumentSiteId{0}, InstrumentMode::Activate,
                                      ActivePauli{0, 0b10}, AffineBool{}, SymbolId{0}}},
        PlannedAction{2, 2, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
    };

    const ExecutablePlan executable(plan);
    REQUIRE(executable.num_new_x_instrument_activations() == 0);

    Executor executor(executable, 19);
    executor.run_shot();

    REQUIRE_FALSE(executor.pending_trap().has_value());
    REQUIRE(executor.state().active_width() == 2);
    REQUIRE(executor.symbols()[0] == 0);
    REQUIRE(executor.state().real_data()[0] == 1.0);
    REQUIRE(executor.state().real_data()[1] == 0.0);
    REQUIRE(executor.state().real_data()[2] == 0.0);
    REQUIRE(executor.state().real_data()[3] == 0.0);
}

TEST_CASE("Sampling executable validates its source plan before lowering") {
    SamplingPlan plan;
    plan.num_qubits = 1;
    plan.symbols = {SymbolKind::Instrument};
    plan.instrument_distributions = {InstrumentDistribution{{}, {}}};
    plan.actions = {
        PlannedAction{
            0, 0,
            ApplyInstrument{InstrumentSiteId{0},
                            static_cast<InstrumentMode>(std::numeric_limits<uint8_t>::max()),
                            ActivePauli{}, AffineBool{}, SymbolId{0}}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
    };

    REQUIRE_THROWS_AS(ExecutablePlan(plan), std::invalid_argument);
}

TEST_CASE("Sampling executable maps nonrotation actions to plan provenance") {
    const clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nT 0\nM 0\nDETECTOR rec[-1]\n"));
    SamplingPlanOptions options;
    options.retain_source_map = true;
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir, options);
    const ExecutablePlan executable(plan);

    REQUIRE(executable.num_actions() == plan.actions.size());
    for (uint32_t action = 0; action < executable.num_actions(); ++action) {
        REQUIRE(executable.action_plan_range(action) ==
                ExecutablePlan::PlanActionRange{action, action + 1});
    }
}

TEST_CASE("Sampling executor applies computational instrument destinations in line") {
    clifft::InstrumentTraceOptions options;
    clifft::InstrumentProbabilities reset;
    reset.p_fire[1] = 1.0;
    reset.p_computational_dest[1][0] = 1.0;
    options.transitions.emplace("reset", reset);

    for (std::string_view circuit : {
             "X 0\nLEVEL_TRANSITION[reset] 0\nM 0",
             "H 0\nLEVEL_TRANSITION[reset] 0\nM 0",
             "H 0\nT 0\nLEVEL_TRANSITION[reset] 0\nM 0",
         }) {
        const clifft::HirModule hir = clifft::trace(clifft::parse(circuit), &options);
        const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
        Executor executor(executable, 17);

        for (uint32_t shot = 0; shot < 20; ++shot) {
            executor.run_shot();
            CAPTURE(circuit, shot);
            REQUIRE_FALSE(executor.pending_trap().has_value());
            REQUIRE(executor.visible_records().size() == 1);
            REQUIRE(executor.visible_records()[0] == 0);
        }
    }
}

TEST_CASE("Sampling executor propagates an entangled destination flip") {
    clifft::InstrumentTraceOptions options;
    clifft::InstrumentProbabilities reset;
    reset.p_fire[0] = 1.0;
    reset.p_fire[1] = 1.0;
    reset.p_computational_dest[0][0] = 1.0;
    reset.p_computational_dest[1][0] = 1.0;
    options.transitions.emplace("reset", reset);
    const clifft::HirModule hir = clifft::trace(
        clifft::parse("H 0\nH 1\nCZ 0 1\nT 1\nLEVEL_TRANSITION[reset] 1\nM 1"), &options);
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    Executor executor(executable, 29);

    for (uint32_t shot = 0; shot < 20; ++shot) {
        executor.run_shot();
        REQUIRE_FALSE(executor.pending_trap().has_value());
        REQUIRE(executor.visible_records()[0] == 0);
    }
}

TEST_CASE("Sampling executor composes sequential computational instruments") {
    clifft::InstrumentTraceOptions options;
    clifft::InstrumentProbabilities reset_to_e;
    reset_to_e.p_fire[0] = 1.0;
    reset_to_e.p_fire[1] = 1.0;
    reset_to_e.p_computational_dest[0][1] = 1.0;
    reset_to_e.p_computational_dest[1][1] = 1.0;
    options.transitions.emplace("reset_e", reset_to_e);

    clifft::InstrumentProbabilities pump_g;
    pump_g.p_fire[0] = 1.0;
    pump_g.p_computational_dest[0][1] = 1.0;
    options.transitions.emplace("pump_g", pump_g);

    // The first site prepares E. The second can fire only from G, so its
    // destination symbol proves that the first site's update reached the next
    // instrument rather than merely producing the expected final record.
    const clifft::HirModule hir =
        clifft::trace(clifft::parse("H 0\nT 0\nLEVEL_TRANSITION[reset_e] 0\n"
                                    "LEVEL_TRANSITION[pump_g] 0\nM 0"),
                      &options);
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir);
    std::optional<SymbolId> second_flip;
    uint32_t instrument_count = 0;
    for (const PlannedAction& action : plan.actions) {
        if (const auto* instrument = std::get_if<ApplyInstrument>(&action.action)) {
            if (instrument_count == 1) {
                second_flip = instrument->destination_flip;
            }
            ++instrument_count;
        }
    }
    REQUIRE(instrument_count == 2);
    REQUIRE(second_flip.has_value());

    const ExecutablePlan executable(plan);
    Executor executor(executable, 31);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        executor.run_shot();
        CAPTURE(shot);
        REQUIRE_FALSE(executor.pending_trap().has_value());
        REQUIRE(executor.symbols()[index(*second_flip)] == 0);
        REQUIRE(executor.visible_records()[0] == 1);
    }
}

TEST_CASE("Sampling executor maps a signed active source to physical levels") {
    clifft::InstrumentTraceOptions options;
    clifft::InstrumentProbabilities reset_to_g;
    reset_to_g.p_fire[0] = 1.0;
    reset_to_g.p_fire[1] = 1.0;
    reset_to_g.p_computational_dest[0][0] = 1.0;
    reset_to_g.p_computational_dest[1][0] = 1.0;
    options.transitions.emplace("reset_g", reset_to_g);

    // This conjugation produces a constant sign while retaining active source
    // support, exercising physical G/E relabeling in both collapse and flip.
    const clifft::HirModule hir = clifft::trace(
        clifft::parse("X 0\nH 0\nT 0\nH 0\nLEVEL_TRANSITION[reset_g] 0\nM 0"), &options);
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir);
    const ApplyInstrument* planned = nullptr;
    for (const PlannedAction& action : plan.actions) {
        if (const auto* instrument = std::get_if<ApplyInstrument>(&action.action)) {
            planned = instrument;
            break;
        }
    }
    REQUIRE(planned != nullptr);
    REQUIRE(planned->mode == InstrumentMode::Active);
    REQUIRE(planned->sign == AffineBool(true));

    const ExecutablePlan executable(plan);
    Executor executor(executable, 37);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        executor.run_shot();
        CAPTURE(shot);
        REQUIRE_FALSE(executor.pending_trap().has_value());
        REQUIRE(executor.visible_records()[0] == 0);
    }
}

TEST_CASE("Sampling executor stops at noncomputational destinations") {
    clifft::InstrumentTraceOptions options;
    const clifft::HirModule hir = clifft::trace(clifft::parse("LOSS(1) 0\nM 0"), &options);
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    Executor executor(executable, 3);

    executor.run_shot();

    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.pending_trap()->site == InstrumentSiteId{0});
    REQUIRE(executor.pending_trap()->source == 0);
    REQUIRE_FALSE(executor.pending_trap()->destination_pending);
}

TEST_CASE("Sampling continuation rejects incompatible handoffs") {
    const SamplingPlan root_plan = dormant_trap_plan();
    const ExecutablePlan root(root_plan);

    SECTION("resume without a trap") {
        Executor executor(root, 1);
        REQUIRE_THROWS_AS(executor.resume(root), std::invalid_argument);
    }

    SECTION("continuation omits the trapped site") {
        SamplingPlan continuation_plan;
        continuation_plan.num_qubits = 1;
        const ExecutablePlan continuation(continuation_plan);
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(continuation), std::invalid_argument);
    }

    SECTION("boundary has the wrong live active width") {
        SamplingPlan continuation_plan = root_plan;
        continuation_plan.initial_active_width = 1;
        continuation_plan.peak_active_width = 1;
        continuation_plan.actions[0].active_before = 1;
        continuation_plan.actions[0].active_after = 1;
        continuation_plan.actions[1].active_before = 1;
        continuation_plan.actions[1].active_after = 1;
        const ExecutablePlan continuation(continuation_plan);
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(continuation), std::invalid_argument);
    }

    SECTION("boundary changes prefix symbol identities") {
        SamplingPlan continuation_plan = root_plan;
        continuation_plan.symbols.push_back(SymbolKind::Unused);
        continuation_plan.actions[1] =
            PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 2}};
        const ExecutablePlan continuation(continuation_plan);
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(continuation), std::invalid_argument);
    }

    SECTION("continuation changes public dimensions") {
        SamplingPlan continuation_plan = root_plan;
        continuation_plan.num_visible_records = 1;
        continuation_plan.actions.push_back(
            PlannedAction{0, 0, RecordClassical{AffineBool{}, RecordSlot{0}}});
        const ExecutablePlan continuation(continuation_plan);
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(continuation), std::invalid_argument);
    }

    SECTION("continuation contains an unbound presampled symbol") {
        SamplingPlan continuation_plan = root_plan;
        continuation_plan.symbols.push_back(SymbolKind::Presampled);
        const ExecutablePlan continuation(continuation_plan);
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(continuation), std::invalid_argument);
    }

    SECTION("forced trace-out record is out of range") {
        Executor executor(root, 1);
        executor.run_shot();
        REQUIRE_THROWS_AS(executor.resume(root, ForcedTraceOut{RecordSlot{0}, 0}),
                          std::invalid_argument);
    }
}

TEST_CASE("Sampling continuation rejects an unconsumed forced trace-out record") {
    const SamplingPlan root_plan = dormant_trap_plan();
    SamplingPlan continuation_plan = root_plan;
    continuation_plan.num_hidden_records = 1;
    continuation_plan.actions.insert(
        continuation_plan.actions.begin(),
        PlannedAction{0, 0, RecordClassical{AffineBool{}, RecordSlot{0}}});
    const ExecutablePlan root(root_plan);
    const ExecutablePlan continuation(continuation_plan);
    Executor executor(root, 1);

    executor.run_shot();
    REQUIRE_THROWS_AS(executor.resume(continuation, ForcedTraceOut{RecordSlot{0}, 0}),
                      std::logic_error);
}

TEST_CASE("Sampling executor defers suffix noise until an instrument continuation") {
    clifft::InstrumentTraceOptions options;
    const clifft::HirModule hir =
        clifft::trace(clifft::parse("X_ERROR(1) 0\nLEAKAGE(1) 0\nX_ERROR(1) 0\nM 0"), &options);
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir);
    const SymbolId suffix_noise = plan.presampled_noise_sites[1].outcomes[0].symbol;
    const ExecutablePlan executable(plan);
    Executor executor(executable, 9);

    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.symbols()[index(suffix_noise)] == 0);

    executor.resume(executable);
    REQUIRE_FALSE(executor.pending_trap().has_value());
    REQUIRE(executor.symbols()[index(suffix_noise)] == 1);
}

TEST_CASE("Sampling continuation reconstructs expressions from true prefix symbols") {
    clifft::InstrumentTraceOptions options;
    const clifft::HirModule hir =
        clifft::trace(clifft::parse("X_ERROR(1) 0\nLEAKAGE(1) 1\nM 0"), &options);
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir);
    const SymbolId prefix_noise = plan.presampled_noise_sites[0].outcomes[0].symbol;
    const ExecutablePlan executable(plan);
    Executor executor(executable, 9);

    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.symbols()[index(prefix_noise)] == 1);

    executor.resume(executable);
    REQUIRE_FALSE(executor.pending_trap().has_value());
    REQUIRE(executor.visible_records()[0] == 1);
}

TEST_CASE("Sampling continuation consumes a forced hidden source record") {
    SamplingPlan root_plan;
    root_plan.num_qubits = 1;
    root_plan.symbols = {SymbolKind::Unused};
    root_plan.instrument_distributions = {InstrumentDistribution{{1.0, 1.0}, {}}};
    root_plan.actions = {
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
    };
    SamplingPlan continuation_plan = root_plan;
    continuation_plan.num_hidden_records = 1;
    continuation_plan.symbols.push_back(SymbolKind::Branch);
    continuation_plan.actions.push_back(PlannedAction{
        0, 0,
        MeasureDormantRandom{0, SymbolId{1}, AffineBool::symbol(SymbolId{1}), RecordSlot{0}}});
    const ExecutablePlan root(root_plan);
    const ExecutablePlan continuation(continuation_plan);
    Executor executor(root, 11);

    executor.run_shot();
    const auto trap = executor.pending_trap();
    REQUIRE(trap.has_value());
    REQUIRE(trap->destination_pending);

    executor.resume(continuation, ForcedTraceOut{RecordSlot{0}, trap->source});
    REQUIRE_FALSE(executor.pending_trap().has_value());
    REQUIRE(executor.hidden_records()[0] == trap->source);

    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.hidden_records().empty());
    REQUIRE(executor.symbols().size() == root_plan.symbols.size());
}

TEST_CASE("Sampling continuation overwrites an expectation with exact zero") {
    SamplingPlan root_plan;
    root_plan.num_qubits = 1;
    root_plan.num_exp_vals = 2;
    root_plan.symbols = {SymbolKind::Unused};
    root_plan.instrument_distributions = {InstrumentDistribution{{1.0, 1.0}, {}}};
    root_plan.actions = {
        PlannedAction{
            0, 0,
            WriteExpectationValue{ActiveExpectation{ActivePauli{}, AffineBool{}}, ExpValSlot{0}}},
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
        PlannedAction{
            0, 0,
            WriteExpectationValue{ActiveExpectation{ActivePauli{}, AffineBool{}}, ExpValSlot{1}}},
    };
    SamplingPlan continuation_plan = root_plan;
    continuation_plan.actions.back() =
        PlannedAction{0, 0, WriteExpectationValue{std::nullopt, ExpValSlot{1}}};
    const ExecutablePlan root(root_plan);
    const ExecutablePlan continuation(continuation_plan);
    Executor executor(root, 13);

    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.exp_vals()[0] == 1.0);
    executor.resume(root);
    REQUIRE(executor.exp_vals()[0] == 1.0);
    REQUIRE(executor.exp_vals()[1] == 1.0);

    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.exp_vals()[0] == 1.0);
    REQUIRE(executor.exp_vals()[1] == 1.0);
    executor.resume(continuation);
    REQUIRE(executor.exp_vals()[0] == 1.0);
    REQUIRE(executor.exp_vals()[1] == 0.0);
}

TEST_CASE("Sampling continuation preserves fused rotation prefixes") {
    constexpr uint32_t kActiveWidth = 6;
    const std::array<RotateActivePauli, 3> prefix_rotations = {
        RotateActivePauli{{0b001011, 0b100101}, 0.25, AffineBool(false)},
        RotateActivePauli{{0b101101, 0b011010}, -0.3, AffineBool(true)},
        RotateActivePauli{{0b100110, 0b110001}, 0.4, AffineBool(false)},
    };
    const std::array<RotateActivePauli, 3> continuation_rotations = {
        RotateActivePauli{{0b001011, 0b001111}, 0.1, AffineBool(false)},
        RotateActivePauli{{0b101101, 0b101100}, -0.2, AffineBool(true)},
        RotateActivePauli{{0b100110, 0b010011}, 0.35, AffineBool(false)},
    };

    SamplingPlan root_plan;
    root_plan.num_qubits = kActiveWidth + 1;
    root_plan.initial_active_width = kActiveWidth;
    root_plan.peak_active_width = kActiveWidth;
    root_plan.instrument_distributions = {InstrumentDistribution{{1.0, 1.0}, {}}};
    for (const RotateActivePauli& rotation : prefix_rotations) {
        root_plan.actions.push_back(PlannedAction{kActiveWidth, kActiveWidth, rotation});
    }
    root_plan.actions.push_back(PlannedAction{
        kActiveWidth, kActiveWidth,
        ApplyInstrument{
            InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}});
    root_plan.actions.push_back(
        PlannedAction{kActiveWidth, kActiveWidth, InstrumentBoundary{InstrumentSiteId{0}, 0, 0}});

    SamplingPlan continuation_plan = root_plan;
    for (const RotateActivePauli& rotation : continuation_rotations) {
        continuation_plan.actions.push_back(PlannedAction{kActiveWidth, kActiveWidth, rotation});
    }

    const ExecutablePlan root(root_plan);
    const ExecutablePlan continuation(continuation_plan);
    REQUIRE(root.num_actions() == 3);
    REQUIRE(continuation.num_actions() == 4);

    State expected(kActiveWidth, kActiveWidth);
    for (const RotateActivePauli& rotation : prefix_rotations) {
        apply_rotation(expected,
                       prepare_rotation(rotation.pauli, kActiveWidth, rotation.half_turns),
                       rotation.sign.constant());
    }

    Executor executor(root, 17);
    executor.run_shot();
    REQUIRE(executor.pending_trap().has_value());
    REQUIRE(executor.pending_trap()->destination_pending);
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        CAPTURE(basis);
        REQUIRE_THAT(executor.state().real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
        REQUIRE_THAT(executor.state().imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
    }

    for (const RotateActivePauli& rotation : continuation_rotations) {
        apply_rotation(expected,
                       prepare_rotation(rotation.pauli, kActiveWidth, rotation.half_turns),
                       rotation.sign.constant());
    }
    executor.resume(continuation);
    REQUIRE_FALSE(executor.pending_trap().has_value());
    for (uint64_t basis = 0; basis < expected.size(); ++basis) {
        CAPTURE(basis);
        REQUIRE_THAT(executor.state().real_data()[basis],
                     Catch::Matchers::WithinAbs(expected.real_data()[basis], 1e-12));
        REQUIRE_THAT(executor.state().imag_data()[basis],
                     Catch::Matchers::WithinAbs(expected.imag_data()[basis], 1e-12));
    }
}

TEST_CASE("Sampling batch and survivor results carry expectation columns") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        T 0
        EXP_VAL X0
        H 1
        M 1
        DETECTOR rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection}));

    const clifft::sampling::SamplingSurvivorResult survivors =
        clifft::sampling::sample_survivors(executable, 1000, uint64_t{17}, true);
    REQUIRE(survivors.passed_shots > 400);
    REQUIRE(survivors.passed_shots < 600);
    REQUIRE(survivors.exp_vals.size() == survivors.passed_shots);
    for (double value : survivors.exp_vals) {
        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(1.0 / std::sqrt(2.0), 1e-12));
    }

    const ExecutablePlan fixed(clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse("H 0\nT 0\nEXP_VAL X0\nEXP_VAL Z0"))));
    const clifft::sampling::SamplingResult result =
        clifft::sampling::sample(fixed, 3, uint64_t{17});
    REQUIRE(result.exp_vals.size() == 6);
    for (uint32_t shot = 0; shot < 3; ++shot) {
        REQUIRE_THAT(result.exp_vals[2 * shot],
                     Catch::Matchers::WithinAbs(1.0 / std::sqrt(2.0), 1e-12));
        REQUIRE_THAT(result.exp_vals[2 * shot + 1], Catch::Matchers::WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Sampling executor presamples mutually exclusive Pauli noise") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        E(0.5) X0
        ELSE_CORRELATED_ERROR(0.5) X1
        M 0 1
    )"));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));

    const clifft::sampling::SamplingResult result =
        clifft::sampling::sample(executable, 10000, uint64_t{17});
    uint32_t q0_ones = 0;
    uint32_t q1_ones = 0;
    for (uint32_t shot = 0; shot < 10000; ++shot) {
        const uint8_t q0 = result.measurements[static_cast<size_t>(shot) * 2];
        const uint8_t q1 = result.measurements[static_cast<size_t>(shot) * 2 + 1];
        REQUIRE_FALSE((q0 && q1));
        q0_ones += q0;
        q1_ones += q1;
    }
    REQUIRE(q0_ones > 4500);
    REQUIRE(q0_ones < 5500);
    REQUIRE(q1_ones > 2000);
    REQUIRE(q1_ones < 3000);
}

TEST_CASE("Sampling driver derives an independent RNG stream for each shot") {
    const clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nM 0"));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    constexpr uint32_t shots = 32;
    constexpr uint64_t seed = 9182;
    const clifft::sampling::SamplingResult result =
        clifft::sampling::sample(executable, shots, seed);
    const clifft::SeedRoot root = clifft::seed_root_from_seed(seed);
    Executor executor(executable);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        const auto words = clifft::derive_state(root, shot, clifft::kSamplingExecutorDomain);
        executor.reseed_full(words[0], words[1], words[2], words[3]);
        executor.run_shot();
        REQUIRE(result.measurements[shot] == executor.visible_records()[0]);
    }
}

TEST_CASE("Threaded fixed-row sampling preserves seeded shot order") {
    const ExecutablePlan executable(clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
        H 0 1
        T 0
        M 0 1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z0
    )"))));
    const clifft::sampling::SamplingResult serial =
        clifft::sampling::sample(executable, 257, uint64_t{9183}, 1);

    for (uint32_t threads : std::array<uint32_t, 2>{2, 0}) {
        const clifft::sampling::SamplingResult threaded =
            clifft::sampling::sample(executable, 257, uint64_t{9183}, threads);
        CAPTURE(threads);
        REQUIRE(threaded.measurements == serial.measurements);
        REQUIRE(threaded.detectors == serial.detectors);
        REQUIRE(threaded.observables == serial.observables);
        REQUIRE(threaded.exp_vals == serial.exp_vals);
    }
}

TEST_CASE("Explicit batch capacities replay seeded fixed rows") {
    const ExecutablePlan executable(clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
        X_ERROR(0.125) 0
        H 0 1
        T 0
        M(0.25) 0 1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z0
    )"))));
    for (uint32_t capacity : std::array<uint32_t, 4>{2, 63, 64, 65}) {
        const clifft::sampling::SamplingResult packed =
            clifft::sampling::sample(executable, 257, uint64_t{91831}, 1, std::nullopt, capacity);
        const clifft::sampling::SamplingResult replay =
            clifft::sampling::sample(executable, 257, uint64_t{91831}, 4, std::nullopt, capacity);
        CAPTURE(capacity);
        REQUIRE(packed.measurements == replay.measurements);
        REQUIRE(packed.detectors == replay.detectors);
        REQUIRE(packed.observables == replay.observables);
        REQUIRE(packed.exp_vals == replay.exp_vals);
    }
}

TEST_CASE("Packed presampled expression program matches a categorical statistical oracle") {
    constexpr uint32_t num_sites = 32;
    constexpr uint32_t num_symbols = 64;
    constexpr uint32_t num_unique_expressions = 32;
    constexpr uint32_t num_records = 40;
    constexpr uint32_t shots = 50'000;
    SamplingPlan plan;
    plan.num_visible_records = num_records;
    for (uint32_t symbol = 0; symbol < num_symbols; ++symbol) {
        plan.symbols.push_back(SymbolKind::Presampled);
    }
    std::array<double, num_symbols> outcome_probabilities{};
    for (uint32_t site = 0; site < num_sites; ++site) {
        const uint32_t first_symbol = 2 * site;
        const double first_probability = 0.005 + 0.0005 * site;
        const double second_probability = 0.01 + 0.0007 * site;
        outcome_probabilities[first_symbol] = first_probability;
        outcome_probabilities[first_symbol + 1] = second_probability;
        plan.presampled_noise_sites.push_back(PresampledNoiseSite{
            first_probability + second_probability,
            {PresampledNoiseOutcome{SymbolId{first_symbol}, first_probability},
             PresampledNoiseOutcome{SymbolId{first_symbol + 1}, second_probability}}});
    }
    std::vector<double> expected_record_probabilities(num_records, 0.0);
    for (uint32_t record = 0; record < num_records; ++record) {
        const uint32_t expression = record % num_unique_expressions;
        std::vector<SymbolId> terms;
        terms.reserve(48);
        for (uint32_t symbol = 0; symbol < 48; ++symbol) {
            if (symbol != expression) {
                terms.push_back(SymbolId{symbol});
            }
        }
        terms.push_back(SymbolId{48 + expression % 16});
        std::array<uint8_t, num_symbols> included{};
        for (SymbolId term : terms) {
            included[index(term)] = 1;
        }
        double even_minus_odd = 1.0;
        for (uint32_t site = 0; site < num_sites; ++site) {
            const uint32_t first_symbol = 2 * site;
            double odd_probability = 0.0;
            if (included[first_symbol] != 0) {
                odd_probability += outcome_probabilities[first_symbol];
            }
            if (included[first_symbol + 1] != 0) {
                odd_probability += outcome_probabilities[first_symbol + 1];
            }
            even_minus_odd *= 1.0 - 2.0 * odd_probability;
        }
        const double odd_probability = 0.5 * (1.0 - even_minus_odd);
        expected_record_probabilities[record] =
            expression % 2 == 0 ? odd_probability : 1.0 - odd_probability;
        plan.actions.push_back(PlannedAction{
            0, 0,
            RecordClassical{AffineBool::from_canonical_terms(expression % 2 != 0, std::move(terms)),
                            RecordSlot{record}}});
    }

    const ExecutablePlan executable(plan);
    REQUIRE(executable.num_batch_noise_carriers() > 0);
    const auto record_frequencies = [](const clifft::sampling::SamplingResult& result) {
        std::array<double, num_records> frequencies{};
        for (uint32_t shot = 0; shot < shots; ++shot) {
            for (uint32_t record = 0; record < num_records; ++record) {
                frequencies[record] +=
                    result.measurements[static_cast<size_t>(shot) * num_records + record];
            }
        }
        for (double& frequency : frequencies) {
            frequency /= shots;
        }
        return frequencies;
    };
    const clifft::sampling::SamplingResult scalar =
        clifft::sampling::sample(executable, shots, uint64_t{91832}, 1, std::nullopt, uint32_t{1});
    const std::array<double, num_records> scalar_frequencies = record_frequencies(scalar);
    for (uint32_t record = 0; record < num_records; ++record) {
        CAPTURE(record);
        REQUIRE_THAT(scalar_frequencies[record],
                     Catch::Matchers::WithinAbs(expected_record_probabilities[record], 0.015));
    }

    for (uint32_t capacity : std::array<uint32_t, 5>{2, 63, 64, 65, 128}) {
        const clifft::sampling::SamplingResult packed =
            clifft::sampling::sample(executable, shots, uint64_t{91832}, 1, std::nullopt, capacity);
        const std::array<double, num_records> packed_frequencies = record_frequencies(packed);
        for (uint32_t record = 0; record < num_records; ++record) {
            CAPTURE(capacity, record);
            REQUIRE_THAT(packed_frequencies[record],
                         Catch::Matchers::WithinAbs(expected_record_probabilities[record], 0.015));
            REQUIRE_THAT(packed_frequencies[record],
                         Catch::Matchers::WithinAbs(scalar_frequencies[record], 0.02));
        }
    }
}

TEST_CASE("Packed detector and observable outputs preserve record snapshots") {
    const std::array<uint8_t, 2> expected_detectors{1, 1};
    const std::array<uint8_t, 3> expected_observables{1, 1, 1};
    clifft::sampling::SamplingPlanOptions options;
    options.expected_detectors = expected_detectors;
    options.expected_observables = expected_observables;
    const ExecutablePlan executable(clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
            X 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(2) rec[-1]
            READOUT_NOISE(1) rec[-1]
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-1]
            OBSERVABLE_INCLUDE(2) rec[-1]
        )")),
                                                                    options));
    const clifft::sampling::SamplingResult scalar =
        clifft::sampling::sample(executable, 257, uint64_t{918321}, 1, std::nullopt, uint32_t{1});

    for (uint32_t capacity : std::array<uint32_t, 5>{2, 63, 64, 65, 128}) {
        const clifft::sampling::SamplingResult packed =
            clifft::sampling::sample(executable, 257, uint64_t{918321}, 1, std::nullopt, capacity);
        CAPTURE(capacity);
        REQUIRE(packed.measurements == scalar.measurements);
        REQUIRE(packed.detectors == scalar.detectors);
        REQUIRE(packed.observables == scalar.observables);
        for (uint32_t shot = 0; shot < 257; ++shot) {
            REQUIRE(packed.measurements[shot] == 0);
            REQUIRE(packed.detectors[2 * shot] == 0);
            REQUIRE(packed.detectors[2 * shot + 1] == 1);
            REQUIRE(packed.observables[3 * shot] == 0);
            REQUIRE(packed.observables[3 * shot + 1] == 1);
            REQUIRE(packed.observables[3 * shot + 2] == 0);
        }
    }
}

TEST_CASE("Scalar and packed sampling are statistically equivalent") {
    constexpr uint32_t shots = 100000;
    const ExecutablePlan executable(clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
            X_ERROR(0.125) 0
            H 1
            M 0 1
            DETECTOR rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
        )"))));
    const clifft::sampling::SamplingResult scalar =
        clifft::sampling::sample(executable, shots, uint64_t{91833}, 1, std::nullopt, uint32_t{1});
    const clifft::sampling::SamplingResult packed = clifft::sampling::sample(
        executable, shots, uint64_t{91833}, 1, std::nullopt, uint32_t{257});

    const auto frequencies = [](const clifft::sampling::SamplingResult& result) {
        std::array<double, 2> ones{};
        for (size_t shot = 0; shot < result.measurements.size() / 2; ++shot) {
            ones[0] += result.measurements[2 * shot];
            ones[1] += result.measurements[2 * shot + 1];
        }
        ones[0] /= shots;
        ones[1] /= shots;
        return ones;
    };
    const std::array<double, 2> scalar_frequencies = frequencies(scalar);
    const std::array<double, 2> packed_frequencies = frequencies(packed);
    REQUIRE_THAT(scalar_frequencies[0], Catch::Matchers::WithinAbs(0.125, 0.01));
    REQUIRE_THAT(packed_frequencies[0], Catch::Matchers::WithinAbs(0.125, 0.01));
    REQUIRE_THAT(scalar_frequencies[1], Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE_THAT(packed_frequencies[1], Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE_THAT(packed_frequencies[0], Catch::Matchers::WithinAbs(scalar_frequencies[0], 0.01));
    REQUIRE_THAT(packed_frequencies[1], Catch::Matchers::WithinAbs(scalar_frequencies[1], 0.01));

    const auto output_frequency = [](const std::vector<uint8_t>& values) {
        return static_cast<double>(std::ranges::count(values, uint8_t{1})) / shots;
    };
    const double scalar_detector = output_frequency(scalar.detectors);
    const double packed_detector = output_frequency(packed.detectors);
    const double scalar_observable = output_frequency(scalar.observables);
    const double packed_observable = output_frequency(packed.observables);
    REQUIRE_THAT(scalar_detector, Catch::Matchers::WithinAbs(0.125, 0.01));
    REQUIRE_THAT(packed_detector, Catch::Matchers::WithinAbs(0.125, 0.01));
    REQUIRE_THAT(packed_detector, Catch::Matchers::WithinAbs(scalar_detector, 0.01));
    REQUIRE_THAT(scalar_observable, Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE_THAT(packed_observable, Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE_THAT(packed_observable, Catch::Matchers::WithinAbs(scalar_observable, 0.01));
}

TEST_CASE("Packed uniform sparse noise matches its Bernoulli rate") {
    constexpr uint32_t sites = 64;
    constexpr uint32_t shots = 100000;
    std::string circuit = "X_ERROR(0.001)";
    for (uint32_t qubit = 0; qubit < sites; ++qubit) {
        circuit.append(" ").append(std::to_string(qubit));
    }
    circuit.append("\nM");
    for (uint32_t qubit = 0; qubit < sites; ++qubit) {
        circuit.append(" ").append(std::to_string(qubit));
    }
    circuit.push_back('\n');
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit))));
    const clifft::sampling::SamplingResult result = clifft::sampling::sample(
        executable, shots, uint64_t{91834}, 1, std::nullopt, uint32_t{2048});

    uint64_t faults = 0;
    for (uint8_t value : result.measurements) {
        faults += value;
    }
    REQUIRE(faults > 5900);
    REQUIRE(faults < 6900);
}

TEST_CASE("Packed certain noise fires every site") {
    const ExecutablePlan executable(clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse("X_ERROR(1) 0 1 2\nM 0 1 2\n"))));
    const clifft::sampling::SamplingResult result =
        clifft::sampling::sample(executable, 257, uint64_t{91835}, 1, std::nullopt, uint32_t{65});
    REQUIRE(std::ranges::all_of(result.measurements, [](uint8_t value) { return value == 1; }));
}

TEST_CASE("Packed symmetric readout matches its Bernoulli rates") {
    constexpr uint32_t shots = 100000;
    const ExecutablePlan executable(clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse("M(0.001) 0\nM(0.5) 1\nX 2\nM(1) 2\n"))));
    const clifft::sampling::SamplingResult result = clifft::sampling::sample(
        executable, shots, uint64_t{91836}, 1, std::nullopt, uint32_t{2048});

    std::array<uint32_t, 3> ones{};
    for (uint32_t shot = 0; shot < shots; ++shot) {
        for (uint32_t record = 0; record < 3; ++record) {
            ones[record] += result.measurements[static_cast<size_t>(shot) * 3 + record];
        }
    }
    REQUIRE(ones[0] > 50);
    REQUIRE(ones[0] < 150);
    REQUIRE(ones[1] > 49000);
    REQUIRE(ones[1] < 51000);
    REQUIRE(ones[2] == 0);
}

TEST_CASE("Packed asymmetric readout matches both conditional rates") {
    constexpr uint32_t shots = 100000;
    const ExecutablePlan executable(clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
        M 0
        READOUT_NOISE(0.3, 0.05) rec[-1]
        X 1
        M 1
        READOUT_NOISE(0.3, 0.05) rec[-1]
    )"))));
    const clifft::sampling::SamplingResult result = clifft::sampling::sample(
        executable, shots, uint64_t{91837}, 1, std::nullopt, uint32_t{2048});

    std::array<uint32_t, 2> ones{};
    for (uint32_t shot = 0; shot < shots; ++shot) {
        ones[0] += result.measurements[static_cast<size_t>(shot) * 2];
        ones[1] += result.measurements[static_cast<size_t>(shot) * 2 + 1];
    }
    REQUIRE(ones[0] > 29000);
    REQUIRE(ones[0] < 31000);
    REQUIRE(ones[1] > 94000);
    REQUIRE(ones[1] < 96000);
}

TEST_CASE("Sampling thread layouts validate explicit worker counts") {
    const ExecutablePlan executable(plan_from("H 0\nM 0\n"));
    REQUIRE_THROWS_WITH(clifft::sampling::sample(executable, 1, uint64_t{11}, 1,
                                                 clifft::sampling::ThreadLayout{
                                                     .shot_workers = 0, .intra_shot_workers = 1}),
                        "thread_layout worker counts must be positive");

#if defined(CLIFFT_TESTS_HAVE_OPENMP)
    const std::array hybrid_layouts{
        clifft::sampling::ThreadLayout{
            .shot_workers = 2, .intra_shot_workers = 2, .intra_shot_min_active_width = 0},
        clifft::sampling::ThreadLayout{
            .shot_workers = 2, .intra_shot_workers = 2, .intra_shot_min_active_width = 18},
    };
    if (clifft::openmp_process_binding_active()) {
        for (const clifft::sampling::ThreadLayout hybrid : hybrid_layouts) {
            REQUIRE_THROWS_WITH(clifft::sampling::sample(executable, 7, uint64_t{12}, 1, hybrid),
                                "hybrid thread_layout requires OMP_PROC_BIND=false");
        }
    } else {
        for (const clifft::sampling::ThreadLayout hybrid : hybrid_layouts) {
            const clifft::sampling::SamplingResult result =
                clifft::sampling::sample(executable, 7, uint64_t{12}, 1, hybrid);
            REQUIRE(result.measurements.size() == 7);
        }
    }
#else
    REQUIRE_THROWS_WITH(
        clifft::sampling::sample(
            executable, 1, uint64_t{12}, 1,
            clifft::sampling::ThreadLayout{
                .shot_workers = 1, .intra_shot_workers = 2, .intra_shot_min_active_width = 0}),
        "thread_layout intra-shot workers require an OpenMP-enabled build");
#endif
}

#if defined(CLIFFT_TESTS_HAVE_OPENMP)
TEST_CASE("Automatic intra-shot sampling preserves seeded results") {
    constexpr uint32_t width = 18;
    std::string circuit;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        circuit.append("H ").append(std::to_string(qubit)).append("\n");
        circuit.append("T ").append(std::to_string(qubit)).append("\n");
    }
    circuit.append("M");
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        circuit.append(" ").append(std::to_string(qubit));
    }
    circuit.append("\n");

    const ExecutablePlan executable(plan_from(circuit));
    REQUIRE(executable.peak_active_width() == width);
    const clifft::sampling::SamplingResult serial =
        clifft::sampling::sample(executable, 2, uint64_t{9184}, 1);
    const clifft::sampling::SamplingResult automatic =
        clifft::sampling::sample(executable, 2, uint64_t{9184}, 4);

    REQUIRE(automatic.measurements == serial.measurements);
    REQUIRE(automatic.detectors == serial.detectors);
    REQUIRE(automatic.observables == serial.observables);
    REQUIRE(automatic.exp_vals == serial.exp_vals);
}

TEST_CASE("Intra-shot rotation and promotion kernels preserve serial coefficients") {
    constexpr uint32_t width = 19;
    SamplingPlan plan;
    plan.num_qubits = width;
    plan.peak_active_width = width;
    for (uint32_t active = 0; active < width; ++active) {
        plan.actions.push_back(
            PlannedAction{active, active + 1, PromoteDormantRotation{0.25, AffineBool(false)}});
    }
    constexpr std::array<ActivePauli, 8> paulis{{
        {0x15555, 0x2aaaa},
        {0x2a9c3, 0x13179},
        {0, 0x3ffff},
        {0x3ff00, 0x00ff3},
        {0x2c71d, 0x19ac6},
        {0x00003, 0x2c5a1},
        {0x20000, 0x15555},
        {0x31b6d, 0x0e493},
    }};
    for (const ActivePauli pauli : paulis) {
        plan.actions.push_back(
            PlannedAction{width, width, RotateActivePauli{pauli, 0.125, AffineBool(false)}});
    }

    const ExecutablePlan executable(plan);
    Executor serial(executable, 17, 1);
    Executor parallel(executable, 17, 4);
    serial.run_shot();
    parallel.run_shot();

    REQUIRE(parallel.state().active_width() == serial.state().active_width());
    REQUIRE(std::ranges::equal(parallel.state().real(), serial.state().real()));
    REQUIRE(std::ranges::equal(parallel.state().imag(), serial.state().imag()));
}

TEST_CASE("Intra-shot active-width threshold is configurable") {
    constexpr uint32_t width = 17;
    SamplingPlan plan;
    plan.num_qubits = width;
    plan.peak_active_width = width;
    for (uint32_t active = 0; active < width; ++active) {
        plan.actions.push_back(
            PlannedAction{active, active + 1, PromoteDormantRotation{0.25, AffineBool(false)}});
    }
    plan.actions.push_back(PlannedAction{
        width, width, RotateActivePauli{ActivePauli{0x15555, 0x1aaaa}, 0.125, AffineBool(false)}});

    const ExecutablePlan executable(plan);
    Executor serial(executable, 19, 1, width);
    Executor parallel(executable, 19, 4, width);
    serial.run_shot();
    parallel.run_shot();

    REQUIRE(std::ranges::equal(parallel.state().real(), serial.state().real()));
    REQUIRE(std::ranges::equal(parallel.state().imag(), serial.state().imag()));
}
#endif

TEST_CASE("Threaded survivor sampling preserves seeded survivors and records") {
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
            H 0
            M 0
            DETECTOR rec[-1]
            H 1
            M 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            EXP_VAL Z1
        )")),
                                        {.postselection_mask = postselection,
                                         .expected_detectors = {},
                                         .expected_observables = {}}));
    const clifft::sampling::SamplingSurvivorResult serial =
        clifft::sampling::sample_survivors(executable, 257, uint64_t{9184}, true, 1);

    for (uint32_t threads : std::array<uint32_t, 2>{3, 0}) {
        const clifft::sampling::SamplingSurvivorResult threaded =
            clifft::sampling::sample_survivors(executable, 257, uint64_t{9184}, true, threads);
        CAPTURE(threads);
        REQUIRE(threaded.total_shots == serial.total_shots);
        REQUIRE(threaded.passed_shots == serial.passed_shots);
        REQUIRE(threaded.logical_errors == serial.logical_errors);
        REQUIRE(threaded.observable_ones == serial.observable_ones);
        REQUIRE(threaded.measurements == serial.measurements);
        REQUIRE(threaded.detectors == serial.detectors);
        REQUIRE(threaded.observables == serial.observables);
        REQUIRE(threaded.exp_vals == serial.exp_vals);
    }
}

TEST_CASE("Explicit batch capacities replay seeded survivor rows") {
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(R"(
            H 0
            M 0
            EXP_VAL Z0
            DETECTOR rec[-1]
            H 1
            T 1
            EXP_VAL X1
            M 1
            OBSERVABLE_INCLUDE(0) rec[-1]
        )")),
                                        {.postselection_mask = postselection,
                                         .expected_detectors = {},
                                         .expected_observables = {}}));
    for (uint32_t capacity : std::array<uint32_t, 4>{2, 63, 64, 65}) {
        const clifft::sampling::SamplingSurvivorResult packed = clifft::sampling::sample_survivors(
            executable, 257, uint64_t{91841}, true, 1, std::nullopt, capacity);
        const clifft::sampling::SamplingSurvivorResult replay = clifft::sampling::sample_survivors(
            executable, 257, uint64_t{91841}, true, 4, std::nullopt, capacity);
        CAPTURE(capacity);
        REQUIRE(packed.total_shots == replay.total_shots);
        REQUIRE(packed.passed_shots == replay.passed_shots);
        REQUIRE(packed.logical_errors == replay.logical_errors);
        REQUIRE(packed.observable_ones == replay.observable_ones);
        REQUIRE(packed.measurements == replay.measurements);
        REQUIRE(packed.detectors == replay.detectors);
        REQUIRE(packed.observables == replay.observables);
        REQUIRE(packed.exp_vals == replay.exp_vals);
    }
}

TEST_CASE("Packed survivors retain long-tail rows after heavy rejection") {
    constexpr uint32_t shots = 4096;
    constexpr uint32_t tail_probes = 96;
    std::string circuit = R"(
        X_ERROR(0.9) 0
        M 0
        EXP_VAL Z0
        DETECTOR rec[-1]
        H 1
    )";
    for (uint32_t probe = 0; probe < tail_probes; ++probe) {
        circuit.append("T 1\nEXP_VAL X1\nH 1\n");
    }
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit));
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection,
                                              .expected_detectors = {},
                                              .expected_observables = {}}));
    REQUIRE(executable.num_actions() > 190);
    REQUIRE(executable.num_exp_vals() == tail_probes + 1);

    const clifft::sampling::SamplingSurvivorResult result = clifft::sampling::sample_survivors(
        executable, shots, uint64_t{91842}, true, 1, std::nullopt, uint32_t{2048});
    const clifft::sampling::SamplingSurvivorResult replay = clifft::sampling::sample_survivors(
        executable, shots, uint64_t{91842}, true, 4, std::nullopt, uint32_t{2048});

    REQUIRE(result.passed_shots > 300);
    REQUIRE(result.passed_shots < 520);
    REQUIRE(result.measurements.size() == result.passed_shots);
    REQUIRE(result.detectors.size() == result.passed_shots);
    REQUIRE(result.exp_vals.size() ==
            static_cast<size_t>(result.passed_shots) * executable.num_exp_vals());
    REQUIRE(std::ranges::all_of(result.measurements, [](uint8_t value) { return value == 0; }));
    REQUIRE(std::ranges::all_of(result.detectors, [](uint8_t value) { return value == 0; }));
    for (uint32_t shot = 0; shot < result.passed_shots; ++shot) {
        const size_t offset = static_cast<size_t>(shot) * executable.num_exp_vals();
        REQUIRE_THAT(result.exp_vals[offset], Catch::Matchers::WithinAbs(1.0, 1e-12));
    }
    REQUIRE(std::ranges::all_of(result.exp_vals, [](double value) {
        return std::isfinite(value) && value >= -1.000000000001 && value <= 1.000000000001;
    }));
    REQUIRE(replay.passed_shots == result.passed_shots);
    REQUIRE(replay.logical_errors == result.logical_errors);
    REQUIRE(replay.observable_ones == result.observable_ones);
    REQUIRE(replay.measurements == result.measurements);
    REQUIRE(replay.detectors == result.detectors);
    REQUIRE(replay.observables == result.observables);
    REQUIRE(replay.exp_vals == result.exp_vals);
}

TEST_CASE("Sampling survivor execution normalizes and rejects detectors") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan executable(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection}));

    const clifft::sampling::SamplingSurvivorResult result =
        clifft::sampling::sample_survivors(executable, 1000, uint64_t{19}, true);
    REQUIRE(result.total_shots == 1000);
    REQUIRE(result.passed_shots > 400);
    REQUIRE(result.passed_shots < 600);
    REQUIRE(result.logical_errors == 0);
    REQUIRE(result.measurements.size() == result.passed_shots);
    REQUIRE(std::ranges::all_of(result.measurements, [](uint8_t value) { return value == 0; }));
    REQUIRE(std::ranges::all_of(result.detectors, [](uint8_t value) { return value == 0; }));
    REQUIRE(std::ranges::all_of(result.observables, [](uint8_t value) { return value == 0; }));
}

TEST_CASE("Sampling batch helpers reject unbound presampled symbols") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolKind::Presampled};
    plan.actions = {
        PlannedAction{0, 0, RecordClassical{AffineBool::symbol(SymbolId{0}), RecordSlot{0}}}};
    const ExecutablePlan executable(plan);

    REQUIRE_THROWS_WITH(sample_records(executable, 0, uint64_t{1234}),
                        "batch sampling requires a distribution for every presampled symbol");
    REQUIRE_THROWS_WITH(record_log_probabilities(executable, std::array<uint8_t, 1>{0}, 1),
                        Catch::Matchers::ContainsSubstring("requires pure-state evolution"));
}

TEST_CASE("Sampling record probabilities reject output annotations in the core API") {
    const std::array<uint8_t, 1> record{0};
    const clifft::HirModule hir =
        clifft::trace(clifft::parse("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"));
    const ExecutablePlan annotated(clifft::sampling::plan_sampling(hir));
    REQUIRE_THROWS_WITH(record_log_probabilities(annotated, record, 1),
                        Catch::Matchers::ContainsSubstring("requires pure-state evolution"));

    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan postselected(
        clifft::sampling::plan_sampling(hir, {.postselection_mask = postselection}));
    REQUIRE_THROWS_WITH(record_log_probabilities(postselected, record, 1),
                        Catch::Matchers::ContainsSubstring("requires pure-state evolution"));
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
