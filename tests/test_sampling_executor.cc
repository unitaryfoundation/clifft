#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/svm/svm.h"
#include "clifft/util/noise_sampling.h"
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
#include <string_view>
#include <variant>
#include <vector>

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
using clifft::sampling::NoiseSiteId;
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
using clifft::sampling::SymbolInfo;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteExpectationValue;

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
                                     .num_exp_vals = legacy_program.num_exp_vals,
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
        REQUIRE(executor.exp_vals().size() == legacy.exp_vals.size());
        for (size_t i = 0; i < legacy.exp_vals.size(); ++i) {
            REQUIRE_THAT(executor.exp_vals()[i],
                         Catch::Matchers::WithinAbs(legacy.exp_vals[i], 1e-12));
        }
    }
}

void require_basis_probabilities_match_legacy(std::string_view circuit_text,
                                              std::span<const uint64_t> basis_masks,
                                              size_t num_basis_masks, size_t words_per_basis_mask) {
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit_text));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    const std::vector<double> actual = clifft::sampling::basis_probabilities(
        executable, basis_masks, num_basis_masks, words_per_basis_mask);
    const std::vector<double> expected = clifft::basis_probabilities(
        clifft::lower(hir), basis_masks, num_basis_masks, words_per_basis_mask);
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(circuit_text, i);
        REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 1e-12));
    }
}

void require_statevector_matches_legacy(std::string_view circuit_text) {
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit_text));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));
    const std::vector<std::complex<double>> actual = clifft::sampling::get_statevector(executable);

    const clifft::CompiledModule legacy_program = clifft::lower(hir);
    clifft::SchrodingerState legacy({.peak_rank = legacy_program.peak_rank,
                                     .num_measurements = legacy_program.total_meas_slots,
                                     .num_qubits = legacy_program.num_qubits,
                                     .num_exp_vals = legacy_program.num_exp_vals,
                                     .seed = 0});
    clifft::execute(legacy_program, legacy);
    const std::vector<std::complex<double>> expected =
        clifft::get_statevector(legacy_program, legacy);

    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(circuit_text, i);
        REQUIRE_THAT(actual[i].real(), Catch::Matchers::WithinAbs(expected[i].real(), 1e-6));
        REQUIRE_THAT(actual[i].imag(), Catch::Matchers::WithinAbs(expected[i].imag(), 1e-6));
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

SamplingPlan categorical_noise_plan() {
    SamplingPlan plan;
    plan.num_noise_sites = 4;
    plan.symbols = {
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{0}},
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{1}},
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{1}},
        SymbolInfo{SymbolKind::Presampled, std::nullopt, NoiseSiteId{3}},
    };
    plan.presampled_noise_sites = {
        PresampledNoiseSite{NoiseSiteId{0}, 0.05, {PresampledNoiseOutcome{SymbolId{0}, 0.05}}},
        PresampledNoiseSite{
            NoiseSiteId{1},
            0.3,
            {PresampledNoiseOutcome{SymbolId{1}, 0.1}, PresampledNoiseOutcome{SymbolId{2}, 0.2}}},
        PresampledNoiseSite{NoiseSiteId{2}, 0.0, {}},
        PresampledNoiseSite{NoiseSiteId{3}, 0.4, {PresampledNoiseOutcome{SymbolId{3}, 0.4}}},
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
    plan.num_noise_sites = 1;
    plan.presampled_noise_sites = {PresampledNoiseSite{NoiseSiteId{0}, 0.0, {}}};
    plan.symbols = {SymbolInfo{SymbolKind::Branch, 0, std::nullopt}};
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

TEST_CASE("Sampling executor fuses constant rotation orbits") {
    auto require_matches_scalar = [](uint32_t active_width,
                                     std::span<const RotateActivePauli> rotations) {
        SamplingPlan plan;
        plan.num_qubits = active_width;
        plan.initial_active_width = active_width;
        plan.max_active_width = active_width;
        plan.symbols = {
            SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt},
        };
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
    wide_selector_plan.max_active_width = 6;
    for (uint32_t axis = 0; axis < 6; ++axis) {
        wide_selector_plan.actions.push_back(PlannedAction{
            6, 6, RotateActivePauli{{0, uint64_t{1} << axis}, 0.25, AffineBool(false)}});
    }
    REQUIRE(ExecutablePlan(wide_selector_plan).num_actions() == 6);
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

TEST_CASE("Sampling basis probabilities match legacy exact queries") {
    constexpr std::array<uint64_t, 4> kTwoQubitBasis{0, 1, 2, 3};
    require_basis_probabilities_match_legacy("H 0\nCX 0 1\n", kTwoQubitBasis, 4, 1);
    require_basis_probabilities_match_legacy("H 0\nT 0\nH 1\nT 1\nCX 0 1\n", kTwoQubitBasis, 4, 1);
    require_basis_probabilities_match_legacy("H 0\nT 0\nCX 0 1\nR_Z(0.123) 1\nH 0\n",
                                             kTwoQubitBasis, 4, 1);

    constexpr std::array<uint64_t, 4> kHighQubitBasis{0, 0, 0, uint64_t{1} << 6};
    require_basis_probabilities_match_legacy("H 70\n", kHighQubitBasis, 2, 2);
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

TEST_CASE("Sampling statevectors match legacy exact queries componentwise") {
    require_statevector_matches_legacy("H 0\n");
    require_statevector_matches_legacy("H 0\nT 0\n");
    require_statevector_matches_legacy("Y 0\nH 0\nT 0\nT 0\nT 0\n");
    require_statevector_matches_legacy("H 0\nT_DAG 0\nH 1\nCX 0 1\nR_Z(0.123) 1\nH 0\n");
    require_statevector_matches_legacy(
        "H 0\nH 1\nR_XX(0.25) 0 1\nR_YY(-0.375) 0 1\nR_PAULI(0.2) X0*Y1\n");
    require_statevector_matches_legacy("H 0\nT 0\nT 0\nH 0\nT 0\n");
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
    // Only the executed prefix is meaningful after an unreachable replay.
    REQUIRE(executor.visible_records()[0] == 1);
}

TEST_CASE("Sampling expression registers reset true symbols between shots") {
    SamplingPlan plan;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt}};
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
    plan.max_active_width = 2;
    plan.num_instrument_sites = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Instrument, 0, std::nullopt}};
    plan.instrument_distributions = {InstrumentDistribution{InstrumentSiteId{0}, {}, {}}};
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
    plan.num_instrument_sites = 1;
    plan.symbols = {SymbolInfo{SymbolKind::Instrument, 0, std::nullopt}};
    plan.instrument_distributions = {InstrumentDistribution{InstrumentSiteId{0}, {}, {}}};
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
    root_plan.num_instrument_sites = 1;
    root_plan.symbols = {SymbolInfo{SymbolKind::Unused, std::nullopt, std::nullopt}};
    root_plan.instrument_distributions = {
        InstrumentDistribution{InstrumentSiteId{0}, {1.0, 1.0}, {}}};
    root_plan.actions = {
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
    };
    SamplingPlan continuation_plan = root_plan;
    continuation_plan.num_hidden_records = 1;
    continuation_plan.symbols.push_back(SymbolInfo{SymbolKind::Branch, 2, std::nullopt});
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
    root_plan.num_instrument_sites = 1;
    root_plan.symbols = {SymbolInfo{SymbolKind::Unused, std::nullopt, std::nullopt}};
    root_plan.instrument_distributions = {
        InstrumentDistribution{InstrumentSiteId{0}, {1.0, 1.0}, {}}};
    root_plan.actions = {
        PlannedAction{0, 0, WriteExpectationValue{ActivePauli{}, AffineBool{}, ExpValSlot{0}}},
        PlannedAction{
            0, 0,
            ApplyInstrument{
                InstrumentSiteId{0}, InstrumentMode::DormantTrap, {}, AffineBool{}, std::nullopt}},
        PlannedAction{0, 0, InstrumentBoundary{InstrumentSiteId{0}, 0, 1}},
        PlannedAction{0, 0, WriteExpectationValue{ActivePauli{}, AffineBool{}, ExpValSlot{1}}},
    };
    SamplingPlan continuation_plan = root_plan;
    continuation_plan.actions.back() =
        PlannedAction{0, 0, WriteExpectationValue{std::nullopt, AffineBool{}, ExpValSlot{1}}};
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
    root_plan.max_active_width = kActiveWidth;
    root_plan.num_instrument_sites = 1;
    root_plan.instrument_distributions = {
        InstrumentDistribution{InstrumentSiteId{0}, {1.0, 1.0}, {}}};
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

TEST_CASE("Sampling batch records match legacy seeded execution") {
    constexpr std::string_view kCircuit = "H 0\nH 1\nT 0\nT 1\nCX 0 1\nM 0 1\n";
    const clifft::HirModule hir = clifft::trace(clifft::parse(kCircuit));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));

    const std::vector<uint8_t> actual = sample_records(executable, 64, uint64_t{5678});
    const clifft::SampleResult expected = clifft::sample(clifft::lower(hir), 64, uint64_t{5678});
    REQUIRE(actual == expected.measurements);
    REQUIRE(sample_records(executable, 0, uint64_t{5678}).empty());
}

TEST_CASE("Sampling executor expectation probes match legacy execution") {
    require_matches_legacy("EXP_VAL X0\nEXP_VAL Z0", 1, 1234);
    require_matches_legacy("H 0\nT 0\nEXP_VAL X0\nEXP_VAL Y0\nEXP_VAL Z0", 1, 2345);
    require_matches_legacy("H 0\nH 1\nT 0\nT 1\nCX 0 1\nEXP_VAL Y0*Z1", 1, 3456);
    require_matches_legacy("H 0\nM 0\nCX rec[-1] 1\nEXP_VAL Z1", 64, 4567);
    require_matches_legacy("X_ERROR(1) 0\nEXP_VAL Z0", 1, 5678);
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
    plan.symbols = {SymbolInfo{SymbolKind::Presampled, std::nullopt, std::nullopt}};
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

TEST_CASE("Sampling replay matches legacy record probabilities") {
    require_replay_matches_legacy("H 0\nM 0\nM 0\n", std::array<uint8_t, 4>{0, 0, 0, 1}, 2);
    require_replay_matches_legacy("H 0\nM 0\nM 0\n", std::array<uint8_t, 4>{1, 0, 1, 1}, 2);
    require_replay_matches_legacy("H 0\nT 0\nR_Z(0.3) 0\nM 0\n", std::array<uint8_t, 2>{0, 1}, 2);
    require_replay_matches_legacy("H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1\n",
                                  std::array<uint8_t, 2>{0, 1}, 2);
    require_replay_matches_legacy("H 0\nCX 0 1\nM 0 1\n",
                                  std::array<uint8_t, 8>{0, 0, 0, 1, 1, 0, 1, 1}, 4);
}

TEST_CASE("Sampling batched record probabilities match legacy replay") {
    constexpr std::string_view kCircuit = "H 0\nCX 0 1\nM 0 1\n";
    constexpr std::array<uint8_t, 8> kRecords{0, 0, 0, 1, 1, 0, 1, 1};
    const clifft::HirModule hir = clifft::trace(clifft::parse(kCircuit));
    const ExecutablePlan executable(clifft::sampling::plan_sampling(hir));

    const std::vector<double> actual = record_log_probabilities(executable, kRecords, 4);
    const std::vector<double> expected =
        clifft::record_probabilities(clifft::lower(hir), kRecords, 4);
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        if (expected[i] == clifft::kUnreachableLogProb) {
            REQUIRE(actual[i] == std::numeric_limits<double>::lowest());
        } else {
            REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 1e-12));
        }
    }
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
