#include "clifft/api/sampling_preparation.h"
#include "clifft/api/survivor_sampler.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sharing_executor.h"
#include "clifft/util/runtime_isa.h"

#include "instrument_test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <string>

using namespace clifft;
using namespace clifft::sampling;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// Observes entrance rejection before any measurement draws or state evolution.
class EntranceProbe : public BatchClassicalState {
  public:
    EntranceProbe(const ExecutablePlan& plan, uint32_t capacity)
        : BatchClassicalState(plan, BatchOutputMode::AggregateSurvivors,
                              BatchSamplingMode::Ordinary,
                              batch_detail::batch_worker_storage_layout(
                                  plan, capacity, BatchOutputMode::AggregateSurvivors,
                                  BatchSamplingMode::Ordinary)) {}
    void run(const SeedRoot& root, uint32_t first, uint32_t shots,
             const EarlyPostselection& checks) {
        reset_classical(root, first, shots);
        sample_presampled_noise();
        apply_early_postselection(checks);
    }
    bool live(uint32_t lane) const { return is_live(lane); }
};

}  // namespace

TEST_CASE("Entrance postselection eliminates a shared measurement with reference offsets") {
    const bool select_both = GENERATE(false, true);
    auto hir = trace(parse(R"(
        H 0
        M 0
        X_ERROR(0.2) 1
        X_ERROR(0.3) 2
        M 1 2
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
    )"));
    hir.postselection.mask = {1, static_cast<uint8_t>(select_both)};
    hir.postselection.expected_detectors = {1, 0};
    const ExecutablePlan plan(plan_sampling(hir));
    const EarlyPostselection checks(plan);
    REQUIRE(checks.num_checks() == (select_both ? 1 : 0));
    EntranceProbe entrance(plan, 257);
    BatchExecutor reference(plan, 257);
    uint32_t rejected = 0;
    for (uint64_t seed : {0, 1, 731}) {
        const auto root = seed_root_from_seed(seed);
        for (uint32_t shots : {257, 0, 1, 63, 65, 257}) {
            entrance.run(root, 13, shots, checks);
            reference.run_batch(root, 13, shots);
            rejected += shots - entrance.surviving_shots();
            for (uint32_t lane = 0; lane < reference.surviving_shots(); ++lane) {
                REQUIRE(entrance.live(reference.shot_index(lane) - 13));
            }
        }
    }
    REQUIRE((rejected > 0) == select_both);
}

TEST_CASE("Entrance postselection detects a constant impossible normalized detector") {
    auto hir = trace(parse("M 0\nDETECTOR rec[-1]\n"));
    hir.postselection.mask = {1};
    hir.postselection.expected_detectors = {1};
    const ExecutablePlan plan(plan_sampling(hir));
    const EarlyPostselection checks(plan);
    REQUIRE(checks.num_checks() == 1);
    EntranceProbe entrance(plan, 65);
    entrance.run(seed_root_from_seed(17), 0, 65, checks);
    REQUIRE(entrance.surviving_shots() == 0);
}

TEST_CASE("Sharing storage boundaries preserve per-shot branches and classical correlations") {
    const double readout_probability = GENERATE(0.001, 0.1);
    const auto mode = GENERATE(BatchNoiseMode::Dense, BatchNoiseMode::Sparse,
                               BatchNoiseMode::SiteSparse, BatchNoiseMode::SiteRows);
    auto hir = trace(parse(R"(
        H 0
        T 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.05, 0.2) 0
        M(0.1) 0
        CX rec[-1] 1
        H 1
        T 1
        H 1
        M 1
        H 2
        T 2
        H 2
        M 2
        X_ERROR(0.2) 3
        M 3
        DETECTOR rec[-1]
        DETECTOR rec[-2] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(1) rec[-3]
        OBSERVABLE_INCLUDE(2) rec[-2]
    )"));
    hir.postselection.mask = {1, 1};
    hir.postselection.expected_detectors = {0, 1};
    hir.readout_noise.front().prob_zero_to_one = readout_probability;
    hir.readout_noise.front().prob_one_to_zero = readout_probability;
    default_hir_pass_manager().run(hir);
    const ExecutablePlan plan(plan_sampling(hir));
    REQUIRE(plan.peak_active_width() > 0);
    const PreparedBatchNoise noise(plan, mode);
    const EarlyPostselection checks(plan);
    REQUIRE(checks.num_checks() > 0);
    const EarlyPostselection late_checks(plan, false);
    SharingExecutor growing(plan, noise, checks, 129, 1);
    SharingExecutor reserved(plan, noise, checks, 129, 129);
    SharingExecutor independent(plan, noise, checks, 129, 129, false);
    SharingExecutor late(plan, noise, late_checks, 129, 1);
    const SharingKey key(plan);
    REQUIRE(key.num_bits() == 0);
    SharingExecutor grouped(plan, noise, checks, 129, 1, true, &key);
    uint64_t boundaries = 0, splits = 0, rejected = 0;
    uint32_t largest_shared_group_count = 0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        for (uint32_t shots : {129, 0, 1, 63, 65}) {
            const auto root = seed_root_from_seed(seed);
            growing.run_batch(root, 17, shots);
            reserved.run_batch(root, 17, shots);
            independent.run_batch(root, 17, shots);
            late.run_batch(root, 17, shots);
            grouped.run_batch(root, 17, shots);
            boundaries += growing.storage_boundaries();
            splits += growing.num_splits();
            rejected += growing.entrance_rejected();
            largest_shared_group_count =
                std::max(largest_shared_group_count, growing.peak_groups());
            REQUIRE(growing.state_capacity() <= 129);
            for (const auto* reference : {&reserved, &independent, &late, &grouped}) {
                REQUIRE(growing.surviving_shots() == reference->surviving_shots());
                REQUIRE(std::ranges::equal(growing.survivor_mask(), reference->survivor_mask()));
                const auto compare = [&](const PackedBitColumns& actual,
                                         const PackedBitColumns& expected) {
                    REQUIRE(actual.num_columns() == expected.num_columns());
                    for (size_t column = 0; column < actual.num_columns(); ++column) {
                        for (size_t word = 0; word < growing.survivor_mask().size(); ++word) {
                            const uint64_t difference =
                                actual.column(column)[word] ^ expected.column(column)[word];
                            REQUIRE((difference & growing.survivor_mask()[word]) == 0);
                        }
                    }
                };
                compare(growing.record_columns(), reference->record_columns());
                compare(growing.observable_columns(), reference->observable_columns());
            }
        }
    }
    REQUIRE(boundaries > 0);
    REQUIRE(splits > 0);
    REQUIRE(rejected > 0);
    REQUIRE(largest_shared_group_count < 129);
}

TEST_CASE("Selected detector scheduling preserves record snapshots and later readout") {
    const bool select_second = GENERATE(false, true);
    const SymbolId first_flip{0}, second_flip{1};
    SamplingPlan source;
    source.num_qubits = 1;
    source.num_visible_records = 4;
    source.num_detectors = 2;
    source.num_observables = 1;
    source.symbols = {SymbolKind::Readout, SymbolKind::Readout};
    source.actions = {
        {0, 0, RecordClassical{AffineBool{}, RecordSlot{0}}},
        {0, 0, RecordClassical{AffineBool{true}, RecordSlot{1}}},
        {0, 0, ApplyReadoutNoise{first_flip, AffineBool{}, RecordSlot{0}, 0.2, 0.4}},
        {0, 0, RecordClassical{AffineBool{}, RecordSlot{2}}},
        {0, 0,
         WriteDetector{RecordParity{true, {RecordSlot{0}, RecordSlot{1}}}, DetectorSlot{0}, true}},
        {0, 0,
         ApplyReadoutNoise{second_flip, AffineBool::symbol(first_flip), RecordSlot{0}, 0.3, 0.7}},
        {0, 0, RecordClassical{AffineBool{}, RecordSlot{3}}},
        {0, 0,
         WriteDetector{RecordParity{true, {RecordSlot{0}, RecordSlot{1}}}, DetectorSlot{1},
                       select_second}},
        {0, 0, WriteObservable{RecordParity{false, {RecordSlot{0}}}, ObservableSlot{0}}},
    };
    const ExecutablePlan plan(source);
    const EarlyPostselection checks(plan), late_checks(plan, false);
    const std::vector<size_t> expected = select_second
                                             ? std::vector<size_t>{0, 1, 2, 4, 3, 5, 7, 6, 8}
                                             : std::vector<size_t>{0, 1, 2, 4, 3, 5, 6, 7, 8};
    REQUIRE(std::ranges::equal(checks.action_order(), expected));
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    SharingExecutor scheduled(plan, noise, checks, 129, 1);
    SharingExecutor late(plan, noise, late_checks, 129, 1);
    for (uint64_t seed = 0; seed < 20; ++seed) {
        const auto root = seed_root_from_seed(seed);
        scheduled.run_batch(root, 0, 129);
        late.run_batch(root, 0, 129);
        REQUIRE(std::ranges::equal(scheduled.survivor_mask(), late.survivor_mask()));
        const auto actual = scheduled.observable_columns().column(0);
        const auto reference = late.observable_columns().column(0);
        for (size_t word = 0; word < actual.size(); ++word) {
            REQUIRE(((actual[word] ^ reference[word]) & scheduled.survivor_mask()[word]) == 0);
        }
    }
}

TEST_CASE("Shared measurement and readout preserve the selected interference law") {
    const bool reference_one = GENERATE(false, true);
    const double readout_probability = GENERATE(0.05, 0.1);
    auto hir = trace(parse(R"(
        H 0
        T 0
        Z_ERROR(0.15) 0
        H 0
        M 0
        CX rec[-1] 1
        M(0.1) 1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    )"));
    hir.postselection.mask = {1};
    hir.postselection.expected_detectors = {static_cast<uint8_t>(reference_one)};
    hir.readout_noise.front().prob_zero_to_one = readout_probability;
    hir.readout_noise.front().prob_one_to_zero = readout_probability;
    const ExecutablePlan plan(plan_sampling(hir));
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const EarlyPostselection checks(plan);
    SharingExecutor worker(plan, noise, checks, 257, 1);
    constexpr uint32_t shots = 65536;
    std::array<uint64_t, 5> observed{};
    const auto root = seed_root_from_seed(457);
    for (uint32_t first = 0; first < shots; first += 257) {
        const uint32_t count = std::min(uint32_t{257}, shots - first);
        worker.run_batch(root, first, count);
        observed.back() += count - worker.surviving_shots();
        for (uint32_t lane = 0; lane < count; ++lane) {
            if (((worker.survivor_mask()[lane >> 6] >> (lane & 63)) & 1) != 0) {
                const uint32_t key = uint32_t(worker.observable_columns().bit(0, lane)) |
                                     (uint32_t(worker.observable_columns().bit(1, lane)) << 1);
                ++observed[key];
            }
        }
    }
    const double one = 0.85 * (1 - std::sqrt(0.5)) / 2 + 0.15 * (1 + std::sqrt(0.5)) / 2;
    const double p = readout_probability;
    const std::array<double, 5> probabilities =
        reference_one ? std::array<double, 5>{0, one * p, (1 - one) * p, 0, 1 - p}
                      : std::array<double, 5>{(1 - one) * (1 - p), 0, 0, one * (1 - p), p};
    for (size_t bin = 0; bin < probabilities.size(); ++bin) {
        CAPTURE(reference_one, bin);
        if (probabilities[bin] == 0) {
            REQUIRE(observed[bin] == 0);
        } else {
            const double expected = shots * probabilities[bin];
            const double tolerance = 7 * std::sqrt(expected * (1 - probabilities[bin])) + 10;
            REQUIRE_THAT(static_cast<double>(observed[bin]),
                         Catch::Matchers::WithinAbs(expected, tolerance));
        }
    }
}

TEST_CASE("Shared wide Pauli kernels preserve interference and collapsed coordinates") {
    const uint32_t pivot = GENERATE(0, 5);
    const uint32_t axis = GENERATE(0, 1, 2);
    const uint64_t bit = uint64_t{1} << pivot;
    const ActivePauli pauli = axis == 0   ? ActivePauli{bit, 0}
                              : axis == 1 ? ActivePauli{bit, bit}
                                          : ActivePauli{0, bit};
    const SymbolId first{0}, second{1};
    SamplingPlan source;
    source.num_qubits = source.initial_active_width = source.peak_active_width = 6;
    source.num_visible_records = source.num_observables = 2;
    source.symbols = {SymbolKind::Branch, SymbolKind::Branch};
    source.actions = {
        {6, 6, RotateActivePauli{ActivePauli{bit, 0}, 0.25, AffineBool{}}},
        {6, 6, RotateActivePauli{ActivePauli{0, bit}, 0.25, AffineBool{}}},
        {6, 5, MeasureActivePauli{pauli, pivot, first, AffineBool::symbol(first), RecordSlot{0}}},
        {5, 4,
         MeasureActivePauli{ActivePauli{0, 1}, 0, second, AffineBool::symbol(second),
                            RecordSlot{1}}},
        {4, 4, WriteObservable{RecordParity{false, {RecordSlot{0}}}, ObservableSlot{0}}},
        {4, 4, WriteObservable{RecordParity{false, {RecordSlot{1}}}, ObservableSlot{1}}},
    };
    const ExecutablePlan plan(source);
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const EarlyPostselection checks(plan);
    SharingExecutor worker(plan, noise, checks, 257, 1);
    constexpr uint32_t shots = 65536;
    std::array<uint64_t, 2> ones{};
    const auto root = seed_root_from_seed(134);
    uint64_t splits = 0;
    for (uint32_t offset = 0; offset < shots; offset += 257) {
        worker.run_batch(root, offset, std::min(uint32_t{257}, shots - offset));
        (void)worker.accumulate_survivor_counts(ones);
        splits += worker.num_splits();
    }
    // RX(pi/4) followed by RZ(pi/4) has Bloch vector (1/2, -1/2, sqrt(1/2)).
    const double probability = axis == 0 ? 0.25 : axis == 1 ? 0.75 : (1 - std::sqrt(0.5)) / 2;
    const double tolerance = 7 * std::sqrt(shots * probability * (1 - probability)) + 10;
    REQUIRE_THAT(static_cast<double>(ones[0]),
                 Catch::Matchers::WithinAbs(shots * probability, tolerance));
    REQUIRE(ones[1] == 0);
    REQUIRE(splits > 0);
}

TEST_CASE("Compact outputs preserve snapshots and feedback across noise preparations") {
    const auto mode = GENERATE(BatchNoiseMode::Dense, BatchNoiseMode::Sparse,
                               BatchNoiseMode::SiteSparse, BatchNoiseMode::SiteRows);
    const bool early = GENERATE(false, true);
    auto hir = trace(parse(R"(
        H 0
        T 0
        PAULI_CHANNEL_1(0.1, 0.05, 0.2) 0
        H 0
        M 0
        DETECTOR rec[-1]
        READOUT_NOISE(0.2, 0.4) rec[-1]
        CX rec[-1] 1
        H 1
        T 1
        H 1
        M 1
        H 2
        M 2
        CX rec[-1] 3
        H 3
        T 3
        H 3
        M 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
    )"));
    hir.postselection.mask = {1, 1};
    hir.postselection.expected_detectors = {1, 0};
    const Program program{ExecutablePlan(plan_sampling(hir))};
    const auto& rows = program.sampling_preparation(true, {.early_postselection = early});
    const auto& counts = program.sampling_preparation(false, {.early_postselection = early});
    const auto& original = rows.plan();
    const auto& compact = counts.plan();
    REQUIRE(&original == &program.sampling_plan(true));
    REQUIRE(compact.num_batch_noise_carriers() == 0);
    REQUIRE(original.num_visible_records() > 0);
    REQUIRE(compact.num_visible_records() == 0);
    REQUIRE(compact.num_hidden_records() < original.num_visible_records());
    REQUIRE(compact.estimated_batch_lane_work(BatchOutputMode::Rows) ==
            original.estimated_batch_lane_work(BatchOutputMode::Rows));
    const auto& old_noise = rows.noise(mode);
    const auto& noise = counts.noise(mode);
    const auto& old_checks = rows.checks();
    const auto& checks = counts.checks();
    const auto& key = counts.sharing_key();
    SharingExecutor reference(original, old_noise, old_checks, 129, 1);
    SharingExecutor worker(compact, noise, checks, 129, 1);
    SharingExecutor grouped(compact, noise, checks, 129, 1, true, &key);
    for (uint64_t seed = 0; seed < 12; ++seed) {
        for (uint32_t shots : {129, 0, 1, 63, 65}) {
            const auto root = seed_root_from_seed(seed);
            reference.run_batch(root, 17, shots);
            for (auto* candidate : {&worker, &grouped}) {
                candidate->run_batch(root, 17, shots);
                REQUIRE(std::ranges::equal(candidate->survivor_mask(), reference.survivor_mask()));
                for (size_t column = 0; column < compact.num_observables(); ++column) {
                    const auto actual = candidate->observable_columns().column(column);
                    const auto expected = reference.observable_columns().column(column);
                    for (size_t word = 0; word < actual.size(); ++word) {
                        REQUIRE(((actual[word] ^ expected[word]) &
                                 reference.survivor_mask()[word]) == 0);
                    }
                }
            }
        }
    }
}

TEST_CASE("Noise grouping closes derived signs over readout and dormant branches") {
    const bool compact_outputs = GENERATE(false, true);
    const bool symmetric = GENERATE(false, true);
    const SymbolId noise_symbol{0}, dormant{1}, flip{2}, first_sign{3}, sign{4}, active{5};
    SamplingPlan source;
    source.num_qubits = 7;
    source.initial_active_width = source.peak_active_width = 6;
    source.num_visible_records = source.num_observables = 2;
    source.symbols = {SymbolKind::Presampled, SymbolKind::Branch,  SymbolKind::Readout,
                      SymbolKind::Derived,    SymbolKind::Derived, SymbolKind::Branch};
    source.presampled_noise_sites = {{0.2, {{noise_symbol, 0.2}}}};
    source.actions = {
        {6, 6, MeasureDormantRandom{6, dormant, AffineBool::symbol(dormant), RecordSlot{0}}},
        {6, 6,
         ApplyReadoutNoise{flip, AffineBool::symbol(dormant), RecordSlot{0}, symmetric ? 0.01 : 0.2,
                           symmetric ? 0.01 : 0.4}},
        {6, 6, DefineSymbol{first_sign, AffineBool(false, {noise_symbol, dormant})}},
        {6, 6, DefineSymbol{sign, AffineBool(true, {first_sign, flip})}},
    };
    const std::array rotations = {
        RotateActivePauli{{0b001000, 0b000101}, 0.17, AffineBool::symbol(first_sign)},
        RotateActivePauli{{0b100000, 0b010010}, -0.23, AffineBool::symbol(sign)},
        RotateActivePauli{{0b101000, 0b100001}, 0.31, AffineBool(false, {first_sign, sign})},
        RotateActivePauli{{0b001000, 0b001110}, -0.11, AffineBool(true, {first_sign})},
        RotateActivePauli{{0b100000, 0b011001}, 0.29, AffineBool(true, {sign})},
        RotateActivePauli{{0b101000, 0b100100}, -0.37, AffineBool(true, {first_sign, sign})},
        RotateActivePauli{{0b001000, 0b010101}, 0.13, AffineBool(false)},
        RotateActivePauli{{0b100000, 0b101010}, -0.19, AffineBool::symbol(first_sign)},
    };
    for (const auto& rotation : rotations) {
        source.actions.push_back({6, 6, rotation});
    }
    source.actions.push_back({6, 5,
                              MeasureActivePauli{ActivePauli{0, 1}, 0, active,
                                                 AffineBool::symbol(active), RecordSlot{1}}});
    source.actions.push_back(
        {5, 5, WriteObservable{RecordParity{false, {RecordSlot{0}}}, ObservableSlot{0}}});
    source.actions.push_back(
        {5, 5, WriteObservable{RecordParity{false, {RecordSlot{1}}}, ObservableSlot{1}}});
    const ExecutablePlan original(source);
    if (internal::runtime_isa() == internal::RuntimeIsa::Avx512) {
        REQUIRE(original.num_actions() < source.actions.size());
    }
    const auto plan = compact_outputs ? original.compact_outputs() : original;
    const SharingKey key(plan);
    REQUIRE(key.num_bits() == 1);
    const PreparedBatchNoise old_noise(original, BatchNoiseMode::SiteSparse);
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const EarlyPostselection old_checks(original), checks(plan);
    SharingExecutor reference(original, old_noise, old_checks, 129, 1);
    SharingExecutor grouped(plan, noise, checks, 129, 1, true, &key);
    uint32_t largest_group_count = 0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        const auto root = seed_root_from_seed(seed);
        reference.run_batch(root, 17, 129);
        grouped.run_batch(root, 17, 129);
        REQUIRE(grouped.peak_groups() >= 4);
        largest_group_count = std::max(largest_group_count, grouped.peak_groups());
        for (size_t column = 0; column < 2; ++column) {
            REQUIRE(std::ranges::equal(reference.observable_columns().column(column),
                                       grouped.observable_columns().column(column)));
        }
    }
    // Readout splits the four noise/dormant groups for at least one seed.
    REQUIRE(largest_group_count > 4);
}

TEST_CASE("Shared state budgets include scratch and bound growth at suspension") {
    REQUIRE(State::storage_bytes(19) == 12 * 1024 * 1024);
    const ExecutablePlan plan(plan_sampling(trace(parse("H 0\nT 0\nH 0\nM 0"))));
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const EarlyPostselection checks(plan);
    const auto bytes = State::storage_bytes(plan.peak_active_width());
    REQUIRE_THROWS_AS(SharingExecutor(plan, noise, checks, 65, 1, true, nullptr, bytes - 1),
                      std::invalid_argument);
    SharingExecutor worker(plan, noise, checks, 65, 128, false, nullptr, 2 * bytes);
    REQUIRE(worker.state_capacity() == 2);
    REQUIRE_THROWS_AS(worker.run_batch(seed_root_from_seed(19), 0, 65), std::runtime_error);
    REQUIRE(worker.state_capacity() == 2);
    worker.run_batch(seed_root_from_seed(19), 0, 1);
    REQUIRE(worker.surviving_shots() == 1);
}

namespace {

// Hadamard-conjugated T gates on separate qubits hold all of them active at once.
ExecutablePlan parallel_t_plan(uint32_t width) {
    std::string qubits;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        qubits += " " + std::to_string(qubit);
    }
    ExecutablePlan plan(plan_sampling(
        trace(parse("H" + qubits + "\nT" + qubits + "\nH" + qubits + "\nM" + qubits))));
    REQUIRE(plan.peak_active_width() == width);
    return plan;
}

}  // namespace

TEST_CASE("Safe shared capacity gives every lane a state within one worker budget") {
    const auto [width, expected] = GENERATE(table<uint32_t, uint32_t>(
        {{1, kMaxAutoSharedBatchShots}, {8, kMaxAutoSharedBatchShots}, {10, 2730}, {13, 341}}));
    const ExecutablePlan plan = parallel_t_plan(width);
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const uint32_t capacity =
        max_safe_shared_capacity(plan, noise, nullptr, kDefaultSharedWorkerStateBudget);
    REQUIRE(capacity == expected);
    const uint64_t state_bytes = State::storage_bytes(width);
    REQUIRE(capacity * state_bytes <= kDefaultSharedWorkerStateBudget);
    REQUIRE(max_safe_shared_capacity(plan, noise, nullptr, 3 * state_bytes) == 3);
    REQUIRE(max_safe_shared_capacity(plan, noise, nullptr, state_bytes - 1) == 0);
}

TEST_CASE("Safe shared capacity halves until classical buffers fit their bound") {
    // Each observable keeps one packed column, about 1 KiB at 8192 lanes.
    std::string text = "H 0\nT 0\nH 0\nM 0\n";
    for (uint32_t observable = 0; observable < 70000; ++observable) {
        text += "OBSERVABLE_INCLUDE(" + std::to_string(observable) + ") rec[-1]\n";
    }
    const ExecutablePlan plan(plan_sampling(trace(parse(text))));
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const uint32_t capacity =
        max_safe_shared_capacity(plan, noise, nullptr, kDefaultSharedWorkerStateBudget);
    REQUIRE(capacity == kMaxAutoSharedBatchShots / 2);
    REQUIRE(SharingExecutor::classical_storage_bytes(plan, noise, nullptr, capacity) <=
            kMaxAutoSharedWorkerClassicalBytes);
    REQUIRE(SharingExecutor::classical_storage_bytes(plan, noise, nullptr, 2 * capacity) >
            kMaxAutoSharedWorkerClassicalBytes);

    // At a small capacity, the per-action split factors dominate the lane scratch.
    const uint64_t retained_bytes = packed_bit_columns_storage_bytes(plan.num_observables(), 64) +
                                    plan.num_actions() * sizeof(uint32_t);
    REQUIRE(SharingExecutor::classical_storage_bytes(plan, noise, nullptr, 64) >= retained_bytes);
}

TEST_CASE("Safe shared capacity includes the worker observable counters") {
    // Executor buffers fit at 8192 lanes, but the sampler's counters exceed the remaining budget.
    const ExecutablePlan plan(
        plan_sampling(trace(parse("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(64299) rec[-1]\n"))));
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    REQUIRE(
        SharingExecutor::classical_storage_bytes(plan, noise, nullptr, kMaxAutoSharedBatchShots) <=
        kMaxAutoSharedWorkerClassicalBytes);
    const uint32_t capacity =
        max_safe_shared_capacity(plan, noise, nullptr, kDefaultSharedWorkerStateBudget);
    REQUIRE(capacity == kMaxAutoSharedBatchShots / 2);
}

TEST_CASE("Automatic shared policy sizes batches to the call and its workers") {
    struct Case {
        uint32_t safe, shots, workers, capacity, used;
    };
    const auto c = GENERATE(values<Case>({
        // Long calls fill the safe capacity on every worker.
        {8192, 1 << 20, 8, 8192, 8},
        {8192, 65536, 8, 8192, 8},
        {341, 65536, 8, 341, 8},
        // An even share gives each requested worker one batch.
        {8192, 16384, 8, 2048, 8},
        {8192, 16385, 8, 2049, 8},
        // Below the lane floor, fewer workers keep fuller batches.
        {8192, 4096, 8, kMinAutoSharedLanesPerWorker, 4},
        {8192, 100, 8, 100, 1},
        {8192, 100, 1, 100, 1},
        // The safe capacity still bounds a single worker.
        {5, 100, 1, 5, 1},
    }));
    const auto policy = resolve_shared_execution_policy(c.safe, c.shots, c.workers);
    REQUIRE(policy.lane_capacity == c.capacity);
    REQUIRE(policy.worker_count == c.used);
}

TEST_CASE("Shared tuning candidates start at the baseline and stay within the safe capacity") {
    const auto [safe, shots, workers] =
        GENERATE(table<uint32_t, uint32_t, uint32_t>({{8192, 16384, 8},
                                                      {8192, 1 << 20, 8},
                                                      {8192, 1 << 20, 9},
                                                      {8192, 1 << 20, 16},
                                                      {341, 65536, 4},
                                                      {8192, 100, 2},
                                                      {1, 100, 1}}));
    const auto candidates = shared_tuning_candidates(safe, shots, workers);
    const auto baseline = resolve_shared_execution_policy(safe, shots, workers);
    REQUIRE_FALSE(candidates.empty());
    REQUIRE(candidates.front().policy.lane_capacity == baseline.lane_capacity);
    REQUIRE(candidates.front().policy.worker_count == baseline.worker_count);
    const uint32_t limit = std::min(safe, shots);
    bool reaches_limit = false;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& policy = candidates[i].policy;
        REQUIRE(candidates[i].min_probe_shots <= candidates[i].max_probe_shots);
        REQUIRE(candidates[i].max_probe_shots <= shots);
        REQUIRE(candidates[i].max_probe_shots <= std::max(65536u, candidates[i].min_probe_shots));
        REQUIRE(candidates[i].min_probe_shots >=
                std::min<uint64_t>(shots, uint64_t{policy.lane_capacity} * policy.worker_count));
        REQUIRE(policy.lane_capacity <= limit);
        REQUIRE(policy.worker_count <= workers);
        REQUIRE(uint64_t{policy.lane_capacity} * (policy.worker_count - 1) < shots);
        reaches_limit |= policy.lane_capacity == limit;
        for (size_t j = 0; j < i; ++j) {
            REQUIRE(candidates[j].policy.lane_capacity != policy.lane_capacity);
        }
    }
    REQUIRE(reaches_limit);
    REQUIRE(shared_tuning_candidates(safe, 0, workers).empty());
}

TEST_CASE("Automatic survivor sampling replays the per-worker capacity at any worker count") {
    const auto circuit = parse(
        "H 0 1\nT 0 1\nDEPOLARIZE1(0.04) 0 1\nH 0 1\nM(0.13) 0 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]");
    const Program program{ExecutablePlan(plan_sampling(trace(circuit)))};
    const uint32_t width = program.sampling_plan(false).peak_active_width();
    // Seven states fit each worker, however many workers share the call.
    const uint64_t budget = 7 * State::storage_bytes(width);
    const uint32_t threads = GENERATE(1u, 2u, 5u);
    const auto mode = BatchNoiseMode::SiteSparse;
    SurvivorSampler automatic(program, std::nullopt, threads, 128, mode, true, true, false, budget);
    SurvivorSampler numeric(program, 7, threads, 128, mode, true, true, false, budget);
    const auto expected = numeric.sample(1301, 31);
    const auto result = automatic.sample(1301, 31);
    REQUIRE(result.passed_shots == expected.passed_shots);
    REQUIRE(result.observable_ones == expected.observable_ones);
    REQUIRE_THROWS_AS(SurvivorSampler(program, std::nullopt, threads, 128, mode, true, true, false,
                                      State::storage_bytes(width) - 1),
                      std::invalid_argument);
}

TEST_CASE("Survivor construction rejects instruments in each preparation path") {
    const auto options = test::source_dependent_jump_options();
    const auto hir = trace(parse("LEVEL_TRANSITION[jump] 0"), &options);
    const Program program{ExecutablePlan(plan_sampling(hir))};
    const bool early = GENERATE(false, true);
    REQUIRE_THROWS_AS(SurvivorSampler(program, 65, 1, 1, BatchNoiseMode::SiteSparse, true, early),
                      std::invalid_argument);
}

TEST_CASE("Compact count plans omit unselected detectors and expectations") {
    auto hir = trace(
        parse("H 0\nT 0\nEXP_VAL X0\nH 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"));
    const ExecutablePlan original(plan_sampling(hir));
    const auto compact = original.compact_outputs();
    REQUIRE(compact.num_actions() < original.num_actions());
    REQUIRE(compact.num_hidden_records() == 1);
    REQUIRE(compact.estimated_batch_lane_work(BatchOutputMode::Rows) ==
            original.estimated_batch_lane_work(BatchOutputMode::AggregateSurvivors));
}

TEST_CASE("Pregrouped rotations retain their existing state budget") {
    SamplingPlan source;
    source.num_qubits = source.initial_active_width = source.peak_active_width = 1;
    source.symbols = {SymbolKind::Presampled};
    source.presampled_noise_sites = {{0.5, {{SymbolId{0}, 0.5}}}};
    source.actions = {{1, 1, RotateActivePauli{{1, 0}, 0.17, AffineBool::symbol(SymbolId{0})}}};
    const ExecutablePlan plan(source);
    const PreparedBatchNoise noise(plan, BatchNoiseMode::SiteSparse);
    const EarlyPostselection checks(plan);
    const SharingKey key(plan);
    SharingExecutor worker(plan, noise, checks, 65, 1, true, &key, 2 * State::storage_bytes(1));
    worker.run_batch(seed_root_from_seed(19), 0, 65);
    REQUIRE(worker.peak_groups() == 2);
    REQUIRE(worker.state_capacity() == 2);
    REQUIRE(worker.surviving_shots() == 65);
}

TEST_CASE("Dense noise preparation builds the presampled program for compact plans") {
    auto hir = trace(parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/surface_d7_r7_p001.stim"));
    hir.postselection.mask.assign(hir.num_detectors, 1);
    default_hir_pass_manager().run(hir);
    const Program program{ExecutablePlan(plan_sampling(hir))};
    const auto& compiled = program.sampling_plan(false);
    REQUIRE(compiled.num_batch_noise_carriers() > 0);

    const auto& counts = program.sampling_preparation(false);
    REQUIRE(counts.plan().num_batch_noise_carriers() == 0);
    REQUIRE(counts.noise(BatchNoiseMode::Dense).num_carriers() > 0);
    REQUIRE(counts.noise(BatchNoiseMode::Sparse).num_carriers() == 0);
    const auto& rows = program.sampling_preparation(true);
    REQUIRE(&program.sampling_plan(true) == &compiled);
    REQUIRE(rows.noise(BatchNoiseMode::Dense).num_carriers() ==
            compiled.num_batch_noise_carriers());

    BatchExecutor ordinary(compiled, 257, BatchOutputMode::AggregateSurvivors,
                           BatchSamplingMode::Ordinary);
    BatchExecutor dense(counts.plan(), 257, BatchOutputMode::AggregateSurvivors,
                        BatchSamplingMode::Ordinary, &counts.noise(BatchNoiseMode::Dense));
    BatchExecutor sparse(counts.plan(), 257, BatchOutputMode::AggregateSurvivors,
                         BatchSamplingMode::Ordinary, &counts.noise(BatchNoiseMode::Sparse));
    BatchExecutor full(rows.plan(), 257, BatchOutputMode::AggregateSurvivors,
                       BatchSamplingMode::Ordinary, &rows.noise(BatchNoiseMode::Dense));
    uint64_t survivors = 0, rejected = 0;
    for (uint64_t seed : {0, 1, 731}) {
        const auto root = seed_root_from_seed(seed);
        for (uint32_t shots : {257, 0, 1, 63, 65, 257}) {
            ordinary.run_batch(root, 13, shots);
            survivors += ordinary.surviving_shots();
            rejected += shots - ordinary.surviving_shots();
            for (auto* candidate : {&dense, &sparse, &full}) {
                candidate->run_batch(root, 13, shots);
                REQUIRE(candidate->surviving_shots() == ordinary.surviving_shots());
                REQUIRE(std::ranges::equal(candidate->survivor_mask(), ordinary.survivor_mask()));
                for (size_t column = 0; column < compiled.num_observables(); ++column) {
                    const auto actual = candidate->observable_columns().column(column);
                    const auto expected = ordinary.observable_columns().column(column);
                    for (size_t word = 0; word < actual.size(); ++word) {
                        REQUIRE(((actual[word] ^ expected[word]) &
                                 ordinary.survivor_mask()[word]) == 0);
                    }
                }
            }
        }
    }
    REQUIRE(survivors > 0);
    REQUIRE(rejected > 0);
}

TEST_CASE("Shared tuning bounds retained output allocations") {
    constexpr uint64_t row_bytes = 4096;
    constexpr uint64_t budget = 8 * 1024 * 1024;
    const auto candidates = shared_tuning_candidates(1024, 65536, 2, row_bytes);
    REQUIRE(candidates.size() > 1);
    for (const auto& candidate : candidates) {
        REQUIRE(candidate.min_probe_shots <= candidate.max_probe_shots);
        REQUIRE(candidate.max_probe_shots * row_bytes <= budget);
    }
    REQUIRE(shared_tuning_candidates(8192, 65536, 2, row_bytes).empty());
    REQUIRE(shared_tuning_candidates(8192, 256, 1, budget + 1).empty());
}
