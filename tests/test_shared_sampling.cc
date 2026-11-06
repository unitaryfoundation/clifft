#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sharing_executor.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace clifft;
using namespace clifft::sampling;

namespace {

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
    uint64_t boundaries = 0, splits = 0, rejected = 0;
    uint32_t largest_shared_group_count = 0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        for (uint32_t shots : {129, 0, 1, 63, 65}) {
            const auto root = seed_root_from_seed(seed);
            growing.run_batch(root, 17, shots);
            reserved.run_batch(root, 17, shots);
            independent.run_batch(root, 17, shots);
            late.run_batch(root, 17, shots);
            boundaries += growing.storage_boundaries();
            splits += growing.num_splits();
            rejected += growing.entrance_rejected();
            largest_shared_group_count =
                std::max(largest_shared_group_count, growing.peak_groups());
            REQUIRE(growing.state_capacity() <= 129);
            for (const auto* reference : {&reserved, &independent, &late}) {
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
    const std::array<double, 5> probabilities =
        reference_one ? std::array<double, 5>{0, one * 0.1, (1 - one) * 0.1, 0, 0.9}
                      : std::array<double, 5>{(1 - one) * 0.9, 0, 0, one * 0.9, 0.1};
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
