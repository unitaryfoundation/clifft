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
    SharingExecutor growing(plan, noise, checks, 129, 1);
    SharingExecutor reserved(plan, noise, checks, 129, 129);
    SharingExecutor independent(plan, noise, checks, 129, 129, false);
    uint64_t boundaries = 0, splits = 0, rejected = 0;
    uint32_t largest_shared_group_count = 0;
    for (uint64_t seed = 0; seed < 20; ++seed) {
        for (uint32_t shots : {129, 0, 1, 63, 65}) {
            const auto root = seed_root_from_seed(seed);
            growing.run_batch(root, 17, shots);
            reserved.run_batch(root, 17, shots);
            independent.run_batch(root, 17, shots);
            boundaries += growing.storage_boundaries();
            splits += growing.num_splits();
            rejected += growing.entrance_rejected();
            largest_shared_group_count =
                std::max(largest_shared_group_count, growing.peak_groups());
            REQUIRE(growing.state_capacity() <= 129);
            for (const auto* reference : {&reserved, &independent}) {
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
