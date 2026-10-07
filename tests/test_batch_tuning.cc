#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <set>

using namespace clifft::sampling;
using clifft::sampling::batch_detail::batch_tuning_candidates;
using clifft::sampling::batch_detail::BatchTuningCandidate;
using clifft::sampling::batch_detail::sweep_batch_candidates;

namespace {

ExecutablePlan compile_tuning_plan(const char* source) {
    return ExecutablePlan(plan_sampling(clifft::trace(clifft::parse(source))));
}

}  // namespace

TEST_CASE("Batch tuning selects measured throughput and accounts for discarded trials") {
    const std::array<BatchTuningCandidate, 3> candidates{
        {{{1, 2}, 2, 2}, {{64, 2}, 128, 128}, {{256, 2}, 512, 512}}};
    double clock = 0;
    uint64_t attempted = 0;
    std::vector<uint32_t> order;
    std::set<uint64_t> seeds;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.3, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy policy) {
            order.push_back(policy.lane_capacity);
            clock += 0.002;
            return [&, policy](uint32_t shots, uint64_t seed) {
                REQUIRE(seeds.insert(seed).second);
                attempted += shots;
                const double rate = policy.lane_capacity == 256  ? 100000
                                    : policy.lane_capacity == 64 ? 50000
                                                                 : 1000;
                clock += shots / rate;
            };
        });
    REQUIRE(order == std::vector<uint32_t>{1, 64, 256});
    REQUIRE(report.batch_size == 256);
    REQUIRE(report.baseline_batch_size == 1);
    REQUIRE(report.shot_workers == 2);
    REQUIRE(report.trial_shots == attempted);
    REQUIRE(report.elapsed_seconds == clock);
    REQUIRE(report.trials.size() == 3);
    uint64_t reported = 0;
    for (const auto& trial : report.trials) {
        REQUIRE(trial.shots > 0);
        REQUIRE(trial.setup_seconds > 0);
        reported += trial.shots + trial.warmup_shots;
    }
    REQUIRE(reported == attempted);
}

TEST_CASE("Batch tuning stops after an overlong warmup and retains the baseline") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{64, 1}, 64, 64}, {{1, 1}, 1, 1}}};
    double clock = 0;
    size_t prepared = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.01, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy) {
            ++prepared;
            return [&](uint32_t, uint64_t) { clock += 0.1; };
        });
    REQUIRE(prepared == 1);
    REQUIRE(report.batch_size == 64);
    REQUIRE(report.trial_shots == 64);
    REQUIRE(report.trials.front().shots == 0);
    REQUIRE(report.stop_reason == "insufficient_measurements");
}

TEST_CASE("Batch tuning charges preparation to the budget before starting shots") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{1, 1}, 1, 1}, {{64, 1}, 64, 64}}};
    double clock = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.01, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy) {
            clock += 0.02;
            return [&](uint32_t, uint64_t) { FAIL("expired calibration must not run shots"); };
        });
    REQUIRE(report.batch_size == 1);
    REQUIRE(report.trial_shots == 0);
    REQUIRE(report.elapsed_seconds == 0.02);
}

TEST_CASE("Batch tuning retains the default when measured rates tie") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{64, 1}, 64, 64}, {{1, 1}, 1, 1}}};
    double clock = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 16, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy) {
            return [&](uint32_t shots, uint64_t) { clock += static_cast<double>(shots) / 64; };
        });
    REQUIRE(report.trials.size() == 2);
    REQUIRE(report.batch_size == 64);
}

TEST_CASE("Batch tuning explores above the automatic width cutoff with bounded storage") {
    const auto plan = compile_tuning_plan("H 0 1 2 3 4 5\nT 0 1 2 3 4 5\nM 0 1 2 3 4 5");
    REQUIRE(plan.peak_active_width() == 6);
    auto baseline =
        resolve_batch_execution_policy(plan, 8193, 2, 1, BatchOutputMode::Rows, std::nullopt);
    REQUIRE(baseline.lane_capacity == 1);
    baseline.worker_count = 2;
    const auto candidates = batch_tuning_candidates(plan, 8193, 2, 1, BatchOutputMode::Rows,
                                                    BatchSamplingMode::Ordinary, 0, baseline);
    REQUIRE(candidates.size() == 5);
    for (const auto& candidate : candidates) {
        REQUIRE(candidate.min_probe_shots >= candidate.policy.worker_count);
        REQUIRE(candidate.max_probe_shots <= 8193);
        if (candidate.policy.lane_capacity > 1) {
            const auto bytes = batch_detail::batch_worker_storage_bytes(
                plan, candidate.policy.lane_capacity, BatchOutputMode::Rows,
                BatchSamplingMode::Ordinary);
            REQUIRE(bytes <= kDefaultBatchWorkerBudget);
            REQUIRE(bytes * candidate.policy.worker_count <= kDefaultBatchTotalWorkerBudget);
        }
    }
    const auto constrained = batch_tuning_candidates(plan, 8193, 2, 1, BatchOutputMode::Rows,
                                                     BatchSamplingMode::FixedFaults,
                                                     kDefaultBatchWorkerBudget, baseline);
    REQUIRE(constrained.size() == 1);
    REQUIRE(constrained.front().policy.lane_capacity == 1);
    REQUIRE(batch_tuning_candidates(plan, 8193, 2, 2, BatchOutputMode::Rows,
                                    BatchSamplingMode::Ordinary, 0, baseline)
                .size() == 1);
}

TEST_CASE("Batch tuning deduplicates clipped capacities and respects total worker memory") {
    const auto plan = compile_tuning_plan("H 0\nT 0\nM 0");
    const auto small = batch_tuning_candidates(plan, 7, 2, 1, BatchOutputMode::Rows,
                                               BatchSamplingMode::Ordinary, 0, {1, 2});
    REQUIRE(small.size() == 2);
    REQUIRE(small.back().policy.lane_capacity == 7);
    REQUIRE(small.back().policy.worker_count == 1);
    REQUIRE(small.back().min_probe_shots == 7);
    const auto constrained = batch_tuning_candidates(plan, 100000, 16, 1, BatchOutputMode::Rows,
                                                     BatchSamplingMode::Ordinary,
                                                     kDefaultBatchWorkerBudget / 2, {1, 16});
    REQUIRE(constrained.size() == 1);
}

TEST_CASE("Batch tuning validates budgets even for empty sampling calls") {
    const auto plan = compile_tuning_plan("M 0");
    for (double budget : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
        REQUIRE_THROWS_AS(sample(plan, 0, 42, 1, std::nullopt, std::nullopt, budget),
                          std::invalid_argument);
    }
    REQUIRE_THROWS_AS(sample(plan, 0, 42, 1, std::nullopt, 64, 0.1), std::invalid_argument);
    const auto empty = sample(plan, 0, 42, 1, std::nullopt, std::nullopt, 0.1);
    REQUIRE(empty.measurements.empty());
    REQUIRE(empty.batch_tuning->stop_reason == "zero_shots");
    REQUIRE(empty.batch_tuning->trial_shots == 0);
    const auto zero_budget = sample(plan, 65, 42, 1, std::nullopt, std::nullopt, 0.0);
    REQUIRE(zero_budget.batch_tuning->stop_reason == "zero_budget");
    REQUIRE(zero_budget.measurements == sample(plan, 65, 42).measurements);
}
