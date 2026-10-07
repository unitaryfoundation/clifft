#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
    REQUIRE(report.sufficient_measurements);
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

TEST_CASE("Batch tuning reports budget exhaustion when only the baseline can be measured") {
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
    REQUIRE(report.trials.front().used_warmup());
    REQUIRE(report.trials.front().shots_per_second() > 0);
    REQUIRE_FALSE(report.sufficient_measurements);
    REQUIRE(report.stop_reason == "budget_exhausted");
}

TEST_CASE("Batch tuning can compare first probes without steady state measurements") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{1, 1}, 1, 1}, {{64, 1}, 64, 64}}};
    double clock = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.1, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy policy) {
            clock += 0.001;
            return [&, policy](uint32_t, uint64_t) {
                clock += policy.lane_capacity == 1 ? 0.06 : 0.03;
            };
        });
    REQUIRE(report.sufficient_measurements);
    REQUIRE(report.batch_size == 64);
    REQUIRE(report.trial_shots == 65);
    for (const auto& trial : report.trials) {
        REQUIRE(trial.used_warmup());
        REQUIRE(trial.shots == 0);
        REQUIRE(trial.setup_seconds == Catch::Approx(0.001));
        REQUIRE(trial.warmup_seconds > trial.setup_seconds);
    }
}

TEST_CASE("Batch tuning prefers steady state timings when they are available") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{1, 1}, 1, 1}, {{64, 1}, 64, 64}}};
    double clock = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.3, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy policy) {
            return [&, policy, first = true](uint32_t shots, uint64_t) mutable {
                clock += first ? 0.02 : shots / (policy.lane_capacity == 1 ? 100.0 : 10000.0);
                first = false;
            };
        });
    REQUIRE(report.batch_size == 64);
    REQUIRE(report.sufficient_measurements);
    REQUIRE_FALSE(report.trials.back().used_warmup());
    REQUIRE(report.trials.back().shots_per_second() == Catch::Approx(10000));
}

TEST_CASE("Batch tuning reports a final probe overrun even with a usable comparison") {
    const std::array<BatchTuningCandidate, 2> candidates{{{{1, 1}, 1, 1}, {{64, 1}, 64, 64}}};
    double clock = 0;
    const auto report = sweep_batch_candidates(
        candidates, candidates.front().policy, 1, 0.2, clifft::seed_root_from_seed(42),
        [&] { return clock; },
        [&](BatchExecutionPolicy policy) {
            return [&, policy](uint32_t, uint64_t) {
                clock += policy.lane_capacity == 1 ? 0.03 : 0.2;
            };
        });
    REQUIRE(report.trials.size() == candidates.size());
    REQUIRE(report.sufficient_measurements);
    REQUIRE(report.batch_size == 64);
    REQUIRE(report.elapsed_seconds > 0.2);
    REQUIRE(report.stop_reason == "budget_exhausted");
}

TEST_CASE("Batch tuning finds packed speedups and skips probes unlikely to fit the budget") {
    const std::array<BatchTuningCandidate, 5> candidates{{{{1, 1}, 1, 8192},
                                                          {{64, 1}, 64, 8192},
                                                          {{256, 1}, 256, 8192},
                                                          {{1024, 1}, 1024, 8192},
                                                          {{2048, 1}, 2048, 8192}}};
    for (const double budget : {0.25, 1.0, 2.0}) {
        CAPTURE(budget);
        double clock = 0;
        std::vector<uint32_t> prepared;
        const auto report = sweep_batch_candidates(
            candidates, candidates.front().policy, 1, budget, clifft::seed_root_from_seed(42),
            [&] { return clock; },
            [&](BatchExecutionPolicy policy) {
                prepared.push_back(policy.lane_capacity);
                return [&, policy](uint32_t shots, uint64_t) {
                    const double per_shot = policy.lane_capacity == 1     ? 0.0162
                                            : policy.lane_capacity == 64  ? 0.001
                                            : policy.lane_capacity == 256 ? 0.00062
                                                                          : 0.0006;
                    clock += shots * per_shot;
                };
            });
        REQUIRE(report.sufficient_measurements);
        REQUIRE(report.batch_size > 1);
        REQUIRE(report.elapsed_seconds <= budget);
        REQUIRE(report.stop_reason == "budget_exhausted");
        // The first packed candidate must be tried even though extrapolating
        // from scalar timing would reject it under the default budget.
        REQUIRE(prepared.at(1) == 64);
        REQUIRE(std::ranges::find(prepared, 2048) == prepared.end());
    }
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
    REQUIRE_FALSE(report.sufficient_measurements);
    REQUIRE(report.stop_reason == "budget_exhausted");
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
    for (double budget : {-1.0, clifft::test::opaque_infinity(), clifft::test::opaque_nan()}) {
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

TEST_CASE("Batch tuning skips all probes when wide packed candidates exceed memory limits") {
    const auto plan = compile_tuning_plan(
        "H 0 1 2 3 4 5 6 7 8 9 10 11 12 13\n"
        "T 0 1 2 3 4 5 6 7 8 9 10 11 12 13\nM 0");
    REQUIRE(plan.peak_active_width() == 14);
    const auto result = sample(plan, 65, 42, 1, std::nullopt, std::nullopt, 0.1);
    REQUIRE(result.batch_tuning->stop_reason == "single_candidate");
    REQUIRE(result.batch_tuning->trials.empty());
    REQUIRE(result.batch_tuning->trial_shots == 0);
    REQUIRE(result.batch_tuning->batch_size == 1);
    REQUIRE(result.measurements == sample(plan, 65, 42).measurements);
}
