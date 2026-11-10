#include "clifft/api/survivor_sampler.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/sharing_executor.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <variant>

namespace clifft {

struct SurvivorSampler::Worker {
    using Executor = std::variant<sampling::BatchExecutor, sampling::SharingExecutor>;
    Executor executor;
    uint32_t passed = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;

    Worker(const sampling::ExecutablePlan& plan, const sampling::PreparedBatchNoise& noise,
           const sampling::EarlyPostselection* checks, uint32_t capacity, uint32_t initial_states,
           bool sharing, const sampling::SharingKey* key, uint64_t max_worker_state_bytes)
        : executor(make_executor(plan, noise, checks, capacity, initial_states, sharing, key,
                                 max_worker_state_bytes)),
          observable_ones(plan.num_observables()) {}

    static Executor make_executor(const sampling::ExecutablePlan& plan,
                                  const sampling::PreparedBatchNoise& noise,
                                  const sampling::EarlyPostselection* checks, uint32_t capacity,
                                  uint32_t initial_states, bool sharing,
                                  const sampling::SharingKey* key,
                                  uint64_t max_worker_state_bytes) {
        // Width-zero plans have no coefficient evolution to share. Their packed
        // executor applies classical checks without per-shot group bookkeeping.
        if (plan.peak_active_width() == 0) {
            return Executor(std::in_place_type<sampling::BatchExecutor>, plan, capacity,
                            sampling::BatchOutputMode::AggregateSurvivors,
                            sampling::BatchSamplingMode::Ordinary, &noise);
        }
        return Executor(std::in_place_type<sampling::SharingExecutor>, plan, noise, *checks,
                        capacity, initial_states, sharing, key, max_worker_state_bytes);
    }
};

SurvivorSampler::SurvivorSampler(const Program& program, std::optional<uint32_t> batch_size,
                                 uint32_t threads, uint32_t initial_states,
                                 sampling::BatchNoiseMode noise_mode, bool sharing,
                                 bool early_postselection, bool group_by_noise,
                                 uint64_t max_worker_state_bytes,
                                 std::optional<double> tuning_budget_seconds)
    : program_(program),
      call_sized_(!batch_size),
      worker_count_(resolve_thread_budget(threads)),
      initial_states_(initial_states),
      worker_state_bytes_(max_worker_state_bytes),
      sharing_(sharing),
      tuning_budget_seconds_(tuning_budget_seconds) {
    sampling::batch_detail::validate_batch_tuning_options(batch_size, tuning_budget_seconds);
    if (batch_size == 0 || initial_states == 0 || max_worker_state_bytes == 0) {
        throw std::invalid_argument(
            "batch_size, initial_states, and max_worker_state_bytes must be positive");
    }
    // The count-only plan keeps the compiled plan's peak width, so the state
    // budget is checked before the Program builds any preparation.
    const uint32_t width = program_.sampling_plan(false).peak_active_width();
    if (width != 0 && worker_state_bytes_ < sampling::State::storage_bytes(width)) {
        throw std::invalid_argument("max_worker_state_bytes cannot hold one shared state");
    }
    const auto& preparation = program_.sampling_preparation(false, early_postselection);
    plan_ = &preparation.plan();
    assert(plan_->peak_active_width() == width &&
           "count-only preparation must keep the compiled peak active width");
    noise_ = &preparation.noise(noise_mode);
    if (width != 0) {
        checks_ = &preparation.checks();
        if (sharing && group_by_noise) {
            key_ = &preparation.sharing_key();
        }
    }
    // Automatic capacities fit even when every lane needs a distinct state.
    // Explicit capacities can exploit sharing beyond this conservative bound.
    if (batch_size) {
        prepare_workers({*batch_size, worker_count_});
    } else {
        safe_capacity_ =
            sampling::max_safe_shared_capacity(*plan_, *noise_, key_, worker_state_bytes_);
    }
}

void SurvivorSampler::prepare_workers(sampling::BatchExecutionPolicy policy) {
    if (batch_size_ != policy.lane_capacity) {
        workers_.clear();
        batch_size_ = policy.lane_capacity;
    }
    workers_.reserve(policy.worker_count);
    while (workers_.size() < policy.worker_count) {
        workers_.push_back(std::make_unique<Worker>(*plan_, *noise_, checks_, batch_size_,
                                                    initial_states_, sharing_, key_,
                                                    worker_state_bytes_));
    }
}

void SurvivorSampler::tune(uint32_t shots, std::optional<uint64_t> seed) {
    using namespace sampling;
    using namespace sampling::batch_detail;
    const auto candidates = shared_tuning_candidates(safe_capacity_, shots, worker_count_);
    const BatchExecutionPolicy baseline = candidates.front().policy;
    const auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    auto report =
        sweep_batch_candidates(candidates, baseline, 1, *tuning_budget_seconds_,
                               make_seed_root(shots, seed), now, [&](BatchExecutionPolicy policy) {
                                   prepare_workers(policy);
                                   return [this](uint32_t count, uint64_t trial_seed) {
                                       (void)sample_batches(count, trial_seed);
                                   };
                               });
    prepare_workers({report.batch_size, report.shot_workers});
    tuning_report_ = std::move(report);
}

SurvivorSampler::~SurvivorSampler() = default;

sampling::SamplingSurvivorResult SurvivorSampler::sample(uint32_t shots,
                                                         std::optional<uint64_t> seed) {
    std::lock_guard lock(mutex_);
    if (shots != 0 && call_sized_) {
        if (tuning_report_) {
            // The tuned capacity is retained; later calls only add the workers they can fill.
            const uint64_t batches = (uint64_t{shots} + batch_size_ - 1) / batch_size_;
            prepare_workers(
                {batch_size_, static_cast<uint32_t>(std::min<uint64_t>(worker_count_, batches))});
        } else if (tuning_budget_seconds_) {
            tune(shots, seed);
        } else {
            prepare_workers(
                sampling::resolve_shared_execution_policy(safe_capacity_, shots, worker_count_));
        }
    }
    auto result = sample_batches(shots, seed);
    result.batch_tuning = tuning_report_;
    return result;
}

sampling::SamplingSurvivorResult SurvivorSampler::sample_batches(uint32_t shots,
                                                                 std::optional<uint64_t> seed) {
    sampling::SamplingSurvivorResult result;
    result.total_shots = shots;
    result.observable_ones.resize(program_.original().num_observables());
    if (shots == 0) {
        return result;
    }
    const auto root = make_seed_root(shots, seed);
    const uint32_t batches =
        static_cast<uint32_t>((uint64_t{shots} + batch_size_ - 1) / batch_size_);
    const uint32_t worker_count = std::min(batches, static_cast<uint32_t>(workers_.size()));
    const auto used = run_shot_ranges(
        batches, worker_count,
        [&](uint32_t worker) {
            auto* context = workers_[worker].get();
            context->passed = context->logical_errors = 0;
            std::ranges::fill(context->observable_ones, 0);
            return context;
        },
        [&](Worker* worker, ShotRange range) {
            std::visit(
                [&](auto& executor) {
                    for (uint32_t batch = range.begin; batch < range.end; ++batch) {
                        const uint32_t first = static_cast<uint32_t>(uint64_t{batch} * batch_size_);
                        executor.run_batch(root, first, std::min(batch_size_, shots - first));
                        worker->passed += executor.surviving_shots();
                        worker->logical_errors +=
                            executor.accumulate_survivor_counts(worker->observable_ones);
                    }
                },
                worker->executor);
        });
    for (const auto* worker : used) {
        result.passed_shots += worker->passed;
        result.logical_errors += worker->logical_errors;
        for (size_t observable = 0; observable < result.observable_ones.size(); ++observable) {
            result.observable_ones[observable] += worker->observable_ones[observable];
        }
    }
    return result;
}

}  // namespace clifft
