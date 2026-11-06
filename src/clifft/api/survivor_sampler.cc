#include "clifft/api/survivor_sampler.h"

#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/sharing_executor.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
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
           const sampling::EarlyPostselection& checks, uint32_t capacity, uint32_t initial_states,
           bool sharing, const sampling::SharingKey* key)
        : executor(make_executor(plan, noise, checks, capacity, initial_states, sharing, key)),
          observable_ones(plan.num_observables()) {}

    static Executor make_executor(const sampling::ExecutablePlan& plan,
                                  const sampling::PreparedBatchNoise& noise,
                                  const sampling::EarlyPostselection& checks, uint32_t capacity,
                                  uint32_t initial_states, bool sharing,
                                  const sampling::SharingKey* key) {
        // Width-zero plans have no coefficient evolution to share. Their packed
        // executor applies classical checks without per-shot group bookkeeping.
        if (plan.peak_active_width() == 0) {
            return Executor(std::in_place_type<sampling::BatchExecutor>, plan, capacity,
                            sampling::BatchOutputMode::AggregateSurvivors,
                            sampling::BatchSamplingMode::Ordinary, &noise);
        }
        return Executor(std::in_place_type<sampling::SharingExecutor>, plan, noise, checks,
                        capacity, initial_states, sharing, key);
    }
};

SurvivorSampler::SurvivorSampler(const Program& program, uint32_t batch_size, uint32_t threads,
                                 uint32_t initial_states, sampling::BatchNoiseMode noise_mode,
                                 bool sharing, bool early_postselection, bool compact_outputs,
                                 bool group_by_noise)
    : program_(&program),
      batch_size_(batch_size),
      compact_plan_(compact_outputs ? std::optional(program.sampling_plan(false).compact_outputs(
                                          early_postselection))
                                    : std::nullopt),
      plan_(compact_plan_ ? *compact_plan_ : program.sampling_plan(false)),
      noise_(plan_, noise_mode),
      checks_(plan_, early_postselection && plan_.peak_active_width() != 0),
      key_(sharing && group_by_noise && plan_.peak_active_width() != 0
               ? std::optional<sampling::SharingKey>(plan_)
               : std::nullopt) {
    if (batch_size == 0 || initial_states == 0) {
        throw std::invalid_argument("batch_size and initial_states must be positive");
    }
    const uint32_t worker_count = resolve_thread_budget(threads);
    workers_.reserve(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        workers_.push_back(std::make_unique<Worker>(
            plan_, noise_, checks_, batch_size, initial_states, sharing, key_ ? &*key_ : nullptr));
    }
}

SurvivorSampler::~SurvivorSampler() = default;

sampling::SamplingSurvivorResult SurvivorSampler::sample(uint32_t shots,
                                                         std::optional<uint64_t> seed) {
    std::lock_guard lock(mutex_);
    sampling::SamplingSurvivorResult result;
    result.total_shots = shots;
    result.observable_ones.resize(program_->original().num_observables());
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
