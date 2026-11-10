#include "clifft/api/survivor_sampler.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/sharing_executor.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <variant>

namespace clifft {

namespace {
constexpr uint32_t kMaxAutoSharingBatchShots = 8192;
}

struct SurvivorSampler::Worker {
    using Executor = std::variant<sampling::BatchExecutor, sampling::SharingExecutor>;
    Executor executor;
    uint32_t passed = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;

    Worker(const sampling::ExecutablePlan& plan, const sampling::PreparedBatchNoise& noise,
           const sampling::EarlyPostselection* checks, uint32_t capacity, uint32_t initial_states,
           bool sharing, const sampling::SharingKey* key, uint64_t max_state_bytes)
        : executor(make_executor(plan, noise, checks, capacity, initial_states, sharing, key,
                                 max_state_bytes)),
          observable_ones(plan.num_observables()) {}

    static Executor make_executor(const sampling::ExecutablePlan& plan,
                                  const sampling::PreparedBatchNoise& noise,
                                  const sampling::EarlyPostselection* checks, uint32_t capacity,
                                  uint32_t initial_states, bool sharing,
                                  const sampling::SharingKey* key, uint64_t max_state_bytes) {
        // Width-zero plans have no coefficient evolution to share. Their packed
        // executor applies classical checks without per-shot group bookkeeping.
        if (plan.peak_active_width() == 0) {
            return Executor(std::in_place_type<sampling::BatchExecutor>, plan, capacity,
                            sampling::BatchOutputMode::AggregateSurvivors,
                            sampling::BatchSamplingMode::Ordinary, &noise);
        }
        return Executor(std::in_place_type<sampling::SharingExecutor>, plan, noise, *checks,
                        capacity, initial_states, sharing, key, max_state_bytes);
    }
};

SurvivorSampler::SurvivorSampler(const Program& program, std::optional<uint32_t> batch_size,
                                 uint32_t threads, uint32_t initial_states,
                                 sampling::BatchNoiseMode noise_mode, bool sharing,
                                 bool early_postselection, bool group_by_noise,
                                 uint64_t max_state_bytes,
                                 std::optional<double> tuning_budget_seconds)
    : program_(program),
      batch_size_(batch_size.value_or(kMaxAutoSharingBatchShots)),
      worker_count_(resolve_thread_budget(threads)),
      initial_states_(initial_states),
      worker_state_bytes_(max_state_bytes / worker_count_),
      sharing_(sharing),
      tuning_budget_seconds_(tuning_budget_seconds),
      preparation_(program_.sampling_preparation(false, early_postselection)),
      plan_(preparation_.plan()),
      noise_(preparation_.noise(noise_mode)),
      checks_(plan_.peak_active_width() != 0 ? &preparation_.checks() : nullptr),
      key_(sharing && group_by_noise && plan_.peak_active_width() != 0 ? &preparation_.sharing_key()
                                                                       : nullptr) {
    sampling::batch_detail::validate_batch_tuning_options(batch_size, tuning_budget_seconds);
    if (batch_size == 0 || initial_states == 0 || max_state_bytes == 0) {
        throw std::invalid_argument(
            "batch_size, initial_states, and max_state_bytes must be positive");
    }
    if (plan_.peak_active_width() != 0) {
        const uint64_t states =
            worker_state_bytes_ / sampling::State::storage_bytes(plan_.peak_active_width());
        if (states == 0) {
            throw std::invalid_argument("max_state_bytes cannot hold one shared state per worker");
        }
        // Automatic capacities fit even when every lane needs a distinct state.
        // Explicit capacities can exploit sharing beyond this conservative bound.
        if (!batch_size) {
            batch_size_ = static_cast<uint32_t>(std::min<uint64_t>(batch_size_, states));
        }
    }
    prepare_workers(batch_size_);
}

void SurvivorSampler::prepare_workers(uint32_t capacity) {
    workers_.clear();
    batch_size_ = capacity;
    workers_.reserve(worker_count_);
    for (uint32_t worker = 0; worker < worker_count_; ++worker) {
        workers_.push_back(std::make_unique<Worker>(plan_, noise_, checks_, capacity,
                                                    initial_states_, sharing_, key_,
                                                    worker_state_bytes_));
    }
}

void SurvivorSampler::tune(uint32_t shots, std::optional<uint64_t> seed) {
    using namespace sampling;
    using namespace sampling::batch_detail;
    const BatchExecutionPolicy baseline{batch_size_, worker_count_};
    std::vector<BatchTuningCandidate> candidates;
    const auto append = [&](uint32_t capacity) {
        if (capacity > baseline.lane_capacity ||
            std::ranges::any_of(candidates, [&](const auto& candidate) {
                return candidate.policy.lane_capacity == capacity;
            })) {
            return;
        }
        const uint32_t probe_shots = static_cast<uint32_t>(
            std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                               std::max<uint64_t>(256, uint64_t{capacity} * worker_count_)));
        candidates.push_back(
            {{capacity, worker_count_}, probe_shots, std::max(uint32_t{65536}, probe_shots)});
    };
    append(baseline.lane_capacity);
    for (uint32_t capacity : {uint32_t{1}, uint32_t{64}, uint32_t{256}, uint32_t{1024}}) {
        append(capacity);
    }
    const auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    auto report =
        sweep_batch_candidates(candidates, baseline, 1, *tuning_budget_seconds_,
                               make_seed_root(shots, seed), now, [&](BatchExecutionPolicy policy) {
                                   prepare_workers(policy.lane_capacity);
                                   return [this](uint32_t count, uint64_t trial_seed) {
                                       (void)sample_batches(count, trial_seed);
                                   };
                               });
    if (batch_size_ != report.batch_size) {
        prepare_workers(report.batch_size);
    }
    tuning_report_ = std::move(report);
}

SurvivorSampler::~SurvivorSampler() = default;

sampling::SamplingSurvivorResult SurvivorSampler::sample(uint32_t shots,
                                                         std::optional<uint64_t> seed) {
    std::lock_guard lock(mutex_);
    if (shots != 0 && tuning_budget_seconds_ && !tuning_report_) {
        tune(shots, seed);
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
