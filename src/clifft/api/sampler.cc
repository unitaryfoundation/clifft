// Native execution selection, reusable shared workers, and ordered row collection.

#include "clifft/api/sampler.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/output.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace clifft {
namespace {

using namespace sampling;

SharedWorkerKey shared_worker_key(const ExecutablePlan& plan, const SamplingExecution& execution,
                                  SamplingOutputs outputs) noexcept {
    return SharedWorkerKey{
        .output_mode = outputs.rows() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors,
        .noise_mode = *execution.noise_preparation,
        .lane_capacity = execution.batch_size,
        .max_worker_state_bytes =
            plan.peak_active_width() == 0 ? 0 : *execution.max_worker_state_bytes,
        .group_by_noise = plan.peak_active_width() == 0 ? false : execution.group_by_noise,
    };
}

template <typename Result>
Result run_shared(const SamplingPreparation& preparation, uint32_t shots,
                  std::optional<uint64_t> seed, SamplingOutputs outputs,
                  const SamplingExecution& execution,
                  std::vector<std::unique_ptr<SharedWorker>>& workers) {
    constexpr bool survivors = std::is_same_v<Result, SamplingSurvivorResult>;
    const auto& plan = preparation.plan();
    Result result;
    result.execution = execution;
    if constexpr (survivors) {
        result.total_shots = shots;
        result.observable_ones.resize(plan.num_observables());
    } else {
        batch_detail::resize_output_rows(result, shots, plan, outputs);
    }
    if (shots == 0) {
        return result;
    }
    const auto& noise = preparation.noise(*execution.noise_preparation);
    const auto* checks = plan.peak_active_width() == 0 ? nullptr : &preparation.checks();
    const auto* key = execution.group_by_noise ? &preparation.sharing_key() : nullptr;
    const auto worker_key = shared_worker_key(plan, execution, outputs);
    const uint32_t capacity = execution.batch_size;
    const uint32_t batch_count = static_cast<uint32_t>((uint64_t{shots} + capacity - 1) / capacity);
    batch_detail::SurvivorRowCollector rows(shots, capacity, survivors && outputs.rows());
    const auto root = make_seed_root(shots, seed);
    std::vector<std::unique_ptr<SharedWorker>> active(execution.shot_workers);
    std::atomic<size_t> next_warm{0};
    (void)run_shot_ranges(
        batch_count, execution.shot_workers,
        [&](uint32_t index) {
            auto& worker = active[index];
            if (!worker) {
                const size_t warm = next_warm.fetch_add(1, std::memory_order_relaxed);
                if (warm < workers.size()) {
                    worker = std::move(workers[warm]);
                    worker->passed = worker->logical_errors = 0;
                    std::fill(worker->observable_ones.begin(), worker->observable_ones.end(), 0);
                } else {
                    worker = std::make_unique<SharedWorker>(plan, noise, checks, key, worker_key);
                }
            }
            return worker.get();
        },
        [&](auto& worker, ShotRange range) {
            std::visit(
                [&](auto& executor) {
                    for (uint32_t batch = range.begin; batch < range.end; ++batch) {
                        const uint32_t first = static_cast<uint32_t>(uint64_t{batch} * capacity);
                        executor.run_batch(root, first, std::min(capacity, shots - first));
                        if constexpr (survivors) {
                            worker->passed += executor.surviving_shots();
                            worker->logical_errors +=
                                executor.accumulate_survivor_counts(worker->observable_ones);
                            if (outputs.rows()) {
                                auto& batch_rows = rows.batch(first);
                                batch_detail::resize_output_rows(
                                    batch_rows, executor.surviving_shots(), plan, outputs);
                                batch_detail::copy_batch_rows(batch_rows, executor, plan, outputs,
                                                              executor.surviving_shots());
                            }
                        } else {
                            batch_detail::copy_batch_rows(result, executor, plan, outputs);
                        }
                    }
                },
                worker->executor);
        },
        1, WorkerInitialization::OnWorkerThread);
    if constexpr (survivors) {
        for (const auto& worker : active) {
            // An OpenMP runtime can supply fewer threads than requested.
            if (!worker) {
                continue;
            }
            result.passed_shots += worker->passed;
            result.logical_errors += worker->logical_errors;
            for (size_t i = 0; i < result.observable_ones.size(); ++i) {
                result.observable_ones[i] += worker->observable_ones[i];
            }
        }
        if (outputs.rows()) {
            rows.finish(result);
        }
    }
    // Preserve both active workers and any unconsumed warm workers for reuse.
    std::vector<std::unique_ptr<SharedWorker>> kept;
    kept.reserve(std::max(workers.size(), active.size()));
    for (auto& worker : active) {
        if (worker) {
            kept.push_back(std::move(worker));
        }
    }
    for (auto& worker : workers) {
        if (worker) {
            kept.push_back(std::move(worker));
        }
    }
    workers = std::move(kept);
    return result;
}

template <typename Result>
Result sample_program(const Program& program, uint32_t shots, std::optional<uint64_t> seed,
                      SamplingOutputs outputs, const SamplingOptions& options) {
    constexpr bool survivors = std::is_same_v<Result, SamplingSurvivorResult>;
    const auto& compiled = program.sampling_plan(outputs.measurements());
    if constexpr (!survivors) {
        if (!outputs.rows() || program.has_postselection()) {
            throw std::invalid_argument("fixed-row sampling requires outputs and no postselection");
        }
    }
    if (compiled.has_instruments() || compiled.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "sampling requires a fixed plan with bound noise distributions");
    }
    batch_detail::validate_batch_tuning_options(options.batch_size, options.tuning_budget_seconds);
    if (options.batch_size == 0 || options.max_worker_state_bytes == 0) {
        throw std::invalid_argument("batch_size and max_worker_state_bytes must be positive");
    }
    const ThreadLayout layout =
        options.sharing == true && !options.thread_layout
            ? ThreadLayout{.shot_workers = std::min(shots, resolve_thread_budget(options.threads)),
                           .intra_shot_workers = 1}
            : resolve_thread_layout(compiled, shots, options.threads, options.thread_layout);
    if (options.noise_preparation.has_value()) {
        if (options.batch_size == 1) {
            throw std::invalid_argument(
                "noise_preparation is incompatible with scalar batch_size=1");
        }
        if (options.thread_layout.has_value() && layout.intra_shot_workers > 1) {
            throw std::invalid_argument(
                "noise_preparation is incompatible with intra-shot workers");
        }
    }
    if (options.group_by_noise == true && options.sharing != true) {
        throw std::invalid_argument("group_by_noise=True requires sharing=True");
    }
    if (options.sharing == true) {
        if (!kPackedBatchExecutionAvailable) {
            throw std::invalid_argument(
                "sharing=True is unavailable in builds without packed batch execution");
        }
        if (outputs.expectations() && compiled.num_exp_vals() != 0) {
            throw std::invalid_argument(
                "sharing=True is incompatible with retained EXP_VAL outputs");
        }
        if (options.batch_size == 1) {
            throw std::invalid_argument("sharing=True is incompatible with scalar batch_size=1");
        }
        if (layout.intra_shot_workers > 1) {
            throw std::invalid_argument("sharing=True is incompatible with intra-shot workers");
        }
    }
    const bool supported = kPackedBatchExecutionAvailable &&
                           !(outputs.expectations() && compiled.num_exp_vals() != 0) &&
                           options.batch_size != 1 && layout.intra_shot_workers == 1;

    const uint32_t workers = std::max(uint32_t{1}, layout.shot_workers);
    const uint32_t width = compiled.peak_active_width();
    const auto mode = outputs.rows() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    const auto tuning_start = options.tuning_budget_seconds
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    bool shared = options.sharing.value_or(false);
    if (!options.sharing && !options.batch_size && supported && width != 0) {
        shared =
            should_share_sampling(compiled, shots, outputs, layout, options.max_worker_state_bytes);
    }
    const bool compare_executors = !options.sharing && supported && width != 0 && shots > 1 &&
                                   options.tuning_budget_seconds.value_or(0) > 0;
    const bool fits_states = !(shared || compare_executors) || width == 0 ||
                             options.max_worker_state_bytes / State::storage_bytes(width) >= 2;
    if (!fits_states) {
        if (options.sharing == true) {
            throw std::invalid_argument(
                "max_worker_state_bytes must hold at least two shared states");
        }
        shared = false;
    }
    const auto shared_noise_mode = options.noise_preparation.value_or(BatchNoiseMode::SiteSparse);
    const SamplingPreparation* shared_preparation = nullptr;
    uint32_t safe = 0;
    if ((shared || compare_executors) && fits_states) {
        shared_preparation = &program.sampling_preparation(
            outputs,
            {.early_postselection = compiled.has_postselection(), .survivor_counts = survivors});
        assert(shared_preparation->plan().peak_active_width() == width &&
               "output preparation must preserve compiled peak active width");
        const auto& noise = shared_preparation->noise(shared_noise_mode);
        const auto* key = width != 0 && options.group_by_noise.value_or(false)
                              ? &shared_preparation->sharing_key()
                              : nullptr;
        safe = max_safe_shared_capacity(shared_preparation->plan(), noise, key,
                                        options.max_worker_state_bytes, mode);
        if (safe < 2) {
            if (options.sharing == true) {
                throw std::invalid_argument(
                    "shared execution requires capacity for at least two lanes");
            }
            shared = false;
        }
    }
    auto run_ordinary = [&](std::optional<uint32_t> capacity, std::optional<double> budget,
                            std::optional<ThreadLayout> run_layout) {
        const auto& preparation = program.sampling_preparation(
            outputs, {.early_postselection = false, .survivor_counts = survivors});
        const PreparedBatchNoise* noise = nullptr;
        const bool tuning_packed =
            budget.value_or(0) > 0 && shots > 1 && layout.intra_shot_workers == 1;
        if (shots != 0 && capacity != 1 &&
            (capacity || tuning_packed ||
             automatic_batch_eligible(preparation.plan(), shots, layout.intra_shot_workers,
                                      mode))) {
            const auto selected =
                select_batch_noise_mode(preparation.plan(), shots, options.batch_size);
            noise = &preparation.noise(options.noise_preparation.value_or(selected));
        }
        if constexpr (survivors) {
            return sampling::sample_survivors(preparation.plan(), shots, seed, outputs,
                                              options.threads, run_layout, capacity, budget, noise);
        } else {
            return sampling::sample(preparation.plan(), shots, seed, options.threads, run_layout,
                                    capacity, budget, outputs, noise);
        }
    };
    if (safe < 2) {
        return run_ordinary(options.batch_size, options.tuning_budget_seconds,
                            options.thread_layout);
    }

    const auto& preparation = *shared_preparation;
    SamplingExecution execution;
    execution.executor = width == 0 ? SamplingExecutor::Packed : SamplingExecutor::Shared;
    execution.sharing = true;
    execution.outputs = outputs;
    if (width != 0) {
        execution.max_worker_state_bytes = options.max_worker_state_bytes;
        execution.group_by_noise = options.group_by_noise.value_or(false);
    }
    execution.noise_preparation = shared_noise_mode;
    auto policy = resolve_shared_execution_policy(safe, shots, workers);
    if (options.batch_size) {
        if (shots == 0) {
            policy.lane_capacity = *options.batch_size;
        } else {
            policy.lane_capacity = std::min(*options.batch_size, std::max(shots, uint32_t{2}));
            policy.worker_count = static_cast<uint32_t>(std::min<uint64_t>(
                workers, (uint64_t{shots} + policy.lane_capacity - 1) / policy.lane_capacity));
        }
    }
    policy.sharing = true;
    execution.batch_size = policy.lane_capacity;
    execution.shot_workers = policy.worker_count;
    std::optional<BatchTuningReport> tuning;
    if (options.tuning_budget_seconds) {
        tuning.emplace();
        tuning->sharing = true;
        tuning->batch_size = tuning->baseline_batch_size = policy.lane_capacity;
        tuning->shot_workers = execution.shot_workers;
        if (shots == 0) {
            tuning->stop_reason = "zero_shots";
        } else if (*options.tuning_budget_seconds == 0) {
            tuning->stop_reason = "zero_budget";
        } else {
            const auto now = [&] {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                     tuning_start)
                    .count();
            };
            const uint64_t row_bytes =
                (outputs.measurements() ? outputs.stride(preparation.plan().num_visible_records())
                                        : 0) +
                (outputs.syndromes()
                     ? uint64_t{outputs.stride(preparation.plan().num_detectors())} +
                           outputs.stride(preparation.plan().num_observables())
                     : 0);
            auto candidates = shared_tuning_candidates(safe, shots, workers, row_bytes);
            if (compare_executors) {
                const auto& ordinary = program.sampling_preparation(
                    outputs, {.early_postselection = false, .survivor_counts = survivors});
                const PreparedBatchNoise* noise = nullptr;
                const auto ordinary_mode = options.noise_preparation.value_or(
                    select_batch_noise_mode(ordinary.plan(), shots, options.batch_size));
                if (options.noise_preparation == BatchNoiseMode::SiteRows ||
                    automatic_batch_eligible(ordinary.plan(), shots, 1, mode)) {
                    noise = &ordinary.noise(ordinary_mode);
                }
                auto ordinary_plan = batch_detail::ordinary_tuning_plan(
                    ordinary.plan(), shots, workers, outputs, survivors, noise);
                // Measure the automatic baseline first, followed by the alternate
                // executor's baseline, so tight budgets still compare executors.
                if (!shared) {
                    candidates.swap(ordinary_plan.candidates);
                    policy = ordinary_plan.baseline;
                }
                if (!candidates.empty() && !ordinary_plan.candidates.empty()) {
                    candidates.insert(candidates.begin() + 1, ordinary_plan.candidates.front());
                    candidates.insert(candidates.end(), ordinary_plan.candidates.begin() + 1,
                                      ordinary_plan.candidates.end());
                }
                tuning->sharing = policy.sharing;
                tuning->batch_size = tuning->baseline_batch_size = policy.lane_capacity;
                tuning->shot_workers = policy.worker_count;
            }
            if (candidates.empty()) {
                tuning->stop_reason = "output_budget";
            } else if (candidates.size() == 1) {
                tuning->stop_reason = "single_candidate";
            } else {
                tuning = batch_detail::sweep_batch_candidates(
                    candidates, policy, 1, std::max(0.0, *options.tuning_budget_seconds - now()),
                    make_seed_root(shots, seed), now,
                    [&](BatchExecutionPolicy candidate) -> batch_detail::BatchTuningProbe {
                        if (!candidate.sharing) {
                            const auto& ordinary = program.sampling_preparation(
                                outputs,
                                {.early_postselection = false, .survivor_counts = survivors});
                            const auto* noise =
                                candidate.lane_capacity > 1
                                    ? &ordinary.noise(options.noise_preparation.value_or(
                                          select_batch_noise_mode(ordinary.plan(), shots,
                                                                  options.batch_size)))
                                    : nullptr;
                            return batch_detail::make_ordinary_tuning_probe(
                                ordinary.plan(), candidate, outputs, survivors, noise);
                        }
                        auto trial = execution;
                        trial.batch_size = candidate.lane_capacity;
                        trial.shot_workers = candidate.worker_count;
                        auto trial_workers =
                            std::make_shared<std::vector<std::unique_ptr<SharedWorker>>>();
                        return [&, trial, trial_workers = std::move(trial_workers)](
                                   uint32_t count, uint64_t trial_seed) {
                            (void)run_shared<Result>(preparation, count, trial_seed, outputs, trial,
                                                     *trial_workers);
                        };
                    },
                    shots);
            }
            tuning->elapsed_seconds = now();
            if (!tuning->sharing) {
                auto result = run_ordinary(tuning->batch_size, std::nullopt,
                                           ThreadLayout{tuning->shot_workers, 1});
                result.batch_tuning = std::move(tuning);
                return result;
            }
            execution.batch_size = tuning->batch_size;
            execution.shot_workers = tuning->shot_workers;
        }
    }
    const auto worker_key = shared_worker_key(preparation.plan(), execution, outputs);
    auto call_workers = preparation.borrow_shared_workers(worker_key, execution.shot_workers);
    auto result = run_shared<Result>(preparation, shots, seed, outputs, execution, call_workers);
    if (execution.shot_workers != 0) {
        preparation.return_shared_workers(worker_key, std::move(call_workers));
    }
    result.batch_tuning = std::move(tuning);
    return result;
}

}  // namespace

bool should_share_sampling(const sampling::ExecutablePlan& compiled, uint32_t shots,
                           sampling::SamplingOutputs outputs, sampling::ThreadLayout layout,
                           uint64_t max_worker_state_bytes) noexcept {
    using namespace sampling;
    const uint32_t width = compiled.peak_active_width();
    if (width == 0 || layout.intra_shot_workers != 1 ||
        (outputs.expectations() && compiled.num_exp_vals() != 0)) {
        return false;
    }
    constexpr uint32_t kMinSharedStates = 2;
    const uint64_t state_capacity = max_worker_state_bytes / State::storage_bytes(width);
    if (state_capacity < kMinSharedStates) {
        return false;
    }
    if (compiled.has_postselection()) {
        return shots >= kDefaultMinAutoBatchShots;
    }
    constexpr uint32_t kMinUnitaryPrefixShots = 16;
    if (compiled.num_presampled_symbols() == 0 && !compiled.has_readout_noise() && width >= 8 &&
        shots >= kMinUnitaryPrefixShots && compiled.has_reusable_unitary_prefix()) {
        return true;
    }
    // Call-level estimate of the lanes shots share. It ignores the worker
    // count, although each shot worker runs its own smaller batch.
    const auto lanes = static_cast<uint32_t>(
        std::min<uint64_t>({shots, kMaxAutoSharedBatchShots, state_capacity}));
    return shots >= kDefaultMinAutoBatchShots && compiled.has_bounded_trajectory_divergence(lanes);
}

sampling::SamplingResult sample(const Program& program, uint32_t shots,
                                std::optional<uint64_t> seed, sampling::SamplingOutputs outputs,
                                const SamplingOptions& options) {
    return sample_program<sampling::SamplingResult>(program, shots, seed, outputs, options);
}

sampling::SamplingSurvivorResult sample_survivors(const Program& program, uint32_t shots,
                                                  std::optional<uint64_t> seed,
                                                  sampling::SamplingOutputs outputs,
                                                  const SamplingOptions& options) {
    return sample_program<sampling::SamplingSurvivorResult>(program, shots, seed, outputs, options);
}

}  // namespace clifft
