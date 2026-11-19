// Native execution selection, per-call shared workers, and ordered row collection.

#include "clifft/api/sampler.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/output.h"
#include "clifft/util/numeric.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace clifft {
namespace {

using namespace sampling;

constexpr uint32_t kMaxInitialSharedStates = 128;
// Bound upfront state preallocation per worker on wide plans; additional states
// grow at explicit suspension boundaries when trajectories split.
constexpr uint64_t kInitialSharedStateBytes = 512 * 1024;

// One call owns its mutable workers. Rows are collected only after a batch
// finishes dispatch, and batch indices keep collection independent of scheduling.
struct SharedWorker {
    using Executor = std::variant<BatchExecutor, SharingExecutor>;
    Executor executor;
    uint32_t passed = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;

    SharedWorker(const SamplingPreparation& preparation, const PreparedBatchNoise& noise,
                 const EarlyPostselection* checks, const SharingKey* key,
                 const SamplingExecution& execution, SamplingOutputs outputs)
        : executor(make_executor(
              preparation.plan(), noise, checks, key, execution,
              outputs.rows() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors)),
          observable_ones(preparation.plan().num_observables()) {}

    static Executor make_executor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                                  const EarlyPostselection* checks, const SharingKey* key,
                                  const SamplingExecution& execution, BatchOutputMode mode) {
        // Width-zero plans have no coefficient evolution to share. Their packed
        // executor applies classical checks without per-shot group bookkeeping.
        if (plan.peak_active_width() == 0) {
            return Executor(std::in_place_type<BatchExecutor>, plan, execution.batch_size, mode,
                            BatchSamplingMode::Ordinary, &noise);
        }
        const uint32_t initial_states = static_cast<uint32_t>(std::min<uint64_t>(
            {execution.batch_size, kMaxInitialSharedStates,
             std::max<uint64_t>(
                 1, kInitialSharedStateBytes / State::storage_bytes(plan.peak_active_width()))}));
        return Executor(std::in_place_type<SharingExecutor>, plan, noise, *checks,
                        execution.batch_size, initial_states, true, key,
                        *execution.max_worker_state_bytes, mode);
    }
};

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
    const auto* checks =
        plan.peak_active_width() || plan.has_postselection() ? &preparation.checks() : nullptr;
    const auto* key = execution.group_by_noise ? &preparation.sharing_key() : nullptr;
    const uint32_t capacity = execution.batch_size;
    const uint32_t batch_count = static_cast<uint32_t>((uint64_t{shots} + capacity - 1) / capacity);
    batch_detail::SurvivorRowCollector rows(shots, capacity, survivors && outputs.rows());
    const auto root = make_seed_root(shots, seed);
    for (auto& worker : workers) {
        if (worker) {
            worker->passed = worker->logical_errors = 0;
            std::fill(worker->observable_ones.begin(), worker->observable_ones.end(), 0);
        }
    }
    (void)run_shot_ranges(
        batch_count, execution.shot_workers,
        [&](uint32_t index) {
            auto& worker = workers[index];
            if (!worker) {
                worker = std::make_unique<SharedWorker>(preparation, noise, checks, key, execution,
                                                        outputs);
            }
            return worker.get();
        },
        [&](auto& worker, ShotRange range) {
            std::visit(
                [&](auto& executor) {
                    for (uint32_t batch = range.begin; batch < range.end; ++batch) {
                        const uint32_t first = static_cast<uint32_t>(uint64_t{batch} * capacity);
                        const auto lanes = std::min(capacity, shots - first);
                        if constexpr (std::is_same_v<std::decay_t<decltype(executor)>,
                                                     BatchExecutor>) {
                            executor.run_batch(root, first, lanes, checks);
                        } else {
                            executor.run_batch(root, first, lanes);
                        }
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
        for (const auto& worker : workers) {
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
    return result;
}

bool prefer_shared(const Program& program, const ExecutablePlan& plan, SamplingOutputs outputs,
                   bool survivors, BatchOutputMode mode, uint32_t shots, uint32_t workers,
                   uint64_t state_budget) {
    if (shots < kDefaultMinAutoBatchShots)
        return false;
    const auto width = plan.peak_active_width();
    const auto lanes =
        resolve_shared_execution_policy(kMaxAutoSharedBatchShots, shots, workers).lane_capacity;
    const bool bulk = lanes == kMaxAutoSharedBatchShots;
    if (width == 0)
        return bulk;

    constexpr uint32_t kMinSharedStates = 64;
    const auto states = state_budget / State::storage_bytes(width);
    if (states < kMinSharedStates)
        return false;
    constexpr uint64_t kMinWork = 2 * 1024 * 1024;
    const auto work =
        saturating_add_u64(plan.estimated_batch_lane_work(mode), plan.num_noise_sites());
    const bool expensive = saturating_multiply_u64(work, lanes) >= kMinWork;
    constexpr uint32_t kMinProfileShots = 256;
    if (!plan.num_presampled_symbols() && !plan.has_readout_noise()) {
        return !plan.has_postselection() && width >= 8 &&
               (shots >= kMinProfileShots || expensive) && plan.has_reusable_unitary_prefix();
    }
    if (shots < kMinProfileShots || (!bulk && !expensive))
        return false;
    const auto& prep = program.sampling_preparation(
        outputs, {.early_postselection = plan.has_postselection(), .survivor_counts = survivors});
    const auto capacity = prep.probe_capacity();
    if (capacity < 2 || capacity > states)
        return false;
    const auto& profile = prep.trajectory_work();
    // With a truncated profile, the unobserved tail might diverge. Require a
    // prepared check that can also remove trajectories before their evolution.
    if (profile.truncated() && !prep.checks().num_checks() &&
        !prep.plan().has_early_postselection(mode)) {
        return false;
    }
    return profile.shared <= profile.independent / 3;
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
    const bool shared =
        options.sharing == true || (!options.sharing && !options.batch_size && supported &&
                                    prefer_shared(program, compiled, outputs, survivors, mode,
                                                  shots, workers, options.max_worker_state_bytes));
    const bool compare_executors = !options.sharing && supported && width != 0 && shots > 1 &&
                                   options.tuning_budget_seconds.value_or(0) > 0;
    const bool fits_states = !(shared || compare_executors) || width == 0 ||
                             options.max_worker_state_bytes / State::storage_bytes(width) >= 2;
    if (options.sharing == true && !fits_states) {
        throw std::invalid_argument("max_worker_state_bytes must hold at least two shared states");
    }
    const auto shared_noise_mode = options.noise_preparation.value_or(BatchNoiseMode::SiteSparse);
    const SamplingPreparation* shared_preparation = nullptr;
    uint32_t safe = 0;
    bool group_by_noise = false;
    if ((shared || compare_executors) && fits_states) {
        shared_preparation = &program.sampling_preparation(
            outputs,
            {.early_postselection = compiled.has_postselection(), .survivor_counts = survivors});
        assert(shared_preparation->plan().peak_active_width() == width);
        const auto& noise = shared_preparation->noise(shared_noise_mode);
        if (width != 0 && options.group_by_noise.value_or(true) &&
            (options.group_by_noise == true ||
             (shots >= kMinAutoSharedLanesPerWorker && compiled.num_noise_sites() != 0))) {
            const auto& key = shared_preparation->sharing_key();
            group_by_noise = options.group_by_noise.value_or(key.prefers_grouping());
        }
        const auto* key = group_by_noise ? &shared_preparation->sharing_key() : nullptr;
        safe = max_safe_shared_capacity(shared_preparation->plan(), noise, key,
                                        options.max_worker_state_bytes, mode);
        if (safe < 2 && options.sharing == true) {
            throw std::invalid_argument(
                "shared execution requires capacity for at least two lanes");
        }
    }
    auto run_ordinary = [&](std::optional<uint32_t> capacity, std::optional<double> budget,
                            std::optional<ThreadLayout> run_layout) {
        const auto& preparation = program.sampling_preparation(
            outputs, {.early_postselection = false, .survivor_counts = survivors});
        const PreparedBatchNoise* noise = nullptr;
        if (shots != 0 && capacity != 1 &&
            (capacity || (budget.value_or(0) > 0 && shots > 1 && layout.intra_shot_workers == 1) ||
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
        execution.group_by_noise = group_by_noise;
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
                const uint64_t extra =
                    survivors ? uint64_t{compiled.num_observables()} * sizeof(uint64_t) : 0;
                auto ordinary_policy = resolve_batch_execution_policy(
                    ordinary.plan(), shots, workers, 1, mode, std::nullopt,
                    BatchSamplingMode::Ordinary, extra, 0, 0, noise);
                if (ordinary_policy.lane_capacity == 1) {
                    ordinary_policy.worker_count = workers;
                }
                auto ordinary_candidates = batch_detail::batch_tuning_candidates(
                    ordinary.plan(), shots, workers, 1, mode, BatchSamplingMode::Ordinary, extra,
                    ordinary_policy, 0, 0, noise);
                // Time the production baseline first, followed by the other executor's
                // baseline, so small budgets can compare executors before varying capacity.
                if (!shared) {
                    candidates.swap(ordinary_candidates);
                    policy = ordinary_policy;
                }
                if (!candidates.empty() && !ordinary_candidates.empty()) {
                    candidates.insert(candidates.begin() + 1, ordinary_candidates.front());
                    candidates.insert(candidates.end(), ordinary_candidates.begin() + 1,
                                      ordinary_candidates.end());
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
                        auto workers = std::make_shared<std::vector<std::unique_ptr<SharedWorker>>>(
                            trial.shot_workers);
                        return [&, trial, workers = std::move(workers)](uint32_t count,
                                                                        uint64_t trial_seed) {
                            (void)run_shared<Result>(preparation, count, trial_seed, outputs, trial,
                                                     *workers);
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
    std::vector<std::unique_ptr<SharedWorker>> call_workers(execution.shot_workers);
    auto result = run_shared<Result>(preparation, shots, seed, outputs, execution, call_workers);
    result.batch_tuning = std::move(tuning);
    return result;
}

}  // namespace

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
