// Native execution selection, per-call shared workers, and ordered row collection.

#include "clifft/api/sampler.h"

#include "clifft/api/sampling_preparation.h"
#include "clifft/sampling/batch/output.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace clifft {
namespace {

using namespace sampling;

constexpr uint32_t kInitialSharedStates = 128;

// One call owns its mutable workers. Rows are collected only after a batch
// finishes dispatch, and batch indices keep collection independent of scheduling.
struct SharedWorker {
    SharingExecutor executor;
    uint32_t passed = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
    std::vector<uint32_t> rows;

    SharedWorker(const SamplingPreparation& preparation, const PreparedBatchNoise& noise,
                 const EarlyPostselection& checks, const SharingKey* key,
                 const SamplingExecution& execution, SamplingOutputs outputs)
        : executor(preparation.plan(), noise, checks, execution.batch_size,
                   std::min(kInitialSharedStates, execution.batch_size), true, key,
                   *execution.max_worker_state_bytes,
                   outputs.rows() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors),
          observable_ones(preparation.plan().num_observables()),
          rows(outputs.rows() ? execution.batch_size : 0) {
        std::iota(rows.begin(), rows.end(), uint32_t{0});
    }
};

template <typename Result>
Result run_shared(const SamplingPreparation& preparation, uint32_t shots,
                  std::optional<uint64_t> seed, SamplingOutputs outputs,
                  const SamplingExecution& execution) {
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
    const auto& checks = preparation.checks();
    const auto* key = execution.group_by_noise ? &preparation.sharing_key() : nullptr;
    const uint32_t capacity = execution.batch_size;
    const uint32_t batch_count = static_cast<uint32_t>((uint64_t{shots} + capacity - 1) / capacity);
    batch_detail::SurvivorRowCollector rows(shots, capacity, survivors && outputs.rows());
    const auto root = make_seed_root(shots, seed);
    auto workers = run_shot_ranges(
        batch_count, execution.shot_workers,
        [&](uint32_t) {
            return std::make_unique<SharedWorker>(preparation, noise, checks, key, execution,
                                                  outputs);
        },
        [&](auto& worker, ShotRange range) {
            for (uint32_t batch = range.begin; batch < range.end; ++batch) {
                const uint32_t first = static_cast<uint32_t>(uint64_t{batch} * capacity);
                auto& executor = worker->executor;
                executor.run_batch(root, first, std::min(capacity, shots - first));
                if constexpr (survivors) {
                    worker->passed += executor.surviving_shots();
                    worker->logical_errors +=
                        executor.accumulate_survivor_counts(worker->observable_ones);
                    if (outputs.rows()) {
                        auto& batch_rows = rows.batch(first);
                        batch_detail::resize_output_rows(batch_rows, executor.surviving_shots(),
                                                         plan, outputs);
                        batch_detail::copy_batch_rows(
                            batch_rows, executor, plan, outputs,
                            std::span(worker->rows).first(executor.surviving_shots()));
                    }
                } else {
                    batch_detail::copy_batch_rows(result, executor, plan, outputs);
                }
            }
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
            rows.finish(result, plan, outputs);
        }
    }
    return result;
}

template <typename Result>
Result sample_program(const Program& program, uint32_t shots, std::optional<uint64_t> seed,
                      SamplingOutputs outputs, const SamplingOptions& options) {
    constexpr bool survivors = std::is_same_v<Result, SamplingSurvivorResult>;
    const auto& compiled = program.sampling_plan(outputs.measurements());
    if constexpr (!survivors) {
        if (!outputs.rows() || program.original().has_postselection()) {
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
    const auto layout =
        resolve_thread_layout(compiled, shots, options.threads, options.thread_layout);
    const bool supported = !(outputs.expectations() && compiled.num_exp_vals() != 0) &&
                           options.batch_size != 1 && layout.intra_shot_workers == 1;
    if (options.sharing == true && !supported) {
        throw std::invalid_argument(
            "shared execution requires no expectation probes or scalar override");
    }

    const uint32_t workers = std::max(uint32_t{1}, layout.shot_workers);
    bool shared = options.sharing.value_or(false);
    if (!options.sharing && !options.batch_size && supported && compiled.peak_active_width() != 0) {
        // Conservative admission is based on the compiled plan, before requesting
        // cached preparation. Numeric batches retain ordinary execution by default.
        // Noise-heavy preparation needs enough work per worker to repay its cold
        // cost. These call-size classes cover cultivation through its escape stage.
        // A few selected detectors alone do not establish useful rejection.
        constexpr uint32_t kMinSelectedDetectors = 8;
        const uint32_t noise_symbols = compiled.num_presampled_symbols();
        constexpr uint32_t kLargeNoiseProgram = 32768;
        constexpr uint32_t kMinSharedShotsPerWorker = 8192;
        const uint32_t min_call_shots = noise_symbols >= kLargeNoiseProgram ? 65536
                                        : noise_symbols >= 8192             ? 32768
                                        : noise_symbols >= 4096             ? 16384
                                                                            : 8192;
        const uint64_t required_shots =
            noise_symbols >= kLargeNoiseProgram
                ? uint64_t{min_call_shots} * workers
                : std::max(uint64_t{min_call_shots}, uint64_t{kMinSharedShotsPerWorker} * workers);
        const bool rejection_workload =
            shots >= required_shots && noise_symbols != 0 &&
            compiled.num_postselected_detectors() >= kMinSelectedDetectors;
        constexpr uint32_t kMinUnitaryPrefixShots = 256;
        // Later rotations can remain after measurement scheduling. Without new
        // active coordinates, branching only consumes the prepared state.
        const bool unitary_prefix =
            !compiled.has_postselection() && noise_symbols == 0 && !compiled.has_readout_noise() &&
            compiled.peak_active_width() >= 8 && shots >= kMinUnitaryPrefixShots &&
            !compiled.has_promotions_after_measurement();
        constexpr uint32_t kMinSharedStates = 64;
        shared =
            (rejection_workload || unitary_prefix) &&
            options.max_worker_state_bytes / State::storage_bytes(compiled.peak_active_width()) >=
                kMinSharedStates;
    }
    shared &= supported && compiled.peak_active_width() != 0;
    if (shared &&
        options.max_worker_state_bytes < State::storage_bytes(compiled.peak_active_width())) {
        if (options.sharing == true) {
            throw std::invalid_argument("max_worker_state_bytes cannot hold one shared state");
        }
        shared = false;
    }
    if (!shared) {
        if (options.group_by_noise == true && compiled.peak_active_width() != 0) {
            throw std::invalid_argument("noise grouping requires shared execution");
        }
        const auto& preparation = program.sampling_preparation(
            outputs, {.early_postselection = false, .survivor_counts = survivors});
        const PreparedBatchNoise* noise = nullptr;
        if (shots != 0 && options.batch_size != 1) {
            noise = &preparation.noise(options.noise_preparation.value_or(BatchNoiseMode::Dense));
        }
        if constexpr (survivors) {
            return sampling::sample_survivors(
                preparation.plan(), shots, seed, outputs, options.threads, options.thread_layout,
                options.batch_size, options.tuning_budget_seconds, noise);
        } else {
            return sampling::sample(preparation.plan(), shots, seed, options.threads,
                                    options.thread_layout, options.batch_size,
                                    options.tuning_budget_seconds, outputs, noise);
        }
    }

    SamplingExecution execution;
    execution.executor = SamplingExecutor::Shared;
    execution.max_worker_state_bytes = options.max_worker_state_bytes;
    execution.early_postselection = compiled.has_postselection();
    execution.group_by_noise = options.group_by_noise.value_or(false);
    execution.noise_preparation = options.noise_preparation.value_or(BatchNoiseMode::SiteSparse);
    const auto& preparation = program.sampling_preparation(
        outputs,
        {.early_postselection = execution.early_postselection, .survivor_counts = survivors});
    const auto& noise = preparation.noise(*execution.noise_preparation);
    const auto* key = execution.group_by_noise ? &preparation.sharing_key() : nullptr;
    const auto mode = outputs.rows() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    const uint32_t safe = max_safe_shared_capacity(preparation.plan(), noise, key,
                                                   options.max_worker_state_bytes, mode);
    auto policy = resolve_shared_execution_policy(safe, shots, workers);
    if (options.batch_size && shots != 0) {
        policy.lane_capacity = std::min(*options.batch_size, shots);
        policy.worker_count = static_cast<uint32_t>(std::min<uint64_t>(
            workers, (uint64_t{shots} + policy.lane_capacity - 1) / policy.lane_capacity));
    }
    execution.batch_size = policy.lane_capacity;
    execution.shot_workers = shots == 0 ? 0 : policy.worker_count;
    std::optional<BatchTuningReport> tuning;
    if (options.tuning_budget_seconds) {
        tuning.emplace();
        tuning->batch_size = tuning->baseline_batch_size = policy.lane_capacity;
        tuning->shot_workers = execution.shot_workers;
        if (shots == 0) {
            tuning->stop_reason = "zero_shots";
        } else if (*options.tuning_budget_seconds == 0) {
            tuning->stop_reason = "zero_budget";
        } else {
            const auto now = [] {
                return std::chrono::duration<double>(
                           std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            };
            auto candidates = shared_tuning_candidates(safe, shots, workers);
            // Numeric one is the scalar override in the integrated API.
            if (candidates.size() > 1) {
                std::erase_if(candidates, [](const auto& candidate) {
                    return candidate.policy.lane_capacity == 1;
                });
            }
            tuning = batch_detail::sweep_batch_candidates(
                candidates, policy, 1, *options.tuning_budget_seconds, make_seed_root(shots, seed),
                now, [&](BatchExecutionPolicy candidate) {
                    auto trial = execution;
                    trial.batch_size = candidate.lane_capacity;
                    trial.shot_workers = candidate.worker_count;
                    return [&, trial](uint32_t count, uint64_t trial_seed) {
                        (void)run_shared<Result>(preparation, count, trial_seed, outputs, trial);
                    };
                });
            execution.batch_size = tuning->batch_size;
            execution.shot_workers = tuning->shot_workers;
        }
    }
    auto result = run_shared<Result>(preparation, shots, seed, outputs, execution);
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
