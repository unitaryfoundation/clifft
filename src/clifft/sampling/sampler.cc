#include "clifft/sampling/sampler.h"

#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/executor.h"
#include "clifft/util/fault_sampling.h"
#include "clifft/util/intra_shot_parallel.h"
#include "clifft/util/shot_parallel.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace clifft::sampling {

namespace {

void reseed_executor_for_shot(Executor& executor, const SeedRoot& root, uint32_t shot) noexcept {
    const std::array<uint64_t, 4> words = derive_state(root, shot, kSamplingExecutorDomain);
    executor.reseed_full(words[0], words[1], words[2], words[3]);
}

struct SamplingWorker {
    SamplingWorker(const ExecutablePlan& plan, uint32_t intra_shot_workers,
                   uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width) {}

    Executor executor;
};

struct ConditionedSamplingWorker {
    ConditionedSamplingWorker(const ExecutablePlan& plan,
                              std::shared_ptr<const KFaultDistribution> fault_distribution,
                              uint32_t intra_shot_workers, uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width),
          fault_sampler(std::move(fault_distribution)) {}

    Executor executor;
    KFaultSampler fault_sampler;
};

struct SurvivorCounts {
    explicit SurvivorCounts(uint32_t num_observables) : observable_ones(num_observables, 0) {}

    uint32_t passed_shots = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
};

struct SurvivorWorker {
    SurvivorWorker(const ExecutablePlan& plan, uint32_t intra_shot_workers,
                   uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width),
          counts(plan.num_observables()) {}

    Executor executor;
    SurvivorCounts counts;
};

struct ConditionedSurvivorWorker {
    ConditionedSurvivorWorker(const ExecutablePlan& plan,
                              std::shared_ptr<const KFaultDistribution> fault_distribution,
                              uint32_t intra_shot_workers, uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width),
          fault_sampler(std::move(fault_distribution)),
          counts(plan.num_observables()) {}

    Executor executor;
    KFaultSampler fault_sampler;
    SurvivorCounts counts;
};

struct BatchSamplingWorker {
    BatchSamplingWorker(const ExecutablePlan& plan, uint32_t capacity) : executor(plan, capacity) {}

    BatchExecutor executor;
};

struct ConditionedBatchSamplingWorker {
    ConditionedBatchSamplingWorker(const ExecutablePlan& plan,
                                   std::shared_ptr<const KFaultDistribution> fault_distribution,
                                   uint32_t capacity)
        : executor(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::FixedFaults),
          fault_sampler(std::move(fault_distribution)) {}

    BatchExecutor executor;
    KFaultSampler fault_sampler;
};

struct BatchSurvivorWorker {
    BatchSurvivorWorker(const ExecutablePlan& plan, uint32_t capacity, bool keep_records)
        : executor(plan, capacity,
                   keep_records ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors),
          counts(plan.num_observables()) {}

    BatchExecutor executor;
    SurvivorCounts counts;
};

struct ConditionedBatchSurvivorWorker {
    ConditionedBatchSurvivorWorker(const ExecutablePlan& plan,
                                   std::shared_ptr<const KFaultDistribution> fault_distribution,
                                   uint32_t capacity, bool keep_records)
        : executor(plan, capacity,
                   keep_records ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors,
                   BatchSamplingMode::FixedFaults),
          fault_sampler(std::move(fault_distribution)),
          counts(plan.num_observables()) {}

    BatchExecutor executor;
    KFaultSampler fault_sampler;
    SurvivorCounts counts;
};

uint64_t survivor_worker_bytes(const ExecutablePlan& plan) noexcept {
    return static_cast<uint64_t>(plan.num_observables()) * sizeof(uint64_t);
}

ThreadLayout resolve_thread_layout(const ExecutablePlan& plan, uint32_t shots,
                                   uint32_t requested_threads,
                                   std::optional<ThreadLayout> override) {
    if (override.has_value()) {
        if (override->shot_workers == 0 || override->intra_shot_workers == 0) {
            throw std::invalid_argument("thread_layout worker counts must be positive");
        }
        if (override->intra_shot_workers > 1 && !intra_shot_parallelism_available()) {
            throw std::invalid_argument(
                "thread_layout intra-shot workers require an OpenMP-enabled build");
        }
        if (override->intra_shot_workers > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("thread_layout intra-shot worker count is too large");
        }
        override->shot_workers = std::min(override->shot_workers, shots);
        if (override->shot_workers > 1 && override->intra_shot_workers > 1 &&
            openmp_process_binding_active()) {
            throw std::invalid_argument("hybrid thread_layout requires OMP_PROC_BIND=false");
        }
        if (!should_parallelize_intra_shot(plan.peak_active_width(), override->intra_shot_workers,
                                           override->intra_shot_min_active_width)) {
            override->intra_shot_workers = 1;
        }
        return *override;
    }

    const uint32_t budget = resolve_thread_budget(requested_threads);
    if (shots != 0 && shots < budget &&
        should_parallelize_intra_shot(plan.peak_active_width(), budget,
                                      kDefaultIntraShotMinActiveWidth)) {
        if (budget > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("intra-shot thread budget is too large");
        }
        return {.shot_workers = 1,
                .intra_shot_workers = budget,
                .intra_shot_min_active_width = kDefaultIntraShotMinActiveWidth};
    }
    return {.shot_workers = std::min(shots, budget), .intra_shot_workers = 1};
}

size_t checked_output_size(uint32_t shots, size_t stride) {
    if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("sampling output size exceeds size_t range");
    }
    return static_cast<size_t>(shots) * stride;
}

template <typename Output>
void copy_batch_lane(Output& output, const BatchExecutor& executor, uint32_t lane, uint32_t shot,
                     const ExecutablePlan& plan) noexcept {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        output.measurements[static_cast<size_t>(shot) * plan.num_visible_records() + record] =
            static_cast<uint8_t>(executor.measurement(lane, record));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        output.detectors[static_cast<size_t>(shot) * plan.num_detectors() + detector] =
            static_cast<uint8_t>(executor.detector(lane, detector));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        output.observables[static_cast<size_t>(shot) * plan.num_observables() + observable] =
            static_cast<uint8_t>(executor.observable(lane, observable));
    }
}

void copy_batch_expectations(std::span<double> output, const BatchExecutor& executor,
                             uint32_t num_exp_vals) noexcept {
    if (num_exp_vals == 0) {
        return;
    }
    // The executor stores columns and the result stores rows. Copy neighboring
    // lanes together to reuse each source cache line without scattering writes
    // across the whole batch's output rows.
    constexpr uint32_t kTileLanes = 8;
    const uint32_t lanes = executor.surviving_shots();
    for (uint32_t first = 0; first < lanes; first += kTileLanes) {
        const uint32_t count = std::min(kTileLanes, lanes - first);
        // Hoist the shot lookup so it is cheap even without cross-file inlining.
        std::array<size_t, kTileLanes> rows{};
        for (uint32_t i = 0; i < count; ++i) {
            rows[i] = static_cast<size_t>(executor.shot_index(first + i)) * num_exp_vals;
        }
        for (uint32_t exp_val = 0; exp_val < num_exp_vals; ++exp_val) {
            for (uint32_t i = 0; i < count; ++i) {
                output[rows[i] + exp_val] = executor.exp_val(first + i, exp_val);
            }
        }
    }
}

template <typename T>
void compact_survivor_rows(std::vector<T>& values, std::span<const uint8_t> survived, size_t stride,
                           uint32_t passed_shots) {
    if (stride != 0) {
        size_t output_row = 0;
        for (size_t shot = 0; shot < survived.size(); ++shot) {
            if (survived[shot] == 0) {
                continue;
            }
            if (output_row != shot) {
                std::copy_n(values.begin() + shot * stride, stride,
                            values.begin() + output_row * stride);
            }
            ++output_row;
        }
    }
    values.resize(static_cast<size_t>(passed_shots) * stride);
}

template <typename MakeWorker, typename RunShot>
SamplingResult sample_fixed_rows(const ExecutablePlan& plan, uint32_t shots,
                                 std::optional<uint64_t> seed, ThreadLayout thread_layout,
                                 MakeWorker&& make_worker, RunShot&& run_shot) {
    SamplingResult result;
    result.measurements.resize(checked_output_size(shots, plan.num_visible_records()));
    result.detectors.resize(checked_output_size(shots, plan.num_detectors()));
    result.observables.resize(checked_output_size(shots, plan.num_observables()));
    result.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    if (shots == 0) {
        return result;
    }

    const SeedRoot root = make_seed_root(shots, seed);
    (void)run_shot_ranges(
        shots, thread_layout.shot_workers, std::forward<MakeWorker>(make_worker),
        [&](auto& worker_handle, ShotRange range) {
            auto& worker = *worker_handle;
            Executor& executor = worker.executor;
            for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                reseed_executor_for_shot(executor, root, shot);
                run_shot(worker);
                std::ranges::copy(executor.visible_records(),
                                  result.measurements.begin() +
                                      static_cast<size_t>(shot) * plan.num_visible_records());
                std::ranges::copy(
                    executor.detectors(),
                    result.detectors.begin() + static_cast<size_t>(shot) * plan.num_detectors());
                std::ranges::copy(executor.observables(),
                                  result.observables.begin() +
                                      static_cast<size_t>(shot) * plan.num_observables());
                std::ranges::copy(
                    executor.exp_vals(),
                    result.exp_vals.begin() + static_cast<size_t>(shot) * plan.num_exp_vals());
            }
        });
    return result;
}

template <typename MakeWorker, typename RunBatch>
SamplingResult sample_fixed_batches(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed, BatchExecutionPolicy batch_policy,
                                    MakeWorker&& make_worker, RunBatch&& run_batch) {
    SamplingResult result;
    result.measurements.resize(checked_output_size(shots, plan.num_visible_records()));
    result.detectors.resize(checked_output_size(shots, plan.num_detectors()));
    result.observables.resize(checked_output_size(shots, plan.num_observables()));
    result.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    if (shots == 0) {
        return result;
    }

    const SeedRoot root = make_seed_root(shots, seed);
    const uint32_t batch_capacity = batch_policy.lane_capacity;
    (void)run_shot_ranges(
        shots, batch_policy.worker_count, std::forward<MakeWorker>(make_worker),
        [&](auto& worker_handle, ShotRange range) {
            auto& worker = *worker_handle;
            BatchExecutor& executor = worker.executor;
            for (uint32_t offset = range.begin; offset < range.end;) {
                const uint32_t batch = std::min(batch_capacity, range.end - offset);
                run_batch(worker, root, offset, batch);
                assert(executor.surviving_shots() == batch &&
                       "fixed-row batch must retain every shot");
                for (uint32_t lane = 0; lane < batch; ++lane) {
                    const uint32_t shot = executor.shot_index(lane);
                    copy_batch_lane(result, executor, lane, shot, plan);
                }
                copy_batch_expectations(result.exp_vals, executor, plan.num_exp_vals());
                offset += batch;
            }
        },
        batch_capacity);
    return result;
}

template <typename MakeWorker, typename RunShot>
SamplingSurvivorResult sample_surviving_rows(const ExecutablePlan& plan, uint32_t shots,
                                             std::optional<uint64_t> seed, bool keep_records,
                                             ThreadLayout thread_layout, MakeWorker&& make_worker,
                                             RunShot&& run_shot) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    if (keep_records) {
        result.measurements.resize(checked_output_size(shots, plan.num_visible_records()));
        result.detectors.resize(checked_output_size(shots, plan.num_detectors()));
        result.observables.resize(checked_output_size(shots, plan.num_observables()));
        result.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    }
    std::vector<uint8_t> survived(keep_records ? shots : 0, 0);
    const SeedRoot root = make_seed_root(shots, seed);
    auto workers = run_shot_ranges(
        shots, thread_layout.shot_workers, std::forward<MakeWorker>(make_worker),
        [&](auto& worker_handle, ShotRange range) {
            auto& worker = *worker_handle;
            Executor& executor = worker.executor;
            SurvivorCounts& counts = worker.counts;
            for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                reseed_executor_for_shot(executor, root, shot);
                run_shot(worker);
                if (executor.discarded()) {
                    continue;
                }
                ++counts.passed_shots;
                bool logical_error = false;
                for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
                    const bool value = executor.observables()[observable] != 0;
                    counts.observable_ones[observable] += static_cast<uint64_t>(value);
                    logical_error |= value;
                }
                counts.logical_errors += static_cast<uint32_t>(logical_error);
                if (keep_records) {
                    survived[shot] = 1;
                    std::ranges::copy(executor.visible_records(),
                                      result.measurements.begin() +
                                          static_cast<size_t>(shot) * plan.num_visible_records());
                    std::ranges::copy(executor.detectors(),
                                      result.detectors.begin() +
                                          static_cast<size_t>(shot) * plan.num_detectors());
                    std::ranges::copy(executor.observables(),
                                      result.observables.begin() +
                                          static_cast<size_t>(shot) * plan.num_observables());
                    std::ranges::copy(
                        executor.exp_vals(),
                        result.exp_vals.begin() + static_cast<size_t>(shot) * plan.num_exp_vals());
                }
            }
        });
    for (const auto& worker : workers) {
        result.passed_shots += worker->counts.passed_shots;
        result.logical_errors += worker->counts.logical_errors;
        for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
            result.observable_ones[observable] += worker->counts.observable_ones[observable];
        }
    }
    if (keep_records) {
        compact_survivor_rows(result.measurements, survived, plan.num_visible_records(),
                              result.passed_shots);
        compact_survivor_rows(result.detectors, survived, plan.num_detectors(),
                              result.passed_shots);
        compact_survivor_rows(result.observables, survived, plan.num_observables(),
                              result.passed_shots);
        compact_survivor_rows(result.exp_vals, survived, plan.num_exp_vals(), result.passed_shots);
    }
    return result;
}

template <typename MakeWorker, typename RunBatch>
SamplingSurvivorResult sample_surviving_batches(const ExecutablePlan& plan, uint32_t shots,
                                                std::optional<uint64_t> seed, bool keep_records,
                                                BatchExecutionPolicy batch_policy,
                                                MakeWorker&& make_worker, RunBatch&& run_batch) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    if (keep_records) {
        result.measurements.resize(checked_output_size(shots, plan.num_visible_records()));
        result.detectors.resize(checked_output_size(shots, plan.num_detectors()));
        result.observables.resize(checked_output_size(shots, plan.num_observables()));
        result.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    }
    std::vector<uint8_t> survived(keep_records ? shots : 0, 0);
    const SeedRoot root = make_seed_root(shots, seed);
    const uint32_t batch_capacity = batch_policy.lane_capacity;
    auto workers = run_shot_ranges(
        shots, batch_policy.worker_count, std::forward<MakeWorker>(make_worker),
        [&](auto& worker_handle, ShotRange range) {
            auto& worker = *worker_handle;
            BatchExecutor& executor = worker.executor;
            SurvivorCounts& counts = worker.counts;
            for (uint32_t offset = range.begin; offset < range.end;) {
                const uint32_t batch = std::min(batch_capacity, range.end - offset);
                run_batch(worker, root, offset, batch);
                if (!keep_records) {
                    counts.passed_shots += executor.surviving_shots();
                    counts.logical_errors +=
                        executor.accumulate_survivor_counts(counts.observable_ones);
                    offset += batch;
                    continue;
                }
                for (uint32_t lane = 0; lane < executor.surviving_shots(); ++lane) {
                    ++counts.passed_shots;
                    bool logical_error = false;
                    for (uint32_t observable = 0; observable < plan.num_observables();
                         ++observable) {
                        const bool value = executor.observable(lane, observable);
                        counts.observable_ones[observable] += static_cast<uint64_t>(value);
                        logical_error |= value;
                    }
                    counts.logical_errors += static_cast<uint32_t>(logical_error);
                    const uint32_t shot = executor.shot_index(lane);
                    survived[shot] = 1;
                    copy_batch_lane(result, executor, lane, shot, plan);
                }
                copy_batch_expectations(result.exp_vals, executor, plan.num_exp_vals());
                offset += batch;
            }
        },
        batch_capacity);
    for (const auto& worker : workers) {
        result.passed_shots += worker->counts.passed_shots;
        result.logical_errors += worker->counts.logical_errors;
        for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
            result.observable_ones[observable] += worker->counts.observable_ones[observable];
        }
    }
    if (keep_records) {
        compact_survivor_rows(result.measurements, survived, plan.num_visible_records(),
                              result.passed_shots);
        compact_survivor_rows(result.detectors, survived, plan.num_detectors(),
                              result.passed_shots);
        compact_survivor_rows(result.observables, survived, plan.num_observables(),
                              result.passed_shots);
        compact_survivor_rows(result.exp_vals, survived, plan.num_exp_vals(), result.passed_shots);
    }
    return result;
}

template <typename Worker>
Worker* reset_probe_worker(Worker* worker) {
    if constexpr (requires { worker->counts; }) {
        worker->counts.passed_shots = 0;
        worker->counts.logical_errors = 0;
        std::ranges::fill(worker->counts.observable_ones, 0);
    }
    return worker;
}

template <typename Result, typename MakeScalar, typename MakePacked, typename RunShot,
          typename RunBatch>
Result sample_configured(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed,
                         ThreadLayout layout, std::optional<uint32_t> batch_size,
                         std::optional<double> tuning_budget_seconds, BatchOutputMode output_mode,
                         BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes,
                         bool keep_records, MakeScalar&& make_scalar, MakePacked&& make_packed,
                         RunShot&& run_shot, RunBatch&& run_batch) {
    if (tuning_budget_seconds.has_value()) {
        if (batch_size.has_value()) {
            throw std::invalid_argument("batch tuning requires automatic batch_size selection");
        }
        if (!is_finite_non_negative(*tuning_budget_seconds)) {
            throw std::invalid_argument("tuning_budget_seconds must be finite and non-negative");
        }
    }
    BatchExecutionPolicy policy = resolve_batch_execution_policy(
        plan, shots, layout.shot_workers, layout.intra_shot_workers, output_mode, batch_size,
        sampling_mode, additional_worker_bytes);
    if (policy.lane_capacity == 1) {
        policy.worker_count = layout.shot_workers;
    }
    auto run = [&](uint32_t count, std::optional<uint64_t> run_seed,
                   BatchExecutionPolicy run_policy, auto&& scalar_factory,
                   auto&& packed_factory) -> Result {
        if constexpr (kPackedBatchExecutionAvailable) {
            if (run_policy.lane_capacity > 1) {
                if constexpr (std::is_same_v<Result, SamplingResult>) {
                    return sample_fixed_batches(plan, count, run_seed, run_policy, packed_factory,
                                                run_batch);
                } else {
                    return sample_surviving_batches(plan, count, run_seed, keep_records, run_policy,
                                                    packed_factory, run_batch);
                }
            }
        }
        ThreadLayout run_layout = layout;
        run_layout.shot_workers = run_policy.worker_count;
        if constexpr (std::is_same_v<Result, SamplingResult>) {
            return sample_fixed_rows(plan, count, run_seed, run_layout, scalar_factory, run_shot);
        } else {
            return sample_surviving_rows(plan, count, run_seed, keep_records, run_layout,
                                         scalar_factory, run_shot);
        }
    };
    std::optional<BatchTuningReport> tuning;
    if (tuning_budget_seconds.has_value()) {
        const auto start = std::chrono::steady_clock::now();
        const auto now = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        tuning.emplace();
        tuning->batch_size = policy.lane_capacity;
        tuning->baseline_batch_size = policy.lane_capacity;
        tuning->shot_workers = policy.worker_count;
        tuning->intra_shot_workers = layout.intra_shot_workers;
        if (shots == 0) {
            tuning->stop_reason = "zero_shots";
        } else if (*tuning_budget_seconds == 0) {
            tuning->stop_reason = "zero_budget";
        } else {
            const auto candidates = batch_detail::batch_tuning_candidates(
                plan, shots, layout.shot_workers, layout.intra_shot_workers, output_mode,
                sampling_mode, additional_worker_bytes, policy);
            if (candidates.size() < 2) {
                tuning->stop_reason = "single_candidate";
            } else {
                const SeedRoot calibration_root = make_seed_root(shots, seed);
                auto make_probe = [&](BatchExecutionPolicy probe_policy) {
                    using ScalarHandle = decltype(make_scalar(uint32_t{}));
                    using PackedHandle = decltype(make_packed(uint32_t{}, uint32_t{}));
                    std::vector<ScalarHandle> scalar_workers;
                    std::vector<PackedHandle> packed_workers;
                    if constexpr (kPackedBatchExecutionAvailable) {
                        if (probe_policy.lane_capacity > 1) {
                            packed_workers.reserve(probe_policy.worker_count);
                            for (uint32_t i = 0; i < probe_policy.worker_count; ++i) {
                                packed_workers.push_back(
                                    make_packed(i, probe_policy.lane_capacity));
                            }
                        }
                    }
                    if (probe_policy.lane_capacity == 1) {
                        scalar_workers.reserve(probe_policy.worker_count);
                        for (uint32_t i = 0; i < probe_policy.worker_count; ++i) {
                            scalar_workers.push_back(make_scalar(i));
                        }
                    }
                    return [&, probe_policy, scalar_workers = std::move(scalar_workers),
                            packed_workers = std::move(packed_workers)](uint32_t count,
                                                                        uint64_t trial_seed) {
                        (void)run(
                            count, trial_seed, probe_policy,
                            [&](uint32_t i) { return reset_probe_worker(scalar_workers[i].get()); },
                            [&](uint32_t i) {
                                return reset_probe_worker(packed_workers[i].get());
                            });
                    };
                };
                const double remaining = std::max(0.0, *tuning_budget_seconds - now());
                tuning = batch_detail::sweep_batch_candidates(candidates, policy,
                                                              layout.intra_shot_workers, remaining,
                                                              calibration_root, now, make_probe);
                policy = {tuning->batch_size, tuning->shot_workers};
            }
        }
        tuning->elapsed_seconds = now();
    }
    Result result = run(shots, seed, policy, make_scalar,
                        [&](uint32_t i) { return make_packed(i, policy.lane_capacity); });
    result.batch_tuning = std::move(tuning);
    return result;
}

}  // namespace

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed,
                      uint32_t threads, std::optional<ThreadLayout> thread_layout,
                      std::optional<uint32_t> batch_size,
                      std::optional<double> tuning_budget_seconds) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "fixed-plan sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "batch sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row sampling does not support postselection; use sample_survivors");
    }

    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    return sample_configured<SamplingResult>(
        plan, shots, seed, resolved, batch_size, tuning_budget_seconds, BatchOutputMode::Rows,
        BatchSamplingMode::Ordinary, 0, true,
        [&](uint32_t) {
            return std::make_unique<SamplingWorker>(plan, resolved.intra_shot_workers,
                                                    resolved.intra_shot_min_active_width);
        },
        [&](uint32_t, uint32_t capacity) {
            return std::make_unique<BatchSamplingWorker>(plan, capacity);
        },
        [](SamplingWorker& worker) noexcept { worker.executor.run_shot(); },
        [](BatchSamplingWorker& worker, const SeedRoot& root, uint32_t first_shot,
           uint32_t batch) noexcept { worker.executor.run_batch(root, first_shot, batch); });
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed, uint32_t threads,
                                    std::optional<ThreadLayout> thread_layout,
                                    std::optional<uint32_t> batch_size,
                                    std::optional<double> tuning_budget_seconds) {
    return sample(plan, shots, seed, threads, thread_layout, batch_size, tuning_budget_seconds)
        .measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records,
                                        uint32_t threads, std::optional<ThreadLayout> thread_layout,
                                        std::optional<uint32_t> batch_size,
                                        std::optional<double> tuning_budget_seconds) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "survivor sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "survivor sampling requires a distribution for every presampled symbol");
    }

    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    const BatchOutputMode output_mode =
        keep_records ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    return sample_configured<SamplingSurvivorResult>(
        plan, shots, seed, resolved, batch_size, tuning_budget_seconds, output_mode,
        BatchSamplingMode::Ordinary, survivor_worker_bytes(plan), keep_records,
        [&](uint32_t) {
            return std::make_unique<SurvivorWorker>(plan, resolved.intra_shot_workers,
                                                    resolved.intra_shot_min_active_width);
        },
        [&](uint32_t, uint32_t capacity) {
            return std::make_unique<BatchSurvivorWorker>(plan, capacity, keep_records);
        },
        [](SurvivorWorker& worker) noexcept { worker.executor.run_shot(); },
        [](BatchSurvivorWorker& worker, const SeedRoot& root, uint32_t first_shot,
           uint32_t batch) noexcept { worker.executor.run_batch(root, first_shot, batch); });
}

SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                        std::optional<uint64_t> seed, uint32_t threads,
                        std::optional<ThreadLayout> thread_layout,
                        std::optional<uint32_t> batch_size,
                        std::optional<double> tuning_budget_seconds) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault sampling does not support instrument traps or trajectory drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault sampling requires a distribution for every presampled symbol");
    }
    if (plan.has_postselection()) {
        throw std::invalid_argument(
            "fixed-row forced-fault sampling does not support postselection; use "
            "sample_k_survivors");
    }
    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    if (shots == 0) {
        return sample(plan, shots, seed, threads, thread_layout, batch_size, tuning_budget_seconds);
    }
    const auto fault_distribution =
        std::make_shared<const KFaultDistribution>(plan.noise_site_probabilities(), k);
    return sample_configured<SamplingResult>(
        plan, shots, seed, resolved, batch_size, tuning_budget_seconds, BatchOutputMode::Rows,
        BatchSamplingMode::FixedFaults, fault_distribution->worker_scratch_bytes(), true,
        [&](uint32_t) {
            return std::make_unique<ConditionedSamplingWorker>(
                plan, fault_distribution, resolved.intra_shot_workers,
                resolved.intra_shot_min_active_width);
        },
        [&](uint32_t, uint32_t capacity) {
            return std::make_unique<ConditionedBatchSamplingWorker>(plan, fault_distribution,
                                                                    capacity);
        },
        [](ConditionedSamplingWorker& worker) noexcept {
            worker.executor.run_shot(worker.fault_sampler);
        },
        [](ConditionedBatchSamplingWorker& worker, const SeedRoot& root, uint32_t first_shot,
           uint32_t batch) noexcept {
            worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
        });
}

SamplingSurvivorResult sample_k_survivors(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                          std::optional<uint64_t> seed, bool keep_records,
                                          uint32_t threads,
                                          std::optional<ThreadLayout> thread_layout,
                                          std::optional<uint32_t> batch_size,
                                          std::optional<double> tuning_budget_seconds) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "forced-fault survivor sampling does not support instrument traps or trajectory "
            "drivers");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "forced-fault survivor sampling requires a distribution for every presampled symbol");
    }
    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    const BatchOutputMode output_mode =
        keep_records ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    if (shots == 0) {
        return sample_survivors(plan, shots, seed, keep_records, threads, thread_layout, batch_size,
                                tuning_budget_seconds);
    }
    const auto fault_distribution =
        std::make_shared<const KFaultDistribution>(plan.noise_site_probabilities(), k);
    return sample_configured<SamplingSurvivorResult>(
        plan, shots, seed, resolved, batch_size, tuning_budget_seconds, output_mode,
        BatchSamplingMode::FixedFaults,
        fault_distribution->worker_scratch_bytes() + survivor_worker_bytes(plan), keep_records,
        [&](uint32_t) {
            return std::make_unique<ConditionedSurvivorWorker>(
                plan, fault_distribution, resolved.intra_shot_workers,
                resolved.intra_shot_min_active_width);
        },
        [&](uint32_t, uint32_t capacity) {
            return std::make_unique<ConditionedBatchSurvivorWorker>(plan, fault_distribution,
                                                                    capacity, keep_records);
        },
        [](ConditionedSurvivorWorker& worker) noexcept {
            worker.executor.run_shot(worker.fault_sampler);
        },
        [](ConditionedBatchSurvivorWorker& worker, const SeedRoot& root, uint32_t first_shot,
           uint32_t batch) noexcept {
            worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
        });
}

std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                             std::span<const uint8_t> forced_records,
                                             size_t num_records) {
    if (plan.num_hidden_records() != 0) {
        throw std::invalid_argument(
            "record_probabilities() does not yet support programs with hidden measurement "
            "slots (e.g. R / reset gates). Compile without resets, or use sample() to "
            "marginalize.");
    }
    if (plan.num_presampled_symbols() != 0 || plan.has_readout_noise() || plan.has_instruments() ||
        plan.num_detectors() != 0 || plan.num_observables() != 0 || plan.has_postselection()) {
        throw std::invalid_argument(
            "record_probabilities() requires pure-state evolution with measurements: noise, "
            "transition instruments, detectors, observables, and post-selection are not "
            "supported.");
    }
    const size_t stride = plan.num_visible_records();
    if (stride == 0) {
        throw std::invalid_argument(
            "record probabilities require a plan with at least one visible record");
    }
    if (num_records > std::numeric_limits<size_t>::max() / stride ||
        forced_records.size() != num_records * stride) {
        throw std::invalid_argument(
            "record buffer length must equal num_records times visible records");
    }
    if (!std::ranges::all_of(forced_records, [](uint8_t value) { return value <= 1; })) {
        throw std::invalid_argument("record bytes must be Boolean");
    }

    std::vector<double> log_probabilities(num_records);
    Executor executor(plan);
    for (size_t record = 0; record < num_records; ++record) {
        const ReplayResult replay =
            executor.replay_shot(forced_records.subspan(record * stride, stride));
        log_probabilities[record] =
            replay.reachable ? replay.log_probability : std::numeric_limits<double>::lowest();
    }
    return log_probabilities;
}

}  // namespace clifft::sampling
