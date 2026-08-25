#include "clifft/sampling/sampler.h"

#include "clifft/sampling/batch_executor.h"
#include "clifft/sampling/executor.h"
#include "clifft/util/fault_sampling.h"
#include "clifft/util/intra_shot_parallel.h"
#include "clifft/util/shot_parallel.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>

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
    ConditionedSamplingWorker(const ExecutablePlan& plan, std::span<const double> probabilities,
                              uint32_t k, uint32_t intra_shot_workers,
                              uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width),
          fault_sampler(probabilities, k) {}

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
    ConditionedSurvivorWorker(const ExecutablePlan& plan, std::span<const double> probabilities,
                              uint32_t k, uint32_t intra_shot_workers,
                              uint32_t intra_shot_min_active_width)
        : executor(plan, 0, intra_shot_workers, intra_shot_min_active_width),
          fault_sampler(probabilities, k),
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
                                   std::span<const double> probabilities, uint32_t k,
                                   uint32_t capacity)
        : executor(plan, capacity), fault_sampler(probabilities, k) {}

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
                                   std::span<const double> probabilities, uint32_t k,
                                   uint32_t capacity, bool keep_records)
        : executor(plan, capacity,
                   keep_records ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors),
          fault_sampler(probabilities, k),
          counts(plan.num_observables()) {}

    BatchExecutor executor;
    KFaultSampler fault_sampler;
    SurvivorCounts counts;
};

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
    auto checked_size = [shots](size_t stride) {
        if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
            throw std::length_error("sampling output size exceeds size_t range");
        }
        return static_cast<size_t>(shots) * stride;
    };

    SamplingResult result;
    result.measurements.resize(checked_size(plan.num_visible_records()));
    result.detectors.resize(checked_size(plan.num_detectors()));
    result.observables.resize(checked_size(plan.num_observables()));
    result.exp_vals.resize(checked_size(plan.num_exp_vals()));
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
                                    std::optional<uint64_t> seed, ThreadLayout thread_layout,
                                    uint32_t batch_capacity, MakeWorker&& make_worker,
                                    RunBatch&& run_batch) {
    auto checked_size = [shots](size_t stride) {
        if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
            throw std::length_error("sampling output size exceeds size_t range");
        }
        return static_cast<size_t>(shots) * stride;
    };

    SamplingResult result;
    result.measurements.resize(checked_size(plan.num_visible_records()));
    result.detectors.resize(checked_size(plan.num_detectors()));
    result.observables.resize(checked_size(plan.num_observables()));
    result.exp_vals.resize(checked_size(plan.num_exp_vals()));
    if (shots == 0) {
        return result;
    }

    const SeedRoot root = make_seed_root(shots, seed);
    const uint32_t batch_workers = std::min<uint32_t>(
        thread_layout.shot_workers,
        static_cast<uint32_t>((static_cast<uint64_t>(shots) + batch_capacity - 1) /
                              batch_capacity));
    (void)run_shot_ranges(
        shots, batch_workers, std::forward<MakeWorker>(make_worker),
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
                    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
                        result.measurements[static_cast<size_t>(shot) * plan.num_visible_records() +
                                            record] =
                            static_cast<uint8_t>(executor.measurement(lane, record));
                    }
                    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
                        result.detectors[static_cast<size_t>(shot) * plan.num_detectors() +
                                         detector] =
                            static_cast<uint8_t>(executor.detector(lane, detector));
                    }
                    for (uint32_t observable = 0; observable < plan.num_observables();
                         ++observable) {
                        result.observables[static_cast<size_t>(shot) * plan.num_observables() +
                                           observable] =
                            static_cast<uint8_t>(executor.observable(lane, observable));
                    }
                    for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
                        result.exp_vals[static_cast<size_t>(shot) * plan.num_exp_vals() + exp_val] =
                            executor.exp_val(lane, exp_val);
                    }
                }
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
        auto checked_reserve = [shots](size_t stride) {
            if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
                throw std::length_error("survivor output size exceeds size_t range");
            }
            return static_cast<size_t>(shots) * stride;
        };
        result.measurements.resize(checked_reserve(plan.num_visible_records()));
        result.detectors.resize(checked_reserve(plan.num_detectors()));
        result.observables.resize(checked_reserve(plan.num_observables()));
        result.exp_vals.resize(checked_reserve(plan.num_exp_vals()));
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
                                                ThreadLayout thread_layout, uint32_t batch_capacity,
                                                MakeWorker&& make_worker, RunBatch&& run_batch) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    if (keep_records) {
        auto checked_reserve = [shots](size_t stride) {
            if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
                throw std::length_error("survivor output size exceeds size_t range");
            }
            return static_cast<size_t>(shots) * stride;
        };
        result.measurements.resize(checked_reserve(plan.num_visible_records()));
        result.detectors.resize(checked_reserve(plan.num_detectors()));
        result.observables.resize(checked_reserve(plan.num_observables()));
        result.exp_vals.resize(checked_reserve(plan.num_exp_vals()));
    }
    std::vector<uint8_t> survived(keep_records ? shots : 0, 0);
    const SeedRoot root = make_seed_root(shots, seed);
    const uint32_t batch_workers = std::min<uint32_t>(
        thread_layout.shot_workers,
        static_cast<uint32_t>((static_cast<uint64_t>(shots) + batch_capacity - 1) /
                              batch_capacity));
    auto workers = run_shot_ranges(
        shots, batch_workers, std::forward<MakeWorker>(make_worker),
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
                    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
                        result.measurements[static_cast<size_t>(shot) * plan.num_visible_records() +
                                            record] =
                            static_cast<uint8_t>(executor.measurement(lane, record));
                    }
                    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
                        result.detectors[static_cast<size_t>(shot) * plan.num_detectors() +
                                         detector] =
                            static_cast<uint8_t>(executor.detector(lane, detector));
                    }
                    for (uint32_t observable = 0; observable < plan.num_observables();
                         ++observable) {
                        result.observables[static_cast<size_t>(shot) * plan.num_observables() +
                                           observable] =
                            static_cast<uint8_t>(executor.observable(lane, observable));
                    }
                    for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
                        result.exp_vals[static_cast<size_t>(shot) * plan.num_exp_vals() + exp_val] =
                            executor.exp_val(lane, exp_val);
                    }
                }
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

}  // namespace

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed,
                      uint32_t threads, std::optional<ThreadLayout> thread_layout,
                      std::optional<uint32_t> batch_size) {
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
    const uint32_t batch_capacity =
        resolve_batch_capacity(plan, shots, resolved.intra_shot_workers, batch_size);
    if (batch_capacity > 1) {
        return sample_fixed_batches(
            plan, shots, seed, resolved, batch_capacity,
            [&](uint32_t) { return std::make_unique<BatchSamplingWorker>(plan, batch_capacity); },
            [](BatchSamplingWorker& worker, const SeedRoot& root, uint32_t first_shot,
               uint32_t batch) noexcept { worker.executor.run_batch(root, first_shot, batch); });
    }
    return sample_fixed_rows(
        plan, shots, seed, resolved,
        [&](uint32_t) {
            return std::make_unique<SamplingWorker>(plan, resolved.intra_shot_workers,
                                                    resolved.intra_shot_min_active_width);
        },
        [](SamplingWorker& worker) noexcept { worker.executor.run_shot(); });
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed, uint32_t threads,
                                    std::optional<ThreadLayout> thread_layout,
                                    std::optional<uint32_t> batch_size) {
    return sample(plan, shots, seed, threads, thread_layout, batch_size).measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records,
                                        uint32_t threads, std::optional<ThreadLayout> thread_layout,
                                        std::optional<uint32_t> batch_size) {
    if (plan.has_instruments()) {
        throw std::invalid_argument(
            "survivor sampling does not support instrument traps; use the trajectory driver");
    }
    if (plan.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "survivor sampling requires a distribution for every presampled symbol");
    }

    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    const uint32_t batch_capacity =
        resolve_batch_capacity(plan, shots, resolved.intra_shot_workers, batch_size);
    if (batch_capacity > 1) {
        return sample_surviving_batches(
            plan, shots, seed, keep_records, resolved, batch_capacity,
            [&](uint32_t) {
                return std::make_unique<BatchSurvivorWorker>(plan, batch_capacity, keep_records);
            },
            [](BatchSurvivorWorker& worker, const SeedRoot& root, uint32_t first_shot,
               uint32_t batch) noexcept { worker.executor.run_batch(root, first_shot, batch); });
    }
    return sample_surviving_rows(
        plan, shots, seed, keep_records, resolved,
        [&](uint32_t) {
            return std::make_unique<SurvivorWorker>(plan, resolved.intra_shot_workers,
                                                    resolved.intra_shot_min_active_width);
        },
        [](SurvivorWorker& worker) noexcept { worker.executor.run_shot(); });
}

SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                        std::optional<uint64_t> seed, uint32_t threads,
                        std::optional<ThreadLayout> thread_layout,
                        std::optional<uint32_t> batch_size) {
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
    const uint32_t batch_capacity =
        resolve_batch_capacity(plan, shots, resolved.intra_shot_workers, batch_size);
    if (shots == 0) {
        return sample_fixed_rows(
            plan, shots, seed, resolved,
            [&](uint32_t) {
                return std::make_unique<SamplingWorker>(plan, resolved.intra_shot_workers,
                                                        resolved.intra_shot_min_active_width);
            },
            [](SamplingWorker& worker) noexcept { worker.executor.run_shot(); });
    }
    const std::vector<double> probabilities = plan.noise_site_probabilities();
    if (batch_capacity > 1) {
        return sample_fixed_batches(
            plan, shots, seed, resolved, batch_capacity,
            [&](uint32_t) {
                return std::make_unique<ConditionedBatchSamplingWorker>(plan, probabilities, k,
                                                                        batch_capacity);
            },
            [](ConditionedBatchSamplingWorker& worker, const SeedRoot& root, uint32_t first_shot,
               uint32_t batch) noexcept {
                worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
            });
    }
    return sample_fixed_rows(
        plan, shots, seed, resolved,
        [&](uint32_t) {
            return std::make_unique<ConditionedSamplingWorker>(
                plan, probabilities, k, resolved.intra_shot_workers,
                resolved.intra_shot_min_active_width);
        },
        [](ConditionedSamplingWorker& worker) noexcept {
            worker.executor.run_shot(worker.fault_sampler);
        });
}

SamplingSurvivorResult sample_k_survivors(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                          std::optional<uint64_t> seed, bool keep_records,
                                          uint32_t threads,
                                          std::optional<ThreadLayout> thread_layout,
                                          std::optional<uint32_t> batch_size) {
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
    const uint32_t batch_capacity =
        resolve_batch_capacity(plan, shots, resolved.intra_shot_workers, batch_size);
    if (shots == 0) {
        return sample_surviving_rows(
            plan, shots, seed, keep_records, resolved,
            [&](uint32_t) {
                return std::make_unique<SurvivorWorker>(plan, resolved.intra_shot_workers,
                                                        resolved.intra_shot_min_active_width);
            },
            [](SurvivorWorker& worker) noexcept { worker.executor.run_shot(); });
    }
    const std::vector<double> probabilities = plan.noise_site_probabilities();
    if (batch_capacity > 1) {
        return sample_surviving_batches(
            plan, shots, seed, keep_records, resolved, batch_capacity,
            [&](uint32_t) {
                return std::make_unique<ConditionedBatchSurvivorWorker>(
                    plan, probabilities, k, batch_capacity, keep_records);
            },
            [](ConditionedBatchSurvivorWorker& worker, const SeedRoot& root, uint32_t first_shot,
               uint32_t batch) noexcept {
                worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
            });
    }
    return sample_surviving_rows(
        plan, shots, seed, keep_records, resolved,
        [&](uint32_t) {
            return std::make_unique<ConditionedSurvivorWorker>(
                plan, probabilities, k, resolved.intra_shot_workers,
                resolved.intra_shot_min_active_width);
        },
        [](ConditionedSurvivorWorker& worker) noexcept {
            worker.executor.run_shot(worker.fault_sampler);
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
