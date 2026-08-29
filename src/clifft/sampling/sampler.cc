#include "clifft/sampling/sampler.h"

#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/policy.h"
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

void validate_fixed_sampling_plan(const ExecutablePlan& plan) {
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
    BatchSamplingWorker(const ExecutablePlan& plan, uint32_t capacity,
                        SamplingOutputSelection outputs)
        : executor(plan, capacity, BatchOutputMode::Rows, BatchSamplingMode::Ordinary, outputs) {}

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
    BatchSurvivorWorker(const ExecutablePlan& plan, uint32_t capacity,
                        SamplingOutputSelection outputs)
        : executor(plan, capacity,
                   outputs.any() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors,
                   BatchSamplingMode::Ordinary, outputs),
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
void allocate_row_outputs(Output& output, uint32_t shots, const ExecutablePlan& plan,
                          SamplingOutputSelection outputs) {
    if (outputs.measurements) {
        output.measurements.resize(checked_output_size(shots, plan.num_visible_records()));
    }
    if (outputs.detectors) {
        output.detectors.resize(checked_output_size(shots, plan.num_detectors()));
    }
    if (outputs.observables) {
        output.observables.resize(checked_output_size(shots, plan.num_observables()));
    }
    if (outputs.exp_vals) {
        output.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    }
}

size_t bit_source_width(const ExecutablePlan& plan, SamplingBitSource source) noexcept {
    switch (source) {
        case SamplingBitSource::Measurements:
            return plan.num_visible_records();
        case SamplingBitSource::Detectors:
            return plan.num_detectors();
        case SamplingBitSource::Observables:
            return plan.num_observables();
    }
    assert(false && "sampling bit source must be valid");
    return 0;
}

std::span<const uint8_t> scalar_bit_source(const Executor& executor,
                                           SamplingBitSource source) noexcept {
    switch (source) {
        case SamplingBitSource::Measurements:
            return executor.visible_records();
        case SamplingBitSource::Detectors:
            return executor.detectors();
        case SamplingBitSource::Observables:
            return executor.observables();
    }
    assert(false && "sampling bit source must be valid");
    return {};
}

SamplingOutputSelection validate_output_buffer(const ExecutablePlan& plan, uint32_t shots,
                                               SamplingOutputBuffer output) {
    SamplingOutputSelection selection;
    for (const SamplingBitOutput& destination : output.bits) {
        const size_t columns = bit_source_width(plan, destination.source);
        if (destination.column_offset > std::numeric_limits<size_t>::max() - columns) {
            throw std::length_error("sampling output column range exceeds size_t");
        }
        const size_t end_column = destination.column_offset + columns;
        const size_t minimum_stride =
            destination.packing == SamplingBitPacking::BitPacked
                ? end_column / 8 + static_cast<size_t>((end_column & 7) != 0)
                : end_column;
        if (destination.row_stride < minimum_stride) {
            throw std::invalid_argument("sampling bit output row stride is too small");
        }
        const size_t required_size = checked_output_size(shots, destination.row_stride);
        if (destination.data.size() < required_size) {
            throw std::invalid_argument("sampling bit output buffer is too small");
        }
        switch (destination.source) {
            case SamplingBitSource::Measurements:
                selection.measurements = true;
                break;
            case SamplingBitSource::Detectors:
                selection.detectors = true;
                break;
            case SamplingBitSource::Observables:
                selection.observables = true;
                break;
        }
    }
    if (!output.exp_vals.empty()) {
        if (output.exp_val_row_stride < plan.num_exp_vals()) {
            throw std::invalid_argument("sampling expectation output row stride is too small");
        }
        const size_t required_size = checked_output_size(shots, output.exp_val_row_stride);
        if (output.exp_vals.size() < required_size) {
            throw std::invalid_argument("sampling expectation output buffer is too small");
        }
        selection.exp_vals = true;
    }
    return selection;
}

void clear_bit_outputs(uint32_t shots, SamplingOutputBuffer output) noexcept {
    for (const SamplingBitOutput& destination : output.bits) {
        const size_t size = static_cast<size_t>(shots) * destination.row_stride;
        std::ranges::fill(destination.data.first(size), uint8_t{0});
    }
}

void write_scalar_outputs(SamplingOutputBuffer output, const Executor& executor, uint32_t shot,
                          const ExecutablePlan& plan) noexcept {
    for (const SamplingBitOutput& destination : output.bits) {
        const std::span<const uint8_t> source = scalar_bit_source(executor, destination.source);
        if (source.empty()) {
            continue;
        }
        uint8_t* row = destination.data.data() + static_cast<size_t>(shot) * destination.row_stride;
        if (destination.packing == SamplingBitPacking::Unpacked) {
            std::ranges::copy(source, row + destination.column_offset);
            continue;
        }
        for (size_t column = 0; column < source.size(); ++column) {
            row[(destination.column_offset + column) >> 3] |=
                source[column] << ((destination.column_offset + column) & 7);
        }
    }
    if (!output.exp_vals.empty()) {
        std::ranges::copy(
            executor.exp_vals(),
            output.exp_vals.begin() + static_cast<size_t>(shot) * output.exp_val_row_stride);
    }
    (void)plan;
}

void write_batch_outputs(SamplingOutputBuffer output, const BatchExecutor& executor, uint32_t lanes,
                         const ExecutablePlan& plan) noexcept {
    assert(executor.surviving_shots() == lanes && "fixed-row batch must retain every shot");
    if (lanes == 0) {
        return;
    }
    const uint32_t first_shot = executor.shot_index(0);
    for (uint32_t lane = 1; lane < lanes; ++lane) {
        assert(executor.shot_index(lane) == first_shot + lane &&
               "fixed-row batch output lanes must preserve shot order");
    }
    for (const SamplingBitOutput& destination : output.bits) {
        const size_t offset = static_cast<size_t>(first_shot) * destination.row_stride;
        executor.write_bit_rows(destination.source, destination.packing,
                                destination.data.subspan(offset), destination.row_stride,
                                destination.column_offset);
    }
    if (!output.exp_vals.empty()) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            double* row = output.exp_vals.data() +
                          static_cast<size_t>(first_shot + lane) * output.exp_val_row_stride;
            for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
                row[exp_val] = executor.exp_val(lane, exp_val);
            }
        }
    }
}

template <typename MakeWorker, typename RunShot>
void sample_fixed_rows_into(const ExecutablePlan& plan, uint32_t shots, SamplingOutputBuffer output,
                            std::optional<uint64_t> seed, ThreadLayout thread_layout,
                            MakeWorker&& make_worker, RunShot&& run_shot) {
    if (shots == 0) {
        return;
    }
    const SeedRoot root = make_seed_root(shots, seed);
    (void)run_shot_ranges(shots, thread_layout.shot_workers, std::forward<MakeWorker>(make_worker),
                          [&](auto& worker_handle, ShotRange range) {
                              auto& worker = *worker_handle;
                              Executor& executor = worker.executor;
                              for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                                  reseed_executor_for_shot(executor, root, shot);
                                  run_shot(worker);
                                  write_scalar_outputs(output, executor, shot, plan);
                              }
                          });
}

template <typename MakeWorker, typename RunBatch>
void sample_fixed_batches_into(const ExecutablePlan& plan, uint32_t shots,
                               SamplingOutputBuffer output, std::optional<uint64_t> seed,
                               BatchExecutionPolicy batch_policy, MakeWorker&& make_worker,
                               RunBatch&& run_batch) {
    if (shots == 0) {
        return;
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
                write_batch_outputs(output, executor, batch, plan);
                offset += batch;
            }
        },
        batch_capacity);
}

template <typename Output>
void copy_scalar_row(Output& output, const Executor& executor, uint32_t shot,
                     const ExecutablePlan& plan, SamplingOutputSelection outputs) noexcept {
    if (outputs.measurements) {
        std::ranges::copy(
            executor.visible_records(),
            output.measurements.begin() + static_cast<size_t>(shot) * plan.num_visible_records());
    }
    if (outputs.detectors) {
        std::ranges::copy(
            executor.detectors(),
            output.detectors.begin() + static_cast<size_t>(shot) * plan.num_detectors());
    }
    if (outputs.observables) {
        std::ranges::copy(
            executor.observables(),
            output.observables.begin() + static_cast<size_t>(shot) * plan.num_observables());
    }
    if (outputs.exp_vals) {
        std::ranges::copy(executor.exp_vals(), output.exp_vals.begin() +
                                                   static_cast<size_t>(shot) * plan.num_exp_vals());
    }
}

template <typename Output>
void copy_batch_lane(Output& output, const BatchExecutor& executor, uint32_t lane, uint32_t shot,
                     const ExecutablePlan& plan, SamplingOutputSelection outputs) noexcept {
    if (outputs.measurements) {
        for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
            output.measurements[static_cast<size_t>(shot) * plan.num_visible_records() + record] =
                static_cast<uint8_t>(executor.measurement(lane, record));
        }
    }
    if (outputs.detectors) {
        for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
            output.detectors[static_cast<size_t>(shot) * plan.num_detectors() + detector] =
                static_cast<uint8_t>(executor.detector(lane, detector));
        }
    }
    if (outputs.observables) {
        for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
            output.observables[static_cast<size_t>(shot) * plan.num_observables() + observable] =
                static_cast<uint8_t>(executor.observable(lane, observable));
        }
    }
    if (outputs.exp_vals) {
        for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
            output.exp_vals[static_cast<size_t>(shot) * plan.num_exp_vals() + exp_val] =
                executor.exp_val(lane, exp_val);
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
                                 SamplingOutputSelection outputs, std::optional<uint64_t> seed,
                                 ThreadLayout thread_layout, MakeWorker&& make_worker,
                                 RunShot&& run_shot) {
    SamplingResult result;
    allocate_row_outputs(result, shots, plan, outputs);
    if (shots == 0) {
        return result;
    }

    const SeedRoot root = make_seed_root(shots, seed);
    (void)run_shot_ranges(shots, thread_layout.shot_workers, std::forward<MakeWorker>(make_worker),
                          [&](auto& worker_handle, ShotRange range) {
                              auto& worker = *worker_handle;
                              Executor& executor = worker.executor;
                              for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                                  reseed_executor_for_shot(executor, root, shot);
                                  run_shot(worker);
                                  copy_scalar_row(result, executor, shot, plan, outputs);
                              }
                          });
    return result;
}

template <typename MakeWorker, typename RunBatch>
SamplingResult sample_fixed_batches(const ExecutablePlan& plan, uint32_t shots,
                                    SamplingOutputSelection outputs, std::optional<uint64_t> seed,
                                    BatchExecutionPolicy batch_policy, MakeWorker&& make_worker,
                                    RunBatch&& run_batch) {
    SamplingResult result;
    allocate_row_outputs(result, shots, plan, outputs);
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
                    copy_batch_lane(result, executor, lane, shot, plan, outputs);
                }
                offset += batch;
            }
        },
        batch_capacity);
    return result;
}

template <typename MakeWorker, typename RunShot>
SamplingSurvivorResult sample_surviving_rows(const ExecutablePlan& plan, uint32_t shots,
                                             SamplingOutputSelection outputs,
                                             std::optional<uint64_t> seed,
                                             ThreadLayout thread_layout, MakeWorker&& make_worker,
                                             RunShot&& run_shot) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    allocate_row_outputs(result, shots, plan, outputs);
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    std::vector<uint8_t> survived(outputs.any() ? shots : 0, 0);
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
                if (outputs.any()) {
                    survived[shot] = 1;
                    copy_scalar_row(result, executor, shot, plan, outputs);
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
    if (outputs.measurements) {
        compact_survivor_rows(result.measurements, survived, plan.num_visible_records(),
                              result.passed_shots);
    }
    if (outputs.detectors) {
        compact_survivor_rows(result.detectors, survived, plan.num_detectors(),
                              result.passed_shots);
    }
    if (outputs.observables) {
        compact_survivor_rows(result.observables, survived, plan.num_observables(),
                              result.passed_shots);
    }
    if (outputs.exp_vals) {
        compact_survivor_rows(result.exp_vals, survived, plan.num_exp_vals(), result.passed_shots);
    }
    return result;
}

template <typename MakeWorker, typename RunBatch>
SamplingSurvivorResult sample_surviving_batches(const ExecutablePlan& plan, uint32_t shots,
                                                SamplingOutputSelection outputs,
                                                std::optional<uint64_t> seed,
                                                BatchExecutionPolicy batch_policy,
                                                MakeWorker&& make_worker, RunBatch&& run_batch) {
    SamplingSurvivorResult result;
    result.total_shots = shots;
    allocate_row_outputs(result, shots, plan, outputs);
    if (shots == 0) {
        return result;
    }
    result.observable_ones.resize(plan.num_observables(), 0);
    std::vector<uint8_t> survived(outputs.any() ? shots : 0, 0);
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
                if (!outputs.any()) {
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
                    copy_batch_lane(result, executor, lane, shot, plan, outputs);
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
    if (outputs.measurements) {
        compact_survivor_rows(result.measurements, survived, plan.num_visible_records(),
                              result.passed_shots);
    }
    if (outputs.detectors) {
        compact_survivor_rows(result.detectors, survived, plan.num_detectors(),
                              result.passed_shots);
    }
    if (outputs.observables) {
        compact_survivor_rows(result.observables, survived, plan.num_observables(),
                              result.passed_shots);
    }
    if (outputs.exp_vals) {
        compact_survivor_rows(result.exp_vals, survived, plan.num_exp_vals(), result.passed_shots);
    }
    return result;
}

void run_fixed_sampling_into(const ExecutablePlan& plan, uint32_t shots,
                             SamplingOutputBuffer output, SamplingOutputSelection outputs,
                             std::optional<uint64_t> seed, uint32_t threads,
                             std::optional<ThreadLayout> thread_layout,
                             std::optional<uint32_t> batch_size) {
    const ThreadLayout resolved = resolve_thread_layout(plan, shots, threads, thread_layout);
    const BatchExecutionPolicy batch_policy = resolve_batch_execution_policy(
        plan, shots, resolved.shot_workers, resolved.intra_shot_workers, BatchOutputMode::Rows,
        batch_size, BatchSamplingMode::Ordinary, 0, outputs);
    if constexpr (kPackedBatchExecutionAvailable) {
        if (batch_policy.lane_capacity > 1) {
            sample_fixed_batches_into(
                plan, shots, output, seed, batch_policy,
                [&](uint32_t) {
                    return std::make_unique<BatchSamplingWorker>(plan, batch_policy.lane_capacity,
                                                                 outputs);
                },
                [](BatchSamplingWorker& worker, const SeedRoot& root, uint32_t first_shot,
                   uint32_t batch) noexcept {
                    worker.executor.run_batch(root, first_shot, batch);
                });
            return;
        }
    }
    sample_fixed_rows_into(
        plan, shots, output, seed, resolved,
        [&](uint32_t) {
            return std::make_unique<SamplingWorker>(plan, resolved.intra_shot_workers,
                                                    resolved.intra_shot_min_active_width);
        },
        [](SamplingWorker& worker) noexcept { worker.executor.run_shot(); });
}

}  // namespace

SamplingResult sample(const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed,
                      uint32_t threads, std::optional<ThreadLayout> thread_layout,
                      std::optional<uint32_t> batch_size) {
    return sample_selected(plan, shots, SamplingOutputSelection::all(), seed, threads,
                           thread_layout, batch_size);
}

SamplingResult sample_selected(const ExecutablePlan& plan, uint32_t shots,
                               SamplingOutputSelection outputs, std::optional<uint64_t> seed,
                               uint32_t threads, std::optional<ThreadLayout> thread_layout,
                               std::optional<uint32_t> batch_size) {
    validate_fixed_sampling_plan(plan);
    SamplingResult result;
    allocate_row_outputs(result, shots, plan, outputs);
    std::array<SamplingBitOutput, 3> bit_outputs;
    size_t output_count = 0;
    const auto add_output = [&](bool selected, SamplingBitSource source, std::vector<uint8_t>& data,
                                size_t stride) {
        if (selected) {
            bit_outputs[output_count++] = {
                .source = source,
                .data = data,
                .row_stride = stride,
            };
        }
    };
    add_output(outputs.measurements, SamplingBitSource::Measurements, result.measurements,
               plan.num_visible_records());
    add_output(outputs.detectors, SamplingBitSource::Detectors, result.detectors,
               plan.num_detectors());
    add_output(outputs.observables, SamplingBitSource::Observables, result.observables,
               plan.num_observables());
    run_fixed_sampling_into(
        plan, shots,
        {.bits = std::span<const SamplingBitOutput>(bit_outputs).first(output_count),
         .exp_vals = result.exp_vals,
         .exp_val_row_stride = plan.num_exp_vals()},
        outputs, seed, threads, thread_layout, batch_size);
    return result;
}

void sample_into(const ExecutablePlan& plan, uint32_t shots, SamplingOutputBuffer output,
                 std::optional<uint64_t> seed, uint32_t threads,
                 std::optional<ThreadLayout> thread_layout, std::optional<uint32_t> batch_size) {
    validate_fixed_sampling_plan(plan);
    const SamplingOutputSelection outputs = validate_output_buffer(plan, shots, output);
    clear_bit_outputs(shots, output);
    run_fixed_sampling_into(plan, shots, output, outputs, seed, threads, thread_layout, batch_size);
}

PackedSamplingResult sample_packed_selected(const ExecutablePlan& plan, uint32_t shots,
                                            SamplingOutputSelection outputs,
                                            std::optional<uint64_t> seed, uint32_t threads,
                                            std::optional<ThreadLayout> thread_layout,
                                            std::optional<uint32_t> batch_size) {
    validate_fixed_sampling_plan(plan);
    PackedSamplingResult result;
    const auto packed_stride = [](size_t columns) noexcept {
        return columns / 8 + static_cast<size_t>((columns & 7) != 0);
    };
    const size_t measurement_stride = packed_stride(plan.num_visible_records());
    const size_t detector_stride = packed_stride(plan.num_detectors());
    const size_t observable_stride = packed_stride(plan.num_observables());
    if (outputs.measurements) {
        result.measurements.resize(checked_output_size(shots, measurement_stride));
    }
    if (outputs.detectors) {
        result.detectors.resize(checked_output_size(shots, detector_stride));
    }
    if (outputs.observables) {
        result.observables.resize(checked_output_size(shots, observable_stride));
    }
    if (outputs.exp_vals) {
        result.exp_vals.resize(checked_output_size(shots, plan.num_exp_vals()));
    }
    std::array<SamplingBitOutput, 3> bit_outputs;
    size_t output_count = 0;
    const auto add_output = [&](bool selected, SamplingBitSource source, std::vector<uint8_t>& data,
                                size_t stride) {
        if (selected) {
            bit_outputs[output_count++] = {
                .source = source,
                .packing = SamplingBitPacking::BitPacked,
                .data = data,
                .row_stride = stride,
            };
        }
    };
    add_output(outputs.measurements, SamplingBitSource::Measurements, result.measurements,
               measurement_stride);
    add_output(outputs.detectors, SamplingBitSource::Detectors, result.detectors, detector_stride);
    add_output(outputs.observables, SamplingBitSource::Observables, result.observables,
               observable_stride);
    run_fixed_sampling_into(
        plan, shots,
        {.bits = std::span<const SamplingBitOutput>(bit_outputs).first(output_count),
         .exp_vals = result.exp_vals,
         .exp_val_row_stride = plan.num_exp_vals()},
        outputs, seed, threads, thread_layout, batch_size);
    return result;
}

std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed, uint32_t threads,
                                    std::optional<ThreadLayout> thread_layout,
                                    std::optional<uint32_t> batch_size) {
    return sample_selected(plan, shots, {.measurements = true}, seed, threads, thread_layout,
                           batch_size)
        .measurements;
}

SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                        std::optional<uint64_t> seed, bool keep_records,
                                        uint32_t threads, std::optional<ThreadLayout> thread_layout,
                                        std::optional<uint32_t> batch_size) {
    const SamplingOutputSelection outputs =
        keep_records ? SamplingOutputSelection::all() : SamplingOutputSelection{};
    return sample_survivors_selected(plan, shots, outputs, seed, threads, thread_layout,
                                     batch_size);
}

SamplingSurvivorResult sample_survivors_selected(const ExecutablePlan& plan, uint32_t shots,
                                                 SamplingOutputSelection outputs,
                                                 std::optional<uint64_t> seed, uint32_t threads,
                                                 std::optional<ThreadLayout> thread_layout,
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
    const BatchOutputMode output_mode =
        outputs.any() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    const BatchExecutionPolicy batch_policy = resolve_batch_execution_policy(
        plan, shots, resolved.shot_workers, resolved.intra_shot_workers, output_mode, batch_size,
        BatchSamplingMode::Ordinary, survivor_worker_bytes(plan), outputs);
    if constexpr (kPackedBatchExecutionAvailable) {
        if (batch_policy.lane_capacity > 1) {
            return sample_surviving_batches(
                plan, shots, outputs, seed, batch_policy,
                [&](uint32_t) {
                    return std::make_unique<BatchSurvivorWorker>(plan, batch_policy.lane_capacity,
                                                                 outputs);
                },
                [](BatchSurvivorWorker& worker, const SeedRoot& root, uint32_t first_shot,
                   uint32_t batch) noexcept {
                    worker.executor.run_batch(root, first_shot, batch);
                });
        }
    }
    return sample_surviving_rows(
        plan, shots, outputs, seed, resolved,
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
    if (shots == 0) {
        (void)resolve_batch_execution_policy(plan, shots, resolved.shot_workers,
                                             resolved.intra_shot_workers, BatchOutputMode::Rows,
                                             batch_size, BatchSamplingMode::FixedFaults);
        return sample_fixed_rows(
            plan, shots, SamplingOutputSelection::all(), seed, resolved,
            [&](uint32_t) {
                return std::make_unique<SamplingWorker>(plan, resolved.intra_shot_workers,
                                                        resolved.intra_shot_min_active_width);
            },
            [](SamplingWorker& worker) noexcept { worker.executor.run_shot(); });
    }
    const auto fault_distribution =
        std::make_shared<const KFaultDistribution>(plan.noise_site_probabilities(), k);
    const BatchExecutionPolicy batch_policy = resolve_batch_execution_policy(
        plan, shots, resolved.shot_workers, resolved.intra_shot_workers, BatchOutputMode::Rows,
        batch_size, BatchSamplingMode::FixedFaults, fault_distribution->worker_scratch_bytes());
    if constexpr (kPackedBatchExecutionAvailable) {
        if (batch_policy.lane_capacity > 1) {
            return sample_fixed_batches(
                plan, shots, SamplingOutputSelection::all(), seed, batch_policy,
                [&](uint32_t) {
                    return std::make_unique<ConditionedBatchSamplingWorker>(
                        plan, fault_distribution, batch_policy.lane_capacity);
                },
                [](ConditionedBatchSamplingWorker& worker, const SeedRoot& root,
                   uint32_t first_shot, uint32_t batch) noexcept {
                    worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
                });
        }
    }
    return sample_fixed_rows(
        plan, shots, SamplingOutputSelection::all(), seed, resolved,
        [&](uint32_t) {
            return std::make_unique<ConditionedSamplingWorker>(
                plan, fault_distribution, resolved.intra_shot_workers,
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
    const SamplingOutputSelection outputs =
        keep_records ? SamplingOutputSelection::all() : SamplingOutputSelection{};
    const BatchOutputMode output_mode =
        outputs.any() ? BatchOutputMode::Rows : BatchOutputMode::AggregateSurvivors;
    if (shots == 0) {
        (void)resolve_batch_execution_policy(plan, shots, resolved.shot_workers,
                                             resolved.intra_shot_workers, output_mode, batch_size,
                                             BatchSamplingMode::FixedFaults);
        return sample_surviving_rows(
            plan, shots, outputs, seed, resolved,
            [&](uint32_t) {
                return std::make_unique<SurvivorWorker>(plan, resolved.intra_shot_workers,
                                                        resolved.intra_shot_min_active_width);
            },
            [](SurvivorWorker& worker) noexcept { worker.executor.run_shot(); });
    }
    const auto fault_distribution =
        std::make_shared<const KFaultDistribution>(plan.noise_site_probabilities(), k);
    const BatchExecutionPolicy batch_policy = resolve_batch_execution_policy(
        plan, shots, resolved.shot_workers, resolved.intra_shot_workers, output_mode, batch_size,
        BatchSamplingMode::FixedFaults,
        fault_distribution->worker_scratch_bytes() + survivor_worker_bytes(plan));
    if constexpr (kPackedBatchExecutionAvailable) {
        if (batch_policy.lane_capacity > 1) {
            return sample_surviving_batches(
                plan, shots, outputs, seed, batch_policy,
                [&](uint32_t) {
                    return std::make_unique<ConditionedBatchSurvivorWorker>(
                        plan, fault_distribution, batch_policy.lane_capacity, outputs.any());
                },
                [](ConditionedBatchSurvivorWorker& worker, const SeedRoot& root,
                   uint32_t first_shot, uint32_t batch) noexcept {
                    worker.executor.run_batch(root, first_shot, batch, worker.fault_sampler);
                });
        }
    }
    return sample_surviving_rows(
        plan, shots, outputs, seed, resolved,
        [&](uint32_t) {
            return std::make_unique<ConditionedSurvivorWorker>(
                plan, fault_distribution, resolved.intra_shot_workers,
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
