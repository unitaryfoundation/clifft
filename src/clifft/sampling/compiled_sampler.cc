#include "clifft/sampling/compiled_sampler.h"

#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/executor.h"
#include "clifft/util/shot_parallel.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace clifft::sampling {

namespace {

inline constexpr size_t kTargetStreamBufferBytes = 8 * 1024 * 1024;

size_t checked_output_size(uint32_t shots, size_t stride) {
    if (stride != 0 && shots > std::numeric_limits<size_t>::max() / stride) {
        throw std::length_error("compiled sampler output size exceeds size_t range");
    }
    return static_cast<size_t>(shots) * stride;
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
    assert(false && "compiled sampler bit source must be valid");
    return 0;
}

bool source_available(SamplingOutputSelection available, SamplingBitSource source) noexcept {
    switch (source) {
        case SamplingBitSource::Measurements:
            return available.measurements;
        case SamplingBitSource::Detectors:
            return available.detectors;
        case SamplingBitSource::Observables:
            return available.observables;
    }
    return false;
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
    assert(false && "compiled sampler bit source must be valid");
    return {};
}

void validate_output(const ExecutablePlan& plan, SamplingOutputSelection available, uint32_t shots,
                     SamplingOutputBuffer output) {
    for (const SamplingBitOutput& destination : output.bits) {
        if (!source_available(available, destination.source)) {
            throw std::invalid_argument("compiled sampler output source was not retained");
        }
        const size_t columns = bit_source_width(plan, destination.source);
        if (destination.column_offset > std::numeric_limits<size_t>::max() - columns) {
            throw std::length_error("compiled sampler output column range exceeds size_t");
        }
        const size_t end_column = destination.column_offset + columns;
        const size_t minimum_stride =
            destination.packing == SamplingBitPacking::BitPacked
                ? end_column / 8 + static_cast<size_t>((end_column & 7) != 0)
                : end_column;
        if (destination.row_stride < minimum_stride) {
            throw std::invalid_argument("compiled sampler bit output row stride is too small");
        }
        if (destination.data.size() < checked_output_size(shots, destination.row_stride)) {
            throw std::invalid_argument("compiled sampler bit output buffer is too small");
        }
    }
    if (!output.exp_vals.empty()) {
        if (!available.exp_vals) {
            throw std::invalid_argument("compiled sampler expectation output was not retained");
        }
        if (output.exp_val_row_stride < plan.num_exp_vals()) {
            throw std::invalid_argument(
                "compiled sampler expectation output row stride is too small");
        }
        if (output.exp_vals.size() < checked_output_size(shots, output.exp_val_row_stride)) {
            throw std::invalid_argument("compiled sampler expectation output buffer is too small");
        }
    }
}

void clear_bit_outputs(uint32_t shots, SamplingOutputBuffer output) noexcept {
    for (const SamplingBitOutput& destination : output.bits) {
        std::ranges::fill(
            destination.data.first(static_cast<size_t>(shots) * destination.row_stride),
            uint8_t{0});
    }
}

void reseed_executor(Executor& executor, const SeedRoot& root, uint32_t shot) noexcept {
    const std::array<uint64_t, 4> words = derive_state(root, shot, kSamplingExecutorDomain);
    executor.reseed_full(words[0], words[1], words[2], words[3]);
}

void write_scalar_outputs(SamplingOutputBuffer output, const Executor& executor,
                          uint32_t shot) noexcept {
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
}

void write_batch_outputs(SamplingOutputBuffer output, const BatchExecutor& executor,
                         uint32_t first_shot, uint32_t shots, const ExecutablePlan& plan) noexcept {
    assert(executor.surviving_shots() == shots && "compiled fixed-row batch must retain all shots");
    for (const SamplingBitOutput& destination : output.bits) {
        executor.write_bit_rows(
            destination.source, destination.packing,
            destination.data.subspan(static_cast<size_t>(first_shot) * destination.row_stride),
            destination.row_stride, destination.column_offset);
    }
    if (!output.exp_vals.empty()) {
        for (uint32_t lane = 0; lane < shots; ++lane) {
            double* row = output.exp_vals.data() +
                          static_cast<size_t>(first_shot + lane) * output.exp_val_row_stride;
            for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
                row[exp_val] = executor.exp_val(lane, exp_val);
            }
        }
    }
}

SeedRoot call_seed_root(const SeedRoot& sampler_root, uint64_t call) noexcept {
    const std::array<uint64_t, 4> words =
        derive_state(sampler_root, call, kCompiledSamplerCallDomain);
    return {{words[0], words[1], words[2], words[3]}};
}

size_t checked_add(size_t left, size_t right, const char* message) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::length_error(message);
    }
    return left + right;
}

struct PreparedFileOutput {
    std::ostream* output = nullptr;
    SamplingFileFormat format = SamplingFileFormat::Format01;
    std::span<const SamplingBitSource> sources;
    size_t columns = 0;
    size_t row_stride = 0;
    std::vector<uint8_t> rows;
    std::unique_ptr<SamplingRowWriter> writer;
};

}  // namespace

struct CompiledSampler::Worker {
    Worker(const ExecutablePlan& plan, uint32_t lane_capacity,
           SamplingOutputSelection available_outputs)
        : scalar(lane_capacity == 1 ? std::make_unique<Executor>(plan, 0) : nullptr),
          batch(lane_capacity > 1 ? std::make_unique<BatchExecutor>(
                                        plan, lane_capacity, BatchOutputMode::Rows,
                                        BatchSamplingMode::Ordinary, available_outputs)
                                  : nullptr) {}

    std::unique_ptr<Executor> scalar;
    std::unique_ptr<BatchExecutor> batch;
};

CompiledSampler::CompiledSampler(std::shared_ptr<const ExecutablePlan> plan,
                                 SamplingOutputSelection available_outputs,
                                 std::optional<uint64_t> seed, uint32_t threads,
                                 std::optional<uint32_t> batch_size)
    : plan_(std::move(plan)),
      available_outputs_(available_outputs),
      seed_root_(seed.has_value() ? seed_root_from_seed(*seed) : make_seed_root(1, std::nullopt)) {
    if (plan_ == nullptr) {
        throw std::invalid_argument("compiled sampler requires an executable plan");
    }
    if (plan_->has_instruments()) {
        throw std::invalid_argument("compiled samplers do not support instrument traps");
    }
    if (plan_->has_postselection()) {
        throw std::invalid_argument("fixed-row compiled samplers do not support postselection");
    }
    if (plan_->num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "compiled samplers require a distribution for every presampled symbol");
    }

    const uint32_t thread_budget = resolve_thread_budget(threads);
    const uint64_t planning_shots64 =
        static_cast<uint64_t>(kDefaultMaxAutoBatchShots) * thread_budget;
    const uint32_t planning_shots = static_cast<uint32_t>(std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(), std::max<uint64_t>(1, planning_shots64)));
    const BatchExecutionPolicy policy = resolve_batch_execution_policy(
        *plan_, planning_shots, thread_budget, 1, BatchOutputMode::Rows, batch_size,
        BatchSamplingMode::Ordinary, 0, available_outputs_);
    lane_capacity_ = policy.lane_capacity;
    const uint32_t worker_count =
        lane_capacity_ > 1 ? policy.worker_count : std::max(uint32_t{1}, thread_budget);
    workers_.reserve(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        workers_.push_back(std::make_unique<Worker>(*plan_, lane_capacity_, available_outputs_));
    }
}

CompiledSampler::~CompiledSampler() = default;

void CompiledSampler::execute_rows(const SeedRoot& root, uint32_t first_root_shot, uint32_t shots,
                                   SamplingOutputBuffer output) {
    clear_bit_outputs(shots, output);
    if (shots == 0) {
        return;
    }

    const uint32_t requested_workers =
        std::min<uint32_t>(static_cast<uint32_t>(workers_.size()), shots);
    if (lane_capacity_ > 1) {
        (void)run_shot_ranges(
            shots, requested_workers, [&](uint32_t worker) { return workers_[worker].get(); },
            [&](Worker* worker, ShotRange range) noexcept {
                for (uint32_t first_shot = range.begin; first_shot < range.end;) {
                    const uint32_t batch = std::min(lane_capacity_, range.end - first_shot);
                    worker->batch->run_batch(root, first_root_shot + first_shot, batch);
                    write_batch_outputs(output, *worker->batch, first_shot, batch, *plan_);
                    first_shot += batch;
                }
            },
            lane_capacity_);
    } else {
        (void)run_shot_ranges(
            shots, requested_workers, [&](uint32_t worker) { return workers_[worker].get(); },
            [&](Worker* worker, ShotRange range) noexcept {
                for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                    reseed_executor(*worker->scalar, root, first_root_shot + shot);
                    worker->scalar->run_shot();
                    write_scalar_outputs(output, *worker->scalar, shot);
                }
            });
    }
}

void CompiledSampler::sample(uint32_t shots, SamplingOutputBuffer output) {
    validate_output(*plan_, available_outputs_, shots, output);
    std::lock_guard lock(mutex_);
    if (shots == 0) {
        clear_bit_outputs(shots, output);
        return;
    }

    const SeedRoot root = call_seed_root(seed_root_, calls_completed_);
    execute_rows(root, 0, shots, output);
    ++calls_completed_;
}

void CompiledSampler::sample_write(uint32_t shots, std::span<const SamplingFileOutput> outputs) {
    std::vector<PreparedFileOutput> files;
    files.reserve(outputs.size());
    size_t bytes_per_shot = 0;
    for (const SamplingFileOutput& requested : outputs) {
        if (requested.output == nullptr) {
            throw std::invalid_argument("compiled sampler file output requires a stream");
        }
        size_t columns = 0;
        for (SamplingBitSource source : requested.sources) {
            if (!source_available(available_outputs_, source)) {
                throw std::invalid_argument("compiled sampler file source was not retained");
            }
            columns = checked_add(columns, bit_source_width(*plan_, source),
                                  "compiled sampler file row exceeds size_t");
        }
        const size_t row_stride = columns / 8 + static_cast<size_t>((columns & 7) != 0);
        bytes_per_shot =
            checked_add(bytes_per_shot, row_stride, "compiled sampler file buffers exceed size_t");
        files.push_back({.output = requested.output,
                         .format = requested.format,
                         .sources = requested.sources,
                         .columns = columns,
                         .row_stride = row_stride});
    }

    uint32_t buffer_shots = std::min(shots, kDefaultMaxAutoBatchShots);
    if (buffer_shots != 0 && bytes_per_shot != 0) {
        const size_t byte_limited = std::max<size_t>(1, kTargetStreamBufferBytes / bytes_per_shot);
        buffer_shots = static_cast<uint32_t>(
            std::min<size_t>(buffer_shots, std::min<size_t>(byte_limited, UINT32_MAX)));
    }
    buffer_shots = std::max(uint32_t{1}, buffer_shots);

    std::vector<SamplingBitOutput> destinations;
    size_t destination_count = 0;
    for (const PreparedFileOutput& file : files) {
        destination_count = checked_add(destination_count, file.sources.size(),
                                        "compiled sampler file destination count exceeds size_t");
    }
    destinations.reserve(destination_count);
    for (PreparedFileOutput& file : files) {
        file.rows.resize(checked_output_size(buffer_shots, file.row_stride));
        file.writer = std::make_unique<SamplingRowWriter>(*file.output, file.format, file.columns,
                                                          buffer_shots);
        size_t column_offset = 0;
        for (SamplingBitSource source : file.sources) {
            destinations.push_back({.source = source,
                                    .packing = SamplingBitPacking::BitPacked,
                                    .data = file.rows,
                                    .row_stride = file.row_stride,
                                    .column_offset = column_offset});
            column_offset += bit_source_width(*plan_, source);
        }
    }

    std::lock_guard lock(mutex_);
    if (shots == 0) {
        return;
    }
    const SeedRoot root = call_seed_root(seed_root_, calls_completed_);
    for (uint32_t first_shot = 0; first_shot < shots;) {
        const uint32_t batch = std::min(buffer_shots, shots - first_shot);
        execute_rows(root, first_shot, batch, {.bits = destinations});
        for (PreparedFileOutput& file : files) {
            file.writer->write_packed_rows(file.rows, batch, file.row_stride);
        }
        first_shot += batch;
    }
    ++calls_completed_;
}

uint64_t CompiledSampler::calls_completed() const {
    std::lock_guard lock(mutex_);
    return calls_completed_;
}

}  // namespace clifft::sampling
