#include "clifft/sampling/batch/policy.h"

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/interleaved_state.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace clifft::sampling {

namespace {

#if !defined(__EMSCRIPTEN__)
[[nodiscard]] uint32_t batch_worker_count(uint32_t shots, uint32_t shot_workers,
                                          uint32_t lane_capacity) noexcept {
    const uint64_t batches = (static_cast<uint64_t>(shots) + lane_capacity - 1) / lane_capacity;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(shot_workers, batches)));
}
#endif

}  // namespace

namespace batch_detail {

BatchWorkerStorageLayout batch_worker_storage_layout(const ExecutablePlan& plan,
                                                     uint32_t lane_capacity,
                                                     BatchOutputMode output_mode,
                                                     BatchSamplingMode sampling_mode) {
    BatchWorkerStorageLayout layout;
    layout.peak_active_width = plan.peak_active_width();
    layout.initial_active_width = plan.initial_active_width();
    layout.lane_capacity = lane_capacity;
    layout.word_capacity = packed_word_count(lane_capacity);
    layout.shot_index_entries = lane_capacity;
    layout.symbol_columns =
        plan.num_batch_noise_carriers() == 0 && plan.num_presampled_symbols() != 0
            ? plan.num_symbols()
            : 0;
    layout.noise_carrier_columns = plan.num_batch_noise_carriers();
    layout.expression_register_columns = plan.num_expression_registers();
    layout.record_columns =
        output_mode == BatchOutputMode::Rows || plan.output_parities_read_records()
            ? static_cast<size_t>(plan.num_visible_records()) + plan.num_hidden_records()
            : 0;
    layout.detector_columns = output_mode == BatchOutputMode::Rows ? plan.num_detectors() : 0;
    layout.observable_columns = plan.num_observables();
    layout.forced_readout_columns =
        sampling_mode == BatchSamplingMode::FixedFaults ? plan.num_readout_noise_sites() : 0;
    layout.exp_value_entries = output_mode == BatchOutputMode::Rows
                                   ? static_cast<uint64_t>(plan.num_exp_vals()) * lane_capacity
                                   : 0;
    layout.live_word_entries = layout.word_capacity;
    layout.scratch_word_entries = layout.word_capacity;
    layout.compaction_source_entries = lane_capacity;
    layout.lane_byte_entries = lane_capacity;
    layout.signed_sine_entries = lane_capacity;
    layout.probability_zero_entries = lane_capacity;
    layout.probability_one_entries = lane_capacity;
    layout.lane_value_entries = lane_capacity;
    return layout;
}

uint64_t batch_worker_storage_bytes(const ExecutablePlan& plan, uint32_t lane_capacity,
                                    BatchOutputMode output_mode, BatchSamplingMode sampling_mode) {
    const BatchWorkerStorageLayout layout =
        batch_worker_storage_layout(plan, lane_capacity, output_mode, sampling_mode);
    uint64_t bytes = interleaved_batch_state_bytes(layout.peak_active_width, layout.lane_capacity);
    const auto add_entries = [&](uint64_t entries, size_t entry_bytes) {
        bytes = saturating_add_u64(bytes, saturating_multiply_u64(entries, entry_bytes));
    };
    const auto add_columns = [&](size_t columns) {
        bytes = saturating_add_u64(bytes,
                                   packed_bit_columns_storage_bytes(columns, layout.lane_capacity));
    };
    add_entries(layout.shot_index_entries, sizeof(uint32_t));
    add_columns(layout.symbol_columns);
    add_columns(layout.noise_carrier_columns);
    add_columns(layout.expression_register_columns);
    add_columns(layout.record_columns);
    add_columns(layout.detector_columns);
    add_columns(layout.observable_columns);
    add_columns(layout.forced_readout_columns);
    add_entries(layout.exp_value_entries, sizeof(double));
    add_entries(layout.live_word_entries, sizeof(uint64_t));
    add_entries(layout.scratch_word_entries, sizeof(uint64_t));
    add_entries(layout.compaction_source_entries, sizeof(uint32_t));
    add_entries(layout.lane_byte_entries, sizeof(uint8_t));
    add_entries(layout.signed_sine_entries, sizeof(double));
    add_entries(layout.probability_zero_entries, sizeof(double));
    add_entries(layout.probability_one_entries, sizeof(double));
    add_entries(layout.lane_value_entries, sizeof(double));
    return bytes;
}

}  // namespace batch_detail

BatchExecutionPolicy resolve_batch_execution_policy(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, std::optional<uint32_t> requested_batch_size,
    BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes) {
    if (requested_batch_size.has_value() && *requested_batch_size == 0) {
        throw std::invalid_argument("batch_size must be a positive integer or 'auto'");
    }
    if (shots == 0 || plan.has_instruments()) {
        return {};
    }
#if defined(__EMSCRIPTEN__)
    (void)shot_workers;
    (void)output_mode;
    (void)sampling_mode;
    (void)additional_worker_bytes;
    if (intra_shot_workers > 1 && requested_batch_size.value_or(1) > 1) {
        throw std::invalid_argument("packed batch_size is incompatible with intra-shot workers");
    }
    if (requested_batch_size.value_or(1) > 1) {
        throw std::invalid_argument("packed batch_size is unavailable in WebAssembly builds");
    }
    return {};
#else
    if (intra_shot_workers > 1) {
        if (requested_batch_size.has_value() && *requested_batch_size > 1) {
            throw std::invalid_argument(
                "packed batch_size is incompatible with intra-shot workers");
        }
        return {};
    }
    if (requested_batch_size.has_value()) {
        const uint32_t capacity =
            std::max(uint32_t{1}, std::min({*requested_batch_size, shots, kMaxExplicitBatchShots}));
        if (capacity > 1 && interleaved_batch_state_bytes(plan.peak_active_width(), capacity) >
                                kMaxExplicitBatchStateBudget) {
            throw std::invalid_argument(
                "explicit batch_size exceeds the 64 MiB packed-state limit; request a smaller "
                "batch_size");
        }
        return {.lane_capacity = capacity,
                .worker_count = batch_worker_count(shots, shot_workers, capacity)};
    }
    if (shots < kDefaultMinAutoBatchShots) {
        return {};
    }
    // A static plan does not predict survivor lifetimes portably enough to
    // choose packed execution without an explicit user request.
    if (plan.has_postselection()) {
        return {};
    }
    if (plan.peak_active_width() > 5 ||
        (plan.peak_active_width() == 5 &&
         plan.estimated_batch_lane_work() > kDefaultMaxWidthFiveBatchLaneWork)) {
        return {};
    }
    uint32_t capacity = std::min(shots, kDefaultMaxAutoBatchShots);
    while (capacity >= kDefaultMinAutoBatchShots) {
        if (interleaved_batch_state_bytes(plan.peak_active_width(), capacity) >
            kDefaultBatchStateBudget) {
            capacity = std::bit_floor(capacity - 1);
            continue;
        }
        const uint64_t worker_bytes = saturating_add_u64(
            batch_detail::batch_worker_storage_bytes(plan, capacity, output_mode, sampling_mode),
            additional_worker_bytes);
        if (worker_bytes <= kDefaultBatchWorkerBudget) {
            const uint32_t requested_workers = batch_worker_count(shots, shot_workers, capacity);
            const uint64_t memory_workers = std::max<uint64_t>(
                1, kDefaultBatchTotalWorkerBudget / std::max<uint64_t>(1, worker_bytes));
            return {.lane_capacity = capacity,
                    .worker_count = static_cast<uint32_t>(
                        std::min<uint64_t>(requested_workers, memory_workers))};
        }
        capacity = std::bit_floor(capacity - 1);
    }
    return {};
#endif
}

}  // namespace clifft::sampling
