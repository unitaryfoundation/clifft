#include "clifft/sampling/batch/policy.h"

#include "clifft/sampling/executable_plan.h"
#include "clifft/util/numeric.h"

#if !defined(__EMSCRIPTEN__)
#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/external_inputs.h"
#include "clifft/sampling/batch/interleaved_state.h"
#endif

#include <algorithm>
#include <bit>
#include <limits>
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

#if !defined(__EMSCRIPTEN__)
namespace batch_detail {

BatchWorkerStorageLayout batch_worker_storage_layout(const ExecutablePlan& plan,
                                                     uint32_t lane_capacity,
                                                     BatchOutputMode output_mode,
                                                     BatchSamplingMode sampling_mode,
                                                     const PreparedBatchNoise* noise) {
    const BatchNoiseMode noise_mode = noise ? noise->mode() : BatchNoiseMode::Dense;
    const uint32_t carriers = noise ? noise->num_carriers() : plan.num_batch_noise_carriers();
    BatchWorkerStorageLayout layout;
    layout.peak_active_width = plan.peak_active_width();
    layout.initial_active_width = plan.initial_active_width();
    layout.lane_capacity = lane_capacity;
    layout.word_capacity = packed_word_count(lane_capacity);
    layout.shot_index_entries = lane_capacity;
    layout.symbol_columns =
        noise_mode == BatchNoiseMode::Dense && carriers == 0 && plan.num_presampled_symbols() != 0
            ? plan.num_symbols()
            : 0;
    layout.noise_carrier_columns = carriers;
    layout.expression_register_columns = plan.num_expression_registers();
    layout.record_columns =
        output_mode == BatchOutputMode::Rows || plan.output_parities_read_records()
            ? static_cast<size_t>(plan.num_visible_records()) + plan.num_hidden_records()
            : 0;
    layout.detector_columns = output_mode == BatchOutputMode::Rows ? plan.num_detectors() : 0;
    layout.observable_columns = plan.num_observables();
    layout.forced_readout_columns =
        sampling_mode == BatchSamplingMode::FixedFaults ? plan.num_readout_noise_sites() : 0;
    if (sampling_mode == BatchSamplingMode::ExternalInputs) {
        if (plan.num_readout_noise_sites() > std::numeric_limits<size_t>::max() / 2) {
            throw std::length_error("packed external readout column count exceeds size_t range");
        }
        layout.forced_readout_columns = size_t{2} * plan.num_readout_noise_sites();
    }
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
                                    BatchOutputMode output_mode, BatchSamplingMode sampling_mode,
                                    const PreparedBatchNoise* noise) {
    return saturating_add_u64(
        interleaved_batch_state_bytes(plan.peak_active_width(), lane_capacity),
        batch_worker_classical_bytes(plan, lane_capacity, output_mode, sampling_mode, noise));
}

uint64_t batch_worker_classical_bytes(const ExecutablePlan& plan, uint32_t lane_capacity,
                                      BatchOutputMode output_mode, BatchSamplingMode sampling_mode,
                                      const PreparedBatchNoise* noise) {
    const BatchWorkerStorageLayout layout =
        batch_worker_storage_layout(plan, lane_capacity, output_mode, sampling_mode, noise);
    uint64_t bytes = 0;
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
    if (noise && noise->mode() == BatchNoiseMode::SiteRows) {
        add_entries(
            saturating_multiply_u64(lane_capacity, (plan.num_expression_registers() + 63) / 64),
            sizeof(uint64_t));
    }
    // External workers retain the caller's input buffer as well as the readout
    // copy whose lane order follows intermediate survivor compaction.
    if (sampling_mode == BatchSamplingMode::ExternalInputs) {
        bytes = saturating_add_u64(bytes, BatchExternalInputs::storage_bytes(plan, lane_capacity));
    }
    return bytes;
}

}  // namespace batch_detail
#endif

bool automatic_batch_eligible(const ExecutablePlan& plan, uint32_t shots,
                              uint32_t intra_shot_workers, BatchOutputMode output_mode) noexcept {
#if defined(__EMSCRIPTEN__)
    (void)plan;
    (void)shots;
    (void)intra_shot_workers;
    (void)output_mode;
    return false;
#else
    if (intra_shot_workers > 1 || plan.has_instruments() || shots < kDefaultMinAutoBatchShots) {
        return false;
    }
    // Packed SIMD kernels beat scalar through width 8; when an explicit
    // non-shared call faces early postselection, cap packed width at 4 so
    // rejected lanes do not carry wide coefficient work.
    const uint32_t max_width = plan.has_early_postselection(output_mode) ? 4 : kMaxAutoBatchWidth;
    return plan.peak_active_width() <= max_width;
#endif
}

BatchNoiseMode select_batch_noise_mode(const ExecutablePlan& plan, uint32_t shots,
                                       std::optional<uint32_t> requested_batch_size) noexcept {
    // Lane capacity before memory limits, which can only lower the fault count.
    const uint32_t lanes = std::min(
        {shots, requested_batch_size.value_or(kDefaultMaxAutoBatchShots), kMaxExplicitBatchShots});
    if (plan.expected_noise_faults() * lanes <= static_cast<double>(plan.num_noise_sites())) {
        return BatchNoiseMode::Sparse;
    }
    return requested_batch_size ? BatchNoiseMode::Dense : BatchNoiseMode::SiteSparse;
}

BatchExecutionPolicy resolve_batch_execution_policy(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, std::optional<uint32_t> requested_batch_size,
    BatchSamplingMode sampling_mode, uint64_t additional_worker_bytes,
    uint64_t additional_worker_bit_columns, const PreparedBatchNoise* noise) {
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
    (void)additional_worker_bit_columns;
    (void)noise;
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
    if (!automatic_batch_eligible(plan, shots, intra_shot_workers, output_mode)) {
        return {};
    }
    // Narrow postselected plans keep batches small so rejected lanes carry
    // less work. Ordinary width-zero plans have no coefficient work, and their
    // classical cost per shot falls as batches grow.
    const bool ordinary_width_zero =
        plan.peak_active_width() == 0 && sampling_mode == BatchSamplingMode::Ordinary;
    const uint32_t max_capacity =
        plan.has_postselection() && plan.peak_active_width() < 5 && !ordinary_width_zero
            ? kDefaultMaxPostselectedBatchShots
            : kDefaultMaxAutoBatchShots;
    // Wider kernels benefit from parallel batches when each requested worker
    // has enough shots to fill one. Shorter requests retain their packed size.
    uint32_t capacity = std::min(shots, max_capacity);
    const uint32_t shots_per_worker = shots / std::max(uint32_t{1}, shot_workers);
    if (plan.peak_active_width() > 5 && shots_per_worker >= kDefaultMinAutoBatchShots) {
        capacity = std::min(capacity, shots_per_worker);
    }
    while (capacity >= kDefaultMinAutoBatchShots) {
        if (interleaved_batch_state_bytes(plan.peak_active_width(), capacity) >
            kDefaultBatchStateBudget) {
            capacity = std::bit_floor(capacity - 1);
            continue;
        }
        const uint64_t worker_bytes = saturating_add_u64(
            batch_detail::batch_worker_storage_bytes(plan, capacity, output_mode, sampling_mode,
                                                     noise),
            saturating_add_u64(
                additional_worker_bytes,
                packed_bit_columns_storage_bytes(additional_worker_bit_columns, capacity)));
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
