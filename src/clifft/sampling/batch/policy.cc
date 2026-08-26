#include "clifft/sampling/batch/policy.h"

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/interleaved_state.h"
#include "clifft/sampling/executable_plan.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace clifft::sampling {

namespace {

#if !defined(__EMSCRIPTEN__)
[[nodiscard]] uint64_t saturating_add(uint64_t left, uint64_t right) noexcept {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    return right > kMax - left ? kMax : left + right;
}

[[nodiscard]] uint64_t saturating_multiply(uint64_t left, uint64_t right) noexcept {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    return left != 0 && right > kMax / left ? kMax : left * right;
}

[[nodiscard]] uint64_t estimated_batch_worker_bytes(const ExecutablePlan& plan,
                                                    uint32_t lane_capacity,
                                                    BatchOutputMode output_mode,
                                                    BatchSamplingMode sampling_mode,
                                                    uint64_t additional_worker_bytes) {
    const uint64_t words = packed_word_count(lane_capacity);
    const uint64_t records =
        output_mode == BatchOutputMode::Rows || plan.has_batch_record_parities()
            ? static_cast<uint64_t>(plan.num_visible_records()) + plan.num_hidden_records()
            : 0;
    const uint64_t symbols =
        plan.num_batch_noise_carriers() == 0 && plan.num_presampled_symbols() != 0
            ? plan.num_symbols()
            : 0;
    const uint64_t forced_readout =
        sampling_mode == BatchSamplingMode::FixedFaults ? plan.num_readout_noise_sites() : 0;
    const uint64_t packed_columns =
        symbols + plan.num_batch_noise_carriers() + plan.num_expression_registers() + records +
        (output_mode == BatchOutputMode::Rows ? plan.num_detectors() : 0) + plan.num_observables() +
        forced_readout;
    const uint64_t packed_bytes =
        saturating_multiply(saturating_multiply(packed_columns, words), sizeof(uint64_t));

    uint64_t bytes = interleaved_batch_state_bytes(plan.peak_active_width(), lane_capacity);
    bytes = saturating_add(bytes, packed_bytes);
    bytes = saturating_add(bytes, saturating_multiply(words, 2 * sizeof(uint64_t)));

    constexpr uint64_t kLaneBytes = 2 * sizeof(uint32_t) + sizeof(uint8_t) + 4 * sizeof(double);
    bytes = saturating_add(bytes, saturating_multiply(lane_capacity, kLaneBytes));
    if (output_mode == BatchOutputMode::Rows) {
        bytes = saturating_add(
            bytes, saturating_multiply(saturating_multiply(plan.num_exp_vals(), lane_capacity),
                                       sizeof(double)));
    }
    return saturating_add(bytes, additional_worker_bytes);
}

[[nodiscard]] uint32_t batch_worker_count(uint32_t shots, uint32_t shot_workers,
                                          uint32_t lane_capacity) noexcept {
    const uint64_t batches = (static_cast<uint64_t>(shots) + lane_capacity - 1) / lane_capacity;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(shot_workers, batches)));
}
#endif

}  // namespace

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
    if (plan.peak_active_width() > 5) {
        return {};
    }
    uint32_t capacity = std::min(shots, kDefaultMaxAutoBatchShots);
    while (capacity >= kDefaultMinAutoBatchShots) {
        if (interleaved_batch_state_bytes(plan.peak_active_width(), capacity) >
            kDefaultBatchStateBudget) {
            capacity = std::bit_floor(capacity - 1);
            continue;
        }
        const uint64_t worker_bytes = estimated_batch_worker_bytes(
            plan, capacity, output_mode, sampling_mode, additional_worker_bytes);
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
