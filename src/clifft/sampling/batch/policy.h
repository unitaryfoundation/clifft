#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace clifft::sampling {

class ExecutablePlan;

// Largest lane capacity considered by automatic packed selection.
inline constexpr uint32_t kDefaultMaxAutoBatchShots = 2048;

// Hard lane-capacity ceiling for explicit packed requests.
inline constexpr uint32_t kMaxExplicitBatchShots = 2048;

// Target coefficient-state bytes retained by one automatic packed worker.
inline constexpr size_t kDefaultBatchStateBudget = 768 * 1024;

// Maximum complete retained footprint of one automatic packed worker.
inline constexpr size_t kDefaultBatchWorkerBudget = 8 * 1024 * 1024;

// Maximum complete retained footprint across all automatic packed workers.
inline constexpr size_t kDefaultBatchTotalWorkerBudget = 64 * 1024 * 1024;

// Maximum dense coefficient state retained by one explicit packed worker.
inline constexpr size_t kMaxExplicitBatchStateBudget = 64 * 1024 * 1024;

// Minimum request and capacity for automatic packed execution.
inline constexpr uint32_t kDefaultMinAutoBatchShots = 64;

// Sustained width-five coefficient work eventually favors the scalar SIMD
// kernels even when a shorter plan at the same peak width batches profitably.
inline constexpr uint64_t kDefaultMaxWidthFiveBatchLaneWork = 16 * 1024;

// Selects retained per-shot rows or survivor-only aggregate output.
enum class BatchOutputMode : uint8_t {
    Rows,
    AggregateSurvivors,
};

// Selects ordinary Bernoulli noise or conditioned fixed-fault storage.
enum class BatchSamplingMode : uint8_t {
    Ordinary,
    FixedFaults,
};

namespace batch_detail {

// Approximate coefficient visits performed for one lane. Row output can add
// work, such as expectation-value probes, beyond the common execution path.
struct BatchLaneWork {
    uint64_t common = 0;
    uint64_t row_output = 0;

    [[nodiscard]] uint64_t for_output_mode(BatchOutputMode output_mode) const noexcept {
        if (output_mode == BatchOutputMode::AggregateSurvivors) {
            return common;
        }
        return common > std::numeric_limits<uint64_t>::max() - row_output
                   ? std::numeric_limits<uint64_t>::max()
                   : common + row_output;
    }
};

// Policy decisions need both total future work and the portion executed while
// the active state has the width whose packed/scalar crossover is calibrated.
struct BatchWorkEstimate {
    BatchLaneWork all_widths;
    BatchLaneWork width_five;
};

}  // namespace batch_detail

#if defined(__EMSCRIPTEN__)
// WebAssembly retains the scalar executor to minimize its binary footprint.
inline constexpr bool kPackedBatchExecutionAvailable = false;
#else
// Native builds include packed execution and its interleaved kernels.
inline constexpr bool kPackedBatchExecutionAvailable = true;
#endif

struct BatchExecutionPolicy {
    // Number of shot lanes in a full packed batch.
    uint32_t lane_capacity = 1;

    // Maximum simultaneous workers allowed by work and memory budgets.
    uint32_t worker_count = 1;
};

namespace batch_detail {

// Retained column and lane storage selected for one packed executor.
struct BatchWorkerStorageLayout {
    uint32_t peak_active_width = 0;
    uint32_t initial_active_width = 0;
    uint32_t lane_capacity = 0;
    size_t word_capacity = 0;
    size_t shot_index_entries = 0;
    size_t symbol_columns = 0;
    size_t noise_carrier_columns = 0;
    size_t expression_register_columns = 0;
    size_t record_columns = 0;
    size_t detector_columns = 0;
    size_t observable_columns = 0;
    size_t forced_readout_columns = 0;
    uint64_t exp_value_entries = 0;
    size_t live_word_entries = 0;
    size_t scratch_word_entries = 0;
    size_t compaction_source_entries = 0;
    size_t lane_byte_entries = 0;
    size_t signed_sine_entries = 0;
    size_t probability_zero_entries = 0;
    size_t probability_one_entries = 0;
    size_t lane_value_entries = 0;
};

[[nodiscard]] BatchWorkerStorageLayout batch_worker_storage_layout(const ExecutablePlan& plan,
                                                                   uint32_t lane_capacity,
                                                                   BatchOutputMode output_mode,
                                                                   BatchSamplingMode sampling_mode);

[[nodiscard]] uint64_t batch_worker_storage_bytes(const ExecutablePlan& plan,
                                                  uint32_t lane_capacity,
                                                  BatchOutputMode output_mode,
                                                  BatchSamplingMode sampling_mode);

}  // namespace batch_detail

// Resolve deterministic lane boundaries first, then cap automatic workers by
// the aggregate retained-memory budget. Callers include wrapper-owned scratch
// that is not part of BatchExecutor in additional_worker_bytes.
[[nodiscard]] BatchExecutionPolicy resolve_batch_execution_policy(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, std::optional<uint32_t> requested_batch_size,
    BatchSamplingMode sampling_mode = BatchSamplingMode::Ordinary,
    uint64_t additional_worker_bytes = 0);

}  // namespace clifft::sampling
