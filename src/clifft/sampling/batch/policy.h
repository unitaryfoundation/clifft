#pragma once

#include <cstddef>
#include <cstdint>
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

#if defined(__EMSCRIPTEN__)
// WebAssembly retains the scalar executor to minimize its binary footprint.
inline constexpr bool kPackedBatchExecutionAvailable = false;
#else
// Native builds include packed execution and its interleaved kernels.
inline constexpr bool kPackedBatchExecutionAvailable = true;
#endif

struct BatchExecutionPolicy {
    // Stable number of shot lanes assigned to each packed batch.
    uint32_t lane_capacity = 1;

    // Maximum simultaneous workers allowed by work and memory budgets.
    uint32_t worker_count = 1;
};

// Resolve deterministic lane boundaries first, then cap automatic workers by
// the aggregate retained-memory budget. Callers include wrapper-owned scratch
// that is not part of BatchExecutor in additional_worker_bytes.
[[nodiscard]] BatchExecutionPolicy resolve_batch_execution_policy(
    const ExecutablePlan& plan, uint32_t shots, uint32_t shot_workers, uint32_t intra_shot_workers,
    BatchOutputMode output_mode, std::optional<uint32_t> requested_batch_size,
    BatchSamplingMode sampling_mode = BatchSamplingMode::Ordinary,
    uint64_t additional_worker_bytes = 0);

}  // namespace clifft::sampling
