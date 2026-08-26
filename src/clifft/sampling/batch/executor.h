#pragma once

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/interleaved_kernels.h"
#include "clifft/sampling/batch/interleaved_state.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/util/shot_seed.h"
#include "clifft/util/xoshiro.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft {
class KFaultSampler;
}

namespace clifft::sampling {

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

enum class BatchOutputMode : uint8_t {
    Rows,
    AggregateSurvivors,
};

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

// Single-threaded packed executor for fixed plans. Coefficients are
// basis-major and shot-interleaved so prepared actions vectorize across lanes.
// Instruments and continuation boundaries remain on Executor.
class BatchExecutor {
  public:
    BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                  BatchOutputMode output_mode = BatchOutputMode::Rows,
                  BatchSamplingMode sampling_mode = BatchSamplingMode::Ordinary);

    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;
    BatchExecutor(BatchExecutor&&) = delete;
    BatchExecutor& operator=(BatchExecutor&&) = delete;

    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                   KFaultSampler& fault_sampler) noexcept;

    [[nodiscard]] uint32_t surviving_shots() const noexcept { return live_count_; }
    [[nodiscard]] uint32_t accumulate_survivor_counts(
        std::span<uint64_t> observable_ones) const noexcept;
    [[nodiscard]] uint32_t shot_index(uint32_t lane) const noexcept;
    [[nodiscard]] bool measurement(uint32_t lane, uint32_t record) const noexcept;
    [[nodiscard]] bool detector(uint32_t lane, uint32_t detector) const noexcept;
    [[nodiscard]] bool observable(uint32_t lane, uint32_t observable) const noexcept;
    [[nodiscard]] double exp_val(uint32_t lane, uint32_t exp_val) const noexcept;

  private:
    void reset_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void sample_presampled_noise() noexcept;
    void assign_forced_faults(KFaultSampler& fault_sampler) noexcept;
    void activate_noise_site(uint32_t lane, uint32_t site) noexcept;
    void initialize_expression_registers() noexcept;
    void initialize_presampled_expressions() noexcept;
    void finalize_presampled_symbols() noexcept;
    void propagate_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept;
    void assign_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept;
    void fill_random_half_bits() noexcept;

    void execute_actions() noexcept;
    void execute_action(const ExecutablePlan::ExecuteRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecutePromotion& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteClassicalRecord& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDetector& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteObservable& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteExpectation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteInstrument& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteBoundary& action,
                        size_t action_index) noexcept;

    [[nodiscard]] std::span<const uint64_t> evaluate(
        ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] std::span<const uint64_t> evaluate_record_parity(uint32_t parity_index) noexcept;
    [[nodiscard]] bool lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept;
    [[nodiscard]] bool is_live(uint32_t lane) const noexcept;
    [[nodiscard]] uint32_t active_lanes() const noexcept { return state_.active_lanes(); }
    [[nodiscard]] bool should_compact(size_t action_index) const noexcept;
    void compact_live_lanes() noexcept;
    void finalize_live_lanes() noexcept;

    const ExecutablePlan* plan_;
    const BatchOutputMode output_mode_;
    const BatchSamplingMode sampling_mode_;
    uint32_t lane_capacity_ = 0;
    size_t word_capacity_ = 0;

    InterleavedBatchState state_;
    Xoshiro256PlusPlus rng_;
    std::vector<uint32_t> shot_indices_;

    PackedBitColumns symbols_;
    PackedBitColumns batch_noise_carriers_;
    PackedBitColumns expression_registers_;
    PackedBitColumns records_;
    PackedBitColumns detectors_;
    PackedBitColumns observables_;
    PackedBitColumns forced_readout_;
    std::vector<double> exp_vals_;

    std::vector<uint64_t> live_words_;
    std::vector<uint64_t> scratch_words_;
    std::vector<uint32_t> compaction_sources_;
    std::vector<uint8_t> lane_bytes_;
    std::vector<double> signed_sines_;
    std::vector<double> probability_zero_;
    std::vector<double> probability_one_;
    std::vector<double> lane_values_;

    uint32_t live_count_ = 0;
};

}  // namespace clifft::sampling
