#pragma once

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/external_inputs.h"
#include "clifft/sampling/batch/interleaved_kernels.h"
#include "clifft/sampling/batch/interleaved_state.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/util/shot_seed.h"
#include "clifft/util/xoshiro.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {
class KFaultSampler;
}

namespace clifft::sampling {

// Single-threaded packed executor for fixed plans. Coefficients are
// basis-major and shot-interleaved so prepared actions vectorize across lanes.
// Instruments and continuation boundaries remain on Executor.
class BatchExecutor : public BatchClassicalState {
  public:
    BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                  BatchOutputMode output_mode = BatchOutputMode::Rows,
                  BatchSamplingMode sampling_mode = BatchSamplingMode::Ordinary,
                  const PreparedBatchNoise* noise = nullptr);

    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;
    BatchExecutor(BatchExecutor&&) = delete;
    BatchExecutor& operator=(BatchExecutor&&) = delete;

    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                   KFaultSampler& fault_sampler) noexcept;

    // Inputs must belong to this plan and have this worker's lane capacity.
    // Every noise/readout site is externally supplied; quantum measurements
    // still use the batch RNG. Padding and inactive input lanes are ignored.
    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                   const BatchExternalInputs& inputs) noexcept;

    [[nodiscard]] bool compacted_during_execution() const noexcept {
        return compacted_during_execution_;
    }
    [[nodiscard]] double exp_val(uint32_t lane, uint32_t exp_val) const noexcept;

  private:
    using BatchClassicalState::execute_action;

    enum class CompactionMode { ContinueExecution, FinalizeOutputs };

    BatchExecutor(const ExecutablePlan& plan, BatchOutputMode output_mode,
                  BatchSamplingMode sampling_mode,
                  const batch_detail::BatchWorkerStorageLayout& storage,
                  const PreparedBatchNoise* noise);

    void reset_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;

    void execute_actions() noexcept;
    void execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecutePromotion& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteExpectation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteInstrument& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteBoundary& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFrameClifford& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFrameMeasurement& action) noexcept;
    [[nodiscard]] bool should_compact(
        const ExecutablePlan::ExecuteDetector& detector) const noexcept;
    void compact_live_lanes(CompactionMode mode) noexcept;
    void finalize_live_lanes() noexcept;

    InterleavedBatchState state_;
    std::vector<double> exp_vals_;
    std::vector<uint32_t> compaction_sources_;
    std::vector<uint8_t> lane_bytes_;
    std::vector<double> signed_sines_;
    std::vector<double> probability_zero_;
    std::vector<double> probability_one_;
    std::vector<double> lane_values_;
    bool compacted_during_execution_ = false;
};

}  // namespace clifft::sampling
