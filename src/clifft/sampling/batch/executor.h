#pragma once

#include "clifft/sampling/batch/bits.h"
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
    BatchExecutor(const ExecutablePlan& plan, BatchOutputMode output_mode,
                  BatchSamplingMode sampling_mode,
                  const batch_detail::BatchWorkerStorageLayout& storage);

    void reset_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void sample_presampled_noise() noexcept;
    void assign_forced_faults(KFaultSampler& fault_sampler) noexcept;
    void activate_noise_site(uint32_t lane, uint32_t site) noexcept;
    void initialize_expression_registers() noexcept;
    void initialize_presampled_expressions() noexcept;
    void finalize_presampled_symbols() noexcept;
    void propagate_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept;
    void fill_random_half_bits() noexcept;

    void execute_actions() noexcept;
    void execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecutePromotion& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteClassicalRecord& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteObservable& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteExpectation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteInstrument& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteBoundary& action) noexcept;

    [[nodiscard]] std::span<const uint64_t> evaluate(
        ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] std::span<const uint64_t> evaluate_observable(
        const ExecutablePlan::PreparedObservableValue& value) noexcept;
    [[nodiscard]] std::span<const uint64_t> evaluate_record_parity(
        ExecutablePlan::PreparedRecordParity parity) noexcept;
    [[nodiscard]] bool lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept;
    [[nodiscard]] bool is_live(uint32_t lane) const noexcept;
    [[nodiscard]] uint32_t active_lanes() const noexcept { return state_.active_lanes(); }
    [[nodiscard]] bool should_compact(
        const ExecutablePlan::ExecuteDetector& detector) const noexcept;
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
