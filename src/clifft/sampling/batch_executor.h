#pragma once

#include "clifft/sampling/batch_bits.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/util/page_allocation.h"
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

inline constexpr uint32_t kDefaultMaxAutoBatchShots = 512;
inline constexpr uint32_t kMaxExplicitBatchShots = 2048;
inline constexpr size_t kDefaultBatchStateBudget = 768 * 1024;
inline constexpr uint32_t kDefaultMinAutoBatchShots = 64;

enum class BatchOutputMode : uint8_t {
    Rows,
    AggregateSurvivors,
};

// Resolve one worker's retained lane capacity. requested_batch_size is empty
// for conservative automatic selection, one for the scalar path, or an
// explicit packed capacity. Validation and allocation happen before dispatch.
[[nodiscard]] uint32_t resolve_batch_capacity(const ExecutablePlan& plan, uint32_t shots,
                                              uint32_t shot_workers, uint32_t intra_shot_workers,
                                              std::optional<uint32_t> requested_batch_size);

// Single-threaded packed executor for fixed plans. It traverses one immutable
// action stream for every lane while retaining the existing contiguous State
// kernels for each live shot. Instruments and continuation boundaries remain
// on Executor and are rejected before construction.
class BatchExecutor {
  public:
    BatchExecutor(const ExecutablePlan& plan, uint32_t lane_capacity,
                  BatchOutputMode output_mode = BatchOutputMode::Rows);

    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;
    BatchExecutor(BatchExecutor&&) = delete;
    BatchExecutor& operator=(BatchExecutor&&) = delete;

    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                   KFaultSampler& fault_sampler) noexcept;

    [[nodiscard]] uint32_t lane_capacity() const noexcept { return lane_capacity_; }
    [[nodiscard]] uint32_t attempted_shots() const noexcept { return attempted_shots_; }
    [[nodiscard]] uint32_t surviving_shots() const noexcept { return live_count_; }
    [[nodiscard]] uint32_t accumulate_survivor_counts(
        std::span<uint64_t> observable_ones) const noexcept;
    [[nodiscard]] uint32_t shot_index(uint32_t lane) const noexcept;
    [[nodiscard]] bool measurement(uint32_t lane, uint32_t record) const noexcept;
    [[nodiscard]] bool detector(uint32_t lane, uint32_t detector) const noexcept;
    [[nodiscard]] bool observable(uint32_t lane, uint32_t observable) const noexcept;
    [[nodiscard]] double exp_val(uint32_t lane, uint32_t exp_val) const noexcept;
    [[nodiscard]] uint64_t dust_clamps() const noexcept { return dust_clamps_; }
    [[nodiscard]] uint64_t compactions() const noexcept { return compactions_; }

  private:
    void reset_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;
    void sample_presampled_noise() noexcept;
    void assign_forced_faults(KFaultSampler& fault_sampler) noexcept;
    void activate_noise_site(uint32_t lane, uint32_t site) noexcept;
    void initialize_expression_registers() noexcept;
    void propagate_symbol(uint32_t symbol) noexcept;
    void assign_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept;

    template <ExecutorBackend Backend>
    void execute_actions() noexcept;
    template <ExecutorBackend Backend>
    void execute_action(const ExecutablePlan::ExecuteRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action,
                        size_t action_index) noexcept;
    void execute_action(const ExecutablePlan::ExecutePromotion& action,
                        size_t action_index) noexcept;
    template <ExecutorBackend Backend>
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
    [[nodiscard]] bool lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept;
    [[nodiscard]] bool is_live(uint32_t lane) const noexcept;
    [[nodiscard]] State& state(uint32_t lane) noexcept;
    [[nodiscard]] const State& state(uint32_t lane) const noexcept;
    [[nodiscard]] bool sample_active_branch(uint32_t lane,
                                            MeasurementProbabilities probabilities) noexcept;
    [[nodiscard]] bool should_compact(size_t action_index) const noexcept;
    void compact_live_lanes() noexcept;
    void finalize_live_lanes() noexcept;

    const ExecutablePlan* plan_;
    BatchOutputMode output_mode_ = BatchOutputMode::Rows;
    uint32_t lane_capacity_ = 0;
    size_t word_capacity_ = 0;
    size_t state_bytes_per_lane_ = 0;

    // One allocation contains every lane's shot-major coefficient and scratch
    // block. State facades borrow their corresponding aligned slices.
    PageAlignedAllocation state_storage_;
    std::vector<State> states_;
    std::vector<uint32_t> state_slots_;
    std::vector<Xoshiro256PlusPlus> rngs_;
    std::vector<uint32_t> shot_indices_;

    PackedBitColumns symbols_;
    PackedBitColumns expression_registers_;
    PackedBitColumns records_;
    PackedBitColumns detectors_;
    PackedBitColumns observables_;
    PackedBitColumns forced_readout_;
    std::vector<double> exp_vals_;

    std::vector<uint64_t> live_words_;
    std::vector<uint64_t> scratch_words_;
    std::vector<uint64_t> compaction_scratch_;

    uint32_t attempted_shots_ = 0;
    uint32_t active_lanes_ = 0;
    uint32_t live_count_ = 0;
    bool fixed_fault_mode_ = false;
    uint64_t dust_clamps_ = 0;
    uint64_t compactions_ = 0;
};

}  // namespace clifft::sampling
