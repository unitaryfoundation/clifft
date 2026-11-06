#pragma once

#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/external_inputs.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/batch/random_bits.h"
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

// Immutable preparation shared by workers using direct fault propagation.
// Site traversal skips silent shots independently for each categorical site.
class PreparedBatchNoise {
  public:
    PreparedBatchNoise(const ExecutablePlan& plan, BatchNoiseMode mode);
    [[nodiscard]] BatchNoiseMode mode() const noexcept { return mode_; }

  private:
    friend class BatchClassicalState;
    const ExecutablePlan* plan_;
    BatchNoiseMode mode_;
    size_t row_stride_ = 0;
    std::vector<PreparedBernoulli> sites_;
    std::vector<uint64_t> outcome_rows_;
};

// Packed noise, affine registers, and outputs shared by CPU batch executors.
// Classical lane capacity is independent of coefficient storage.
class BatchClassicalState {
  public:
    [[nodiscard]] std::span<const uint64_t> survivor_mask() const noexcept { return live_words_; }
    [[nodiscard]] uint32_t surviving_shots() const noexcept { return live_count_; }
    [[nodiscard]] uint32_t accumulate_survivor_counts(
        std::span<uint64_t> observable_ones) const noexcept;
    [[nodiscard]] uint32_t shot_index(uint32_t lane) const noexcept;
    // Finalized outputs for lanes [0, surviving_shots()). The record table's
    // visible prefix is columns [0, plan.num_visible_records()).
    [[nodiscard]] std::span<const uint32_t> shot_indices() const noexcept;
    [[nodiscard]] const PackedBitColumns& record_columns() const noexcept { return records_; }
    [[nodiscard]] const PackedBitColumns& detector_columns() const noexcept { return detectors_; }
    [[nodiscard]] const PackedBitColumns& observable_columns() const noexcept {
        return observables_;
    }
    [[nodiscard]] bool measurement(uint32_t lane, uint32_t record) const noexcept;
    [[nodiscard]] bool detector(uint32_t lane, uint32_t detector) const noexcept;
    [[nodiscard]] bool observable(uint32_t lane, uint32_t observable) const noexcept;

  protected:
    BatchClassicalState(const ExecutablePlan& plan, BatchOutputMode output_mode,
                        BatchSamplingMode sampling_mode,
                        const batch_detail::BatchWorkerStorageLayout& storage,
                        const PreparedBatchNoise* noise = nullptr);

    void reset_classical(const SeedRoot& root, uint32_t first_shot, uint32_t shots) noexcept;

    void sample_presampled_noise() noexcept;
    void assign_external_inputs(const BatchExternalInputs& inputs) noexcept;
    void assign_forced_faults(KFaultSampler& fault_sampler) noexcept;
    struct NoiseSiteActivator;
    void initialize_expression_registers() noexcept;
    void initialize_presampled_expressions() noexcept;
    void finalize_presampled_symbols() noexcept;
    void propagate_symbol(uint32_t symbol, std::span<const uint64_t> values) noexcept;
    void fill_random_half_bits() noexcept;
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteClassicalRecord& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteSymbolDefinition& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteObservable& action) noexcept;

    [[nodiscard]] std::span<const uint64_t> evaluate(
        ExecutablePlan::PreparedExpression expression) const noexcept;
    [[nodiscard]] std::span<const uint64_t> evaluate_observable(
        const ExecutablePlan::PreparedObservableValue& value) noexcept;
    [[nodiscard]] std::span<const uint64_t> evaluate_record_parity(
        ExecutablePlan::PreparedRecordParity parity) noexcept;
    [[nodiscard]] bool lane_bit(std::span<const uint64_t> bits, uint32_t lane) const noexcept;
    [[nodiscard]] bool is_live(uint32_t lane) const noexcept;
    [[nodiscard]] uint32_t active_lanes() const noexcept { return active_lanes_; }

    const PreparedBatchNoise* noise_;
    std::vector<uint64_t> noise_rows_;
    const ExecutablePlan* plan_;
    const BatchOutputMode output_mode_;
    const BatchSamplingMode sampling_mode_;
    uint32_t lane_capacity_ = 0;
    size_t word_capacity_ = 0;
    Xoshiro256PlusPlus rng_;
    std::vector<uint32_t> shot_indices_;
    PackedBitColumns symbols_;
    PackedBitColumns batch_noise_carriers_;
    PackedBitColumns expression_registers_;
    PackedBitColumns records_;
    PackedBitColumns detectors_;
    PackedBitColumns observables_;
    PackedBitColumns forced_readout_;
    std::vector<uint64_t> live_words_;
    std::vector<uint64_t> scratch_words_;
    uint32_t live_count_ = 0;
    uint32_t active_lanes_ = 0;
};

}  // namespace clifft::sampling
