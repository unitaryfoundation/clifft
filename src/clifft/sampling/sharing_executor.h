#pragma once

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/state.h"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace clifft::sampling {

// Shared-trajectory workers combine BatchClassicalState's per-shot classical
// data with SharingExecutor's groups and pool of coefficient states. An optional
// SharingKey, prepared before execution, groups shots upfront by noise effects
// on future rotation signs. SharingExecutor applies prepared actions to each
// group and splits it when needed; SurvivorSampler retains the workers and
// their storage across sampling calls.

// Immutable noise-sign basis shared by workers. Branches affecting this basis
// split groups as they are sampled, keeping every later rotation sign uniform.
class SharingKey {
  public:
    explicit SharingKey(const ExecutablePlan& plan);
    [[nodiscard]] size_t num_bits() const noexcept { return registers_.size(); }

  private:
    friend class SharingExecutor;
    const ExecutablePlan* plan_;
    std::vector<uint32_t> registers_;
    std::vector<uint8_t> split_symbols_;
};

// Shares coefficient states across shots with identical prepared evolution.
// Active measurements and state-dependent readout retain per-shot RNG streams;
// dormant measurements and sparse symmetric readout use packed draws. A pool
// growth boundary suspends dispatch before an action can require more states.
class SharingExecutor : public BatchClassicalState {
  public:
    SharingExecutor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                    const EarlyPostselection& checks, uint32_t lane_capacity,
                    uint32_t initial_states, bool share = true, const SharingKey* key = nullptr,
                    uint64_t max_state_bytes = kDefaultBatchTotalWorkerBudget);

    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots);
    [[nodiscard]] uint32_t state_capacity() const noexcept {
        return static_cast<uint32_t>(states_.size());
    }
    [[nodiscard]] uint32_t peak_groups() const noexcept { return peak_groups_; }
    [[nodiscard]] uint32_t storage_boundaries() const noexcept { return storage_boundaries_; }
    [[nodiscard]] uint32_t entrance_rejected() const noexcept { return entrance_rejected_; }
    [[nodiscard]] uint64_t num_splits() const noexcept { return num_splits_; }

  private:
    using BatchClassicalState::execute_action;
    static constexpr uint32_t kNoShot = std::numeric_limits<uint32_t>::max();

    void initialize_groups(const SeedRoot& root, uint32_t first_shot, uint32_t shots);
    void grow_state_pool(uint32_t required);
    // Returns zero on completion, otherwise the capacity required to resume.
    template <ExecutorBackend Backend>
    [[nodiscard]] uint32_t dispatch_until_storage_boundary() noexcept;
    void copy_state(uint32_t to, uint32_t from) noexcept;
    [[nodiscard]] uint32_t split(uint32_t group) noexcept;
    [[nodiscard]] uint32_t split_by_sign(uint32_t group, std::span<const uint64_t> values) noexcept;
    // Uses the packed outcome in scratch_words_ and prepared sign dependencies.
    void split_on_symbol(uint32_t symbol) noexcept;
    void prune() noexcept;
    template <class F>
    void sign_action(std::span<const uint64_t> values, F&& apply) noexcept;

    template <ExecutorBackend Backend>
    void execute_action(const ExecutablePlan::ExecuteRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecutePromotion& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFusedRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDynamicFusedRotation& action) noexcept;
    template <ExecutorBackend Backend>
    void execute_action(const ExecutablePlan::ExecuteActiveMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDormantMeasurement& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteDetector& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteReadoutNoise& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteExpectation&) noexcept;
    void execute_action(const ExecutablePlan::ExecuteInstrument&) noexcept;
    void execute_action(const ExecutablePlan::ExecuteBoundary&) noexcept;

    const EarlyPostselection* checks_;
    bool share_;
    const SharingKey* key_;
    size_t key_stride_ = 0;
    std::vector<uint64_t> key_rows_;
    std::vector<uint32_t> key_hash_;
    std::vector<uint32_t> split_factors_;
    uint32_t max_states_ = 0;
    std::vector<State> states_;
    std::vector<Xoshiro256PlusPlus> shot_rng_;
    std::vector<uint32_t> heads_, next_, groups_, free_;
    std::vector<uint8_t> choices_;
    uint32_t group_count_ = 0;
    uint32_t free_count_ = 0;
    size_t action_index_ = 0;
    bool prune_pending_ = false;
    uint32_t peak_groups_ = 0;
    uint32_t storage_boundaries_ = 0;
    uint32_t entrance_rejected_ = 0;
    uint64_t num_splits_ = 0;
};

}  // namespace clifft::sampling
