#pragma once

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/tuning.h"
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

// Default bound on the coefficients, collapse scratch, and page alignment of
// the states one shared worker holds. States are allocated as trajectories
// split, so a call can reach this bound once per active worker.
inline constexpr uint64_t kDefaultSharedWorkerStateBudget = 64 * 1024 * 1024;

// Largest lane capacity considered by automatic shared selection.
inline constexpr uint32_t kMaxAutoSharedBatchShots = 8192;

// Bound on the classical buffers of one automatically sized shared worker.
inline constexpr uint64_t kMaxAutoSharedWorkerClassicalBytes = 64 * 1024 * 1024;

// Automatic selection spreads a call across the requested workers only while
// each worker keeps at least this many lanes to share states among.
inline constexpr uint32_t kMinAutoSharedLanesPerWorker = 1024;

class SharingKey;

// Largest lane capacity, up to kMaxAutoSharedBatchShots, that one worker can
// run within max_worker_state_bytes when every lane keeps its own state and
// within the classical buffer bound. Zero when the budget cannot hold a state.
[[nodiscard]] uint32_t max_safe_shared_capacity(
    const ExecutablePlan& plan, const PreparedBatchNoise& noise, const SharingKey* key,
    uint64_t max_worker_state_bytes,
    BatchOutputMode output_mode = BatchOutputMode::AggregateSurvivors);

// Automatic capacity and worker count for one call. Depends only on the safe
// capacity and the call, so a seeded call replays independently of earlier
// calls. Short calls shrink the capacity to the call and to each worker's share.
[[nodiscard]] BatchExecutionPolicy resolve_shared_execution_policy(uint32_t safe_capacity,
                                                                   uint32_t shots,
                                                                   uint32_t workers) noexcept;

// Capacities calibration compares for one call: the automatic baseline first,
// then a fixed ladder up to the safe capacity. Each candidate uses as many of
// the requested workers as it has batches for.
[[nodiscard]] std::vector<batch_detail::BatchTuningCandidate> shared_tuning_candidates(
    uint32_t safe_capacity, uint32_t shots, uint32_t workers, uint64_t output_row_bytes = 0);

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
// Dormant measurements and sparse symmetric readout use packed draws;
// active measurements and all other readout retain per-shot RNG streams. A pool
// growth boundary suspends dispatch before an action can require more states.
class SharingExecutor : public BatchClassicalState {
  public:
    SharingExecutor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                    const EarlyPostselection& checks, uint32_t lane_capacity,
                    uint32_t initial_states, bool share = true, const SharingKey* key = nullptr,
                    uint64_t max_worker_state_bytes = kDefaultSharedWorkerStateBudget,
                    BatchOutputMode output_mode = BatchOutputMode::AggregateSurvivors);

    // Classical buffers retained by one worker, excluding its state pool.
    [[nodiscard]] static uint64_t classical_storage_bytes(
        const ExecutablePlan& plan, const PreparedBatchNoise& noise, const SharingKey* key,
        uint32_t lane_capacity, BatchOutputMode output_mode = BatchOutputMode::AggregateSurvivors);

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
