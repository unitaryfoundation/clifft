#pragma once

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/online_frame.h"
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
// group and splits it when needed; each sampling call owns its workers and
// their storage.
//
// For a plan with a runtime Clifford frame, every group also refers to a frame
// tableau, and each shot keeps one sign bit per tableau row: its frame is that
// tableau with the flagged rows negated. Tableau updates do not depend on row
// signs, so groups that differ only in their coefficients keep sharing one
// tableau and each update runs once for all of them. Random frame measurements
// and conditional Pauli rotations change only sign bits. Shots take a new
// tableau where a conditional quarter turn fires, and a group splits where its
// shots differ in a frame rotation's sign or in the outcome of a measurement
// that collapses the coefficients. Groups sharing a tableau merge again when
// its last active coordinate is measured. A run of consecutive quarter turns
// that the frame fixes regroups the shots once, by the set each one fires.

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
// A frame plan is sized for the width where its frame begins.
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
// growth boundary suspends dispatch before an action can require more states,
// or before a frame rotation promotes past the width the states were sized for.
class SharingExecutor : public BatchClassicalState {
  public:
    SharingExecutor(const ExecutablePlan& plan, const PreparedBatchNoise& noise,
                    const EarlyPostselection& checks, uint32_t lane_capacity,
                    uint32_t initial_states, bool share = true, const SharingKey* key = nullptr,
                    uint64_t max_worker_state_bytes = kDefaultSharedWorkerStateBudget,
                    BatchOutputMode output_mode = BatchOutputMode::AggregateSurvivors);

    // Bytes one group holds within max_worker_state_bytes: a coefficient state
    // of the given width and, for a plan with a runtime Clifford frame, its
    // tableau.
    [[nodiscard]] static uint64_t group_storage_bytes(const ExecutablePlan& plan, uint32_t width);

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
    [[nodiscard]] uint64_t num_merges() const noexcept { return num_merges_; }

  private:
    using BatchClassicalState::execute_action;
    static constexpr uint32_t kNoShot = std::numeric_limits<uint32_t>::max();

    // Storage an action needs before dispatch can resume.
    struct StorageRequest {
        uint32_t states = 0;
        uint32_t width = 0;
        [[nodiscard]] explicit operator bool() const noexcept { return states != 0; }
    };

    // One frame tableau and what the action in progress has prepared for the
    // groups that share it.
    struct SharedFrame {
        enum class Axis : uint8_t { Identity, Dormant, Active };

        SharedFrame(uint32_t width, size_t sign_words)
            : frame(width), mapped(width), rows(sign_words) {}
        OnlineFrame frame;
        // Image of the action's Pauli, mapped before the action changes the frame.
        PauliString mapped;
        // Rows whose sign the action flips in the shots it selects.
        std::vector<uint64_t> rows;
        // Groups that refer to this tableau.
        uint32_t users = 0;
        // Value of frame_stamp_ when the fields below were prepared.
        uint32_t stamp = 0;
        // Conditional quarter turn: the tableau taking the shots that fire.
        // Measurement of the last active coordinate: the group collecting them.
        uint32_t link = 0;
        // Dormant pivot of a promotion or measurement, or the measured active
        // coordinate.
        uint32_t pivot = 0;
        ActivePauli active{};
        Axis axis = Axis::Identity;
        bool axis_sign = false;
        // True once the action's frame update has run.
        bool applied = false;
        // False while the frame still holds its reset rows.
        bool changed = false;
    };

    void initialize_groups(const SeedRoot& root, uint32_t first_shot, uint32_t shots);
    void grow_state_pool(StorageRequest request);
    // Returns an empty request on completion, otherwise what resuming needs.
    template <ExecutorBackend Backend>
    [[nodiscard]] StorageRequest dispatch_until_storage_boundary() noexcept;
    // Copies one group's coefficients into another's storage; the copy
    // refers to the same frame.
    void copy_group(uint32_t to, uint32_t from) noexcept;
    [[nodiscard]] uint32_t split(uint32_t group) noexcept;
    // Splits a group whose shots differ in the given bit and returns the group
    // that took the shots with a one, or kNoShot when every shot agrees.
    [[nodiscard]] uint32_t split_by_bit(uint32_t group, std::span<const uint64_t> values) noexcept;
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
    void execute_action(const ExecutablePlan::ExecuteFrameClifford& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    void execute_action(const ExecutablePlan::ExecuteFrameMeasurement& action) noexcept;

    // Rows of the frame that a stored frame Pauli selects: bit q for the image
    // of X_q and bit 64 * frame_words + q for the image of Z_q. A shot's sign
    // bits use the same layout.
    [[nodiscard]] std::span<const uint64_t> frame_pauli_rows(uint32_t pauli) const noexcept;
    [[nodiscard]] std::span<uint64_t> shot_signs(uint32_t shot) noexcept;
    // Whether the shot's frame maps the Pauli to minus its tableau's image.
    [[nodiscard]] bool shot_flips(uint32_t shot, std::span<const uint64_t> rows) const noexcept;
    void flip_shot_signs(uint32_t shot, std::span<const uint64_t> rows) noexcept;
    // Takes a free tableau, as a copy of another when one is given.
    [[nodiscard]] uint32_t acquire_frame(const SharedFrame* source) noexcept;
    void release_frame(uint32_t frame) noexcept;
    // Fills the frame's row mask with its rows that anticommute with p.
    void find_anticommuting_rows(SharedFrame& slot, PauliStringView p) noexcept;
    // Maps the rotation's Pauli through the group's frame, once per action.
    SharedFrame& prepare_frame_rotation(
        uint32_t group, const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    // True when some group's frame rotation would promote past state_width_.
    [[nodiscard]] bool frame_rotation_needs_width(
        const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    // Removes the groups whose shots another group took and frees their states.
    void drop_empty_groups() noexcept;
    // For the run of conditional quarter turns that starts at action_index_,
    // records which of them each live shot fires and sorts the shots of every
    // group into classes by that set. Returns the number of groups the run adds.
    [[nodiscard]] uint32_t prepare_clifford_run(uint32_t length) noexcept;
    // Moves each class of the prepared run to its own group and tableau.
    void execute_clifford_run(uint32_t length) noexcept;

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
    uint64_t num_merges_ = 0;
    // Width every pooled state is allocated for, and the budget that bounds
    // the pool when that width grows.
    uint32_t state_width_ = 0;
    uint64_t max_worker_state_bytes_ = 0;

    // Runtime Clifford frame storage; empty for a plan without one. The pool
    // holds one tableau per state, so every group can have its own.
    std::vector<SharedFrame> frames_;
    std::vector<uint32_t> frame_of_, free_frames_;
    uint32_t free_frame_count_ = 0;
    // Advances with every dispatched action.
    uint32_t frame_stamp_ = 0;
    size_t sign_stride_ = 0;
    std::vector<uint64_t> shot_signs_;
    std::vector<uint64_t> frame_random_;
    // Length of the run of conditional quarter turns fixed by the frame that
    // starts at each dispatch position, or zero. Empty when the plan has none.
    std::vector<uint32_t> clifford_runs_;
    // One bit per turn of the longest run for every lane: the turns it fires.
    std::vector<uint64_t> run_bits_;
    // A class is the shots of one tableau that fire the same set. It is named
    // by its first shot, which indexes the class tables below.
    std::vector<uint32_t> run_table_;
    // Per shot: its class, or kNoShot when it fires nothing.
    std::vector<uint32_t> run_class_;
    // Per class: the tableau it came from and the one it moves to.
    std::vector<uint32_t> class_source_, class_frame_;
    // Per class: the group that last counted it while preparing the run, the
    // group made for it, and the group whose shots that one took.
    std::vector<uint32_t> class_seen_by_, class_group_, class_group_owner_;
    // Per group: the class that keeps the group, or kNoShot when the shots
    // that fire nothing do.
    std::vector<uint32_t> run_stay_;
};

}  // namespace clifft::sampling
