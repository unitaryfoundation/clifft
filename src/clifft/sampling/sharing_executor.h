#pragma once

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/tuning.h"
#include "clifft/sampling/online_frame.h"
#include "clifft/sampling/state.h"
#include "clifft/util/page_allocation.h"

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace clifft::sampling {

// Shared-trajectory workers combine BatchClassicalState's per-shot classical
// data with SharingExecutor's groups and pool of coefficient states. An optional
// SharingKey, prepared before execution, groups shots upfront by noise effects
// on future rotation signs. SharingExecutor applies prepared actions to each
// group and splits it when needed.
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
// its last active coordinate is measured, and all of them when the plan
// retires the frame. A run of consecutive quarter turns that the frame fixes
// regroups the shots once, by the set each one fires.

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

// True when shared execution groups the plan's shots: they share coefficient
// states, or the tableau of a runtime Clifford frame. Other plans run on the
// packed executor.
[[nodiscard]] inline bool shares_groups(const ExecutablePlan& plan) noexcept {
    return plan.peak_active_width() != 0 || plan.uses_runtime_frame();
}

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
// The first candidate is the baseline a tuned call falls back to: the capacity
// given, or the automatic one for safe_capacity.
[[nodiscard]] std::vector<batch_detail::BatchTuningCandidate> shared_tuning_candidates(
    uint32_t safe_capacity, uint32_t shots, uint32_t workers, uint64_t output_row_bytes = 0,
    std::optional<uint32_t> baseline_capacity = std::nullopt);

// Thrown at a storage boundary when the groups of a batch need more
// coefficient storage than max_worker_state_bytes holds. Fewer lanes make
// fewer groups, so the lanes of the batch can run again in smaller ranges.
class SharedStateBudgetError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Immutable noise-sign basis shared by workers. Branches affecting this basis
// split groups as they are sampled, keeping every later rotation sign uniform.
class SharingKey {
  public:
    explicit SharingKey(const ExecutablePlan& plan);
    [[nodiscard]] size_t num_bits() const noexcept { return registers_.size(); }
    // False when no rotation sign depends on a symbol: the key then puts every
    // shot in one group and never splits one.
    [[nodiscard]] bool separates_groups() const noexcept { return separates_groups_; }

  private:
    friend class SharingExecutor;
    const ExecutablePlan* plan_;
    std::vector<uint32_t> registers_;
    std::vector<uint8_t> split_symbols_;
    bool separates_groups_ = false;
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

    // Draws the batch whole and runs lanes [first_lane, first_lane +
    // lane_count) of it. A lane's row does not depend on which other lanes
    // run, so ranges that cover the batch return the rows of the whole batch.
    void run_batch(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                   uint32_t first_lane = 0,
                   uint32_t lane_count = std::numeric_limits<uint32_t>::max());
    [[nodiscard]] uint32_t state_capacity() const noexcept {
        return static_cast<uint32_t>(states_.size());
    }
    [[nodiscard]] uint64_t state_pool_bytes() const noexcept;
    // Returns the coefficient storage to one state of the width a new
    // executor starts with. Call only between batches.
    void reset_state_pool();
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
        // Rows whose sign a measurement branch flips.
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

    // Tableau bytes of one group of a plan with a runtime Clifford frame.
    [[nodiscard]] static uint64_t frame_storage_bytes(const ExecutablePlan& plan);
    // A conditional quarter turn the frame fixes. Consecutive ones commute,
    // so a run of them regroups the shots once.
    [[nodiscard]] static bool is_run_turn(const ExecutablePlan::Action& action) noexcept;

    // Runs the lanes on the pool as it is. Returns false when the pool
    // cannot grow to hold their groups within the budget.
    [[nodiscard]] bool run_batch_once(const SeedRoot& root, uint32_t first_shot, uint32_t shots,
                                      uint32_t first_lane, uint32_t lane_count);
    [[nodiscard]] bool initialize_groups(const SeedRoot& root, uint32_t first_shot, uint32_t shots);
    // Grows the pool, or returns false when it cannot within the budget.
    [[nodiscard]] bool grow_state_pool(StorageRequest request);
    [[noreturn]] void throw_over_budget() const;
    // Slots a pool of old_count grows to for a request, up to max_states.
    [[nodiscard]] static uint32_t grown_states(uint32_t old_count, uint32_t required,
                                               uint32_t max_states) noexcept;
    // Returns an empty request on completion, otherwise what resuming needs.
    template <ExecutorBackend Backend>
    [[nodiscard]] StorageRequest dispatch_until_storage_boundary() noexcept;
    void copy_state(uint32_t to, uint32_t from) noexcept;
    [[nodiscard]] std::span<const uint32_t> lanes(uint32_t group) const noexcept {
        return {order_.data() + begin_[group], end_[group] - begin_[group]};
    }
    void swap_lanes(uint32_t first, uint32_t second) noexcept;
    // Gives the lanes at [tail, end) of a group their own state. Returns the
    // new group, or kNoShot when the tail is empty or holds the whole group.
    uint32_t detach_tail(uint32_t group, uint32_t tail) noexcept;
    // Separates lanes by choices_; lanes that chose one form the new group.
    [[nodiscard]] uint32_t split(uint32_t group) noexcept;
    // Splits groups until the sign is uniform within each one.
    void split_by_sign(std::span<const uint64_t> values) noexcept;
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
    void execute_action(const ExecutablePlan::ExecuteFrameRetirement& action) noexcept;

    // Rows of the frame that a stored frame Pauli selects: bit q for the image
    // of X_q and bit 64 * frame_words + q for the image of Z_q. A shot's sign
    // bits use the same layout.
    [[nodiscard]] std::span<const uint64_t> frame_pauli_rows(uint32_t pauli) const noexcept;
    // Whether the shot's frame maps the Pauli to minus its tableau's image.
    [[nodiscard]] bool shot_flips(uint32_t shot, std::span<const uint64_t> rows) const noexcept;
    void flip_shot_signs(uint32_t shot, std::span<const uint64_t> rows) noexcept;
    // Takes a free tableau, as a copy of another when one is given.
    [[nodiscard]] uint32_t acquire_frame(const SharedFrame* source) noexcept;
    void release_frame(uint32_t frame) noexcept;
    // Maps the rotation's Pauli through the group's frame, once per action.
    SharedFrame& prepare_frame_rotation(
        uint32_t group, const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    // True when some group's frame rotation would promote past state_width_.
    [[nodiscard]] bool frame_rotation_needs_width(
        const ExecutablePlan::ExecuteFrameRotation& action) noexcept;
    // Frees the groups whose lanes another group took and lays every group
    // out again as a range of order_.
    void collect_merged_groups() noexcept;
    // For the run of conditional quarter turns that starts at action_index_,
    // records which of them each live shot fires and names the class of every
    // shot that fires any: the shots of one tableau that fire the same set.
    // Returns the number of groups the run adds.
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
    // Contiguous slabs outlive the non-owning State views in states_.
    std::vector<PageAlignedAllocation> state_chunks_;
    std::vector<State> states_;
    std::vector<Xoshiro256PlusPlus> shot_rng_;
    // Every group is the range [begin_, end_) of order_, a permutation of the
    // lanes, so a split moves the departing lanes to the tail of the range and
    // cuts it there. position_ inverts order_. departing_ counts the lanes a
    // sign split has moved to a group's tail and is zero between splits.
    std::vector<uint32_t> order_, position_, group_of_;
    std::vector<uint32_t> begin_, end_, departing_;
    std::vector<uint32_t> groups_, free_;
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
    // Bytes of the request the pool last refused, and whether it asked for
    // wider states.
    uint64_t refused_bytes_ = 0;
    bool refused_widening_ = false;

    // Runtime Clifford frame storage; empty for a plan without one. The pool
    // holds one tableau per state, so every group can have its own.
    std::vector<SharedFrame> frames_;
    std::vector<uint32_t> frame_of_, free_frames_;
    uint32_t free_frame_count_ = 0;
    // Advances with every dispatched action.
    uint32_t frame_stamp_ = 0;
    // Sign bits per lane, stored lane by lane.
    size_t sign_stride_ = 0;
    std::vector<uint64_t> shot_signs_;
    std::vector<uint64_t> frame_random_;
    // Groups that hold a lane on which the conditional quarter turn in
    // progress fires, and the rows that turn changes.
    std::vector<uint32_t> fired_groups_;
    std::vector<uint64_t> changed_rows_;
    // Length of the run of conditional quarter turns fixed by the frame that
    // starts at each dispatch position, or zero. Empty when the plan has none.
    std::vector<uint32_t> clifford_runs_;
    // For each dispatch position in a run, the offset in that run of the first
    // turn about the same Pauli. Turns about one Pauli compose, so a shot
    // records them on the first turn's bit: what is left is that turn or
    // nothing, and possibly the Pauli itself.
    std::vector<uint32_t> run_first_;
    // True at the first position of a run that turns twice about some Pauli.
    std::vector<uint8_t> run_repeats_;
    // One bit per turn of the longest run for every lane: set when the shot
    // applies that turn, after turns about the same Pauli have composed.
    std::vector<uint64_t> run_bits_;
    // The same layout for the Paulis left over. Empty when no run repeats one.
    std::vector<uint64_t> run_paulis_;
    // A class is named by its first shot, which indexes the class tables.
    std::vector<uint32_t> run_table_;
    // Per shot: its class, or kNoShot when it fires nothing.
    std::vector<uint32_t> run_class_;
    // Per class: the tableau it came from and the one it moves to.
    std::vector<uint32_t> class_source_, class_frame_;
    // Per class: the group that last counted it while preparing the run, the
    // group being regrouped that holds some of its shots, how many, and the
    // group made for them.
    std::vector<uint32_t> class_seen_by_, class_owner_, class_size_, class_group_;
    // Per group: the class that keeps the group, or kNoShot when the shots
    // that fire nothing do.
    std::vector<uint32_t> run_stay_;
    // The lanes of the group being regrouped, and the classes leaving it.
    std::vector<uint32_t> run_lanes_, run_leaving_;
};

}  // namespace clifft::sampling
