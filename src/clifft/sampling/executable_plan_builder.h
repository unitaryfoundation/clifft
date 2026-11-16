#pragma once

#include "clifft/sampling/executable_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace clifft::sampling {

// One-shot construction context that lowers SamplingPlan actions, affine
// expressions, noise sites, and continuation boundaries into fixed descriptors
// and dependency storage consumed by Executor. Keeping its temporary term and
// boundary vectors here avoids retaining build scratch in the immutable
// ExecutablePlan; the friend-only entry point prevents rebuilding a finalized
// plan before this context is discarded.
class ExecutablePlanBuilder {
  private:
    friend class ExecutablePlan;

    struct ProgramStorageEstimate {
        size_t expression_terms = 0;
        bool has_postselection = false;
    };

    static ExecutablePlan compact_outputs(const ExecutablePlan& input, SamplingOutputs outputs,
                                          bool early_postselection, bool survivor_counts);
    static void build(ExecutablePlan& output, const SamplingPlan& source);
    ExecutablePlanBuilder(ExecutablePlan& output, const SamplingPlan& source);

    // Coordinates the one-shot lowering stages below.
    void compile();

    // Reserve fixed program storage and initialize plan-wide indices.
    void initialize_program();

    // Prepare presampled distributions and continuation segment boundaries.
    void prepare_noise_and_boundaries();

    // Scan semantic actions once, selecting CPU descriptors and bounded
    // adjacent-rotation fusion without introducing a general pass pipeline.
    void lower_action_stream();
    void lower_action(const PlannedAction& planned, size_t& boundary_index);
    static batch_detail::BatchWorkEstimate action_batch_lane_work(
        const ExecutablePlan& plan, const ExecutablePlan::Action& action, uint32_t width);
    void record_batch_lane_work(batch_detail::BatchWorkEstimate work);
    // Appends one frame Pauli to the plan's packed storage and returns its index.
    [[nodiscard]] uint32_t store_frame_pauli(const FramePauli& pauli);

    // The leading run of frame actions that are mutually commuting conditional
    // Cliffords. A Pauli commutes with every earlier one exactly when it
    // commutes with a basis of their span, kept here in echelon form with each
    // row holding X words followed by Z words.
    struct CommutingFramePrefix {
        bool open = true;
        std::vector<std::vector<uint64_t>> basis;
        std::vector<uint32_t> pivots;

        // Adds the Pauli and returns true, or closes the run and returns false.
        [[nodiscard]] bool extend(const FramePauli& pauli);
    };
    void record_action_origin(uint32_t plan_begin, uint32_t plan_end);
    static void prepare_batch_compaction_costs(ExecutablePlan& output,
                                               std::span<const batch_detail::BatchLaneWork> work);

    // Transpose action-order affine terms into symbol-to-register CSR storage.
    void build_expression_dependencies();
    static void prepare_noise_schedule(ExecutablePlan& output,
                                       std::span<const uint32_t> first_actions);

    // Check construction-only invariants in Debug builds.
    void validate_executable_plan() const;
    [[nodiscard]] ProgramStorageEstimate estimate_program_storage() const;

    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_expression(
        const AffineBool& expression);
    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_measurement_correction(
        const AffineBool& outcome, uint32_t branch);
    [[nodiscard]] ExecutablePlan::PreparedRecordParity prepare_record_parity(
        const RecordParity& parity);
    [[nodiscard]] ExecutablePlan::PreparedObservableValue prepare_observable_value(
        const ObservableValue& value);

    // Deduplicates affine expressions into shared register storage while
    // retaining action-order terms only until the dependency CSR is built.
    class ExpressionInterner {
      public:
        explicit ExpressionInterner(std::vector<uint8_t>& constants);

        void reserve(size_t expected_expressions, size_t expected_terms);
        [[nodiscard]] uint32_t prepare_constant(bool constant);
        [[nodiscard]] uint32_t prepare(const AffineBool& expression, uint32_t action_index,
                                       std::span<uint32_t> symbol_first_actions);
        [[nodiscard]] uint32_t prepare_measurement_correction(
            const AffineBool& outcome, uint32_t branch, uint32_t action_index,
            std::span<uint32_t> symbol_first_actions);
        [[nodiscard]] std::span<const uint32_t> terms() const noexcept { return terms_; }
        [[nodiscard]] std::span<const uint32_t> term_begins() const noexcept {
            return term_begins_;
        }
        [[nodiscard]] size_t num_registers() const noexcept { return term_begins_.size(); }

      private:
        static constexpr uint32_t kNoRegister = std::numeric_limits<uint32_t>::max();
        struct Slot {
            uint64_t hash = 0;
            uint32_t register_id = kNoRegister;
        };

        void ensure_term_capacity(size_t additional_terms) const;
        [[nodiscard]] uint32_t intern_appended(bool constant, uint32_t tentative_begin,
                                               uint64_t hash);

        std::vector<uint8_t>& constants_;
        size_t initial_table_hint_ = 16;
        std::vector<uint32_t> terms_;
        std::vector<uint32_t> term_begins_;
        std::array<uint32_t, 2> constant_registers_{kNoRegister, kNoRegister};
        std::vector<Slot> table_;
        size_t num_interned_ = 0;
    };

    ExecutablePlan& output_;
    const SamplingPlan& source_;
    ExecutorBackend backend_ = ExecutorBackend::Scalar;
    ExpressionInterner expressions_;
    std::vector<uint32_t> boundary_noise_starts_;
    CommutingFramePrefix frame_prefix_;
    std::vector<uint8_t> bound_presampled_symbols_;
    std::vector<uint32_t> symbol_first_actions_;
    batch_detail::BatchWorkEstimate estimated_batch_lane_work_;
    std::vector<batch_detail::BatchLaneWork> action_batch_lane_work_;
};

}  // namespace clifft::sampling
