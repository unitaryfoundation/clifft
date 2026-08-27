#pragma once

#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>
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
    void record_action_origin(uint32_t plan_begin, uint32_t plan_end);
    void prepare_batch_compaction_costs();

    // Transpose action-order affine terms into symbol-to-register CSR storage.
    void build_expression_dependencies();

    // Check construction-only invariants in Debug builds.
    void validate_executable_plan() const;
    [[nodiscard]] size_t estimate_expression_terms() const;

    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_expression(
        const AffineBool& expression);
    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_measurement_correction(
        const AffineBool& outcome, uint32_t branch);
    [[nodiscard]] ExecutablePlan::PreparedRecordParity prepare_record_parity(
        const RecordParity& parity);
    [[nodiscard]] ExecutablePlan::PreparedObservableValue prepare_observable_value(
        const ObservableValue& value);
    void ensure_expression_term_capacity(size_t additional_terms) const;

    ExecutablePlan& output_;
    const SamplingPlan& source_;
    ExecutorBackend backend_ = ExecutorBackend::Scalar;
    // Retain action-order terms only until the dependency CSR is complete.
    std::vector<uint32_t> expression_terms_;
    std::vector<uint32_t> expression_term_begins_;
    std::vector<uint32_t> boundary_noise_starts_;
    std::vector<uint8_t> bound_presampled_symbols_;
    std::vector<uint64_t> action_batch_lane_work_;
};

}  // namespace clifft::sampling
