#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/util/runtime_isa.h"

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

    void compile();
    void initialize_program();
    void prepare_noise_and_boundaries();
    void lower_actions();
    void lower_action(const PlannedAction& planned, size_t& boundary_index);
    void build_expression_dependencies();
    void validate_executable_plan() const;
    [[nodiscard]] size_t estimate_expression_terms() const;

    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_expression(
        const AffineBool& expression);
    [[nodiscard]] ExecutablePlan::PreparedExpression prepare_measurement_correction(
        const AffineBool& outcome, uint32_t branch);
    void ensure_expression_term_capacity(size_t additional_terms) const;

    ExecutablePlan& output_;
    const SamplingPlan& source_;
    clifft::internal::RuntimeIsa runtime_isa_ = clifft::internal::RuntimeIsa::Scalar;
    // Retain action-order terms only until the dependency CSR is complete.
    std::vector<uint32_t> expression_terms_;
    std::vector<uint32_t> expression_term_begins_;
    std::vector<uint32_t> boundary_noise_starts_;
};

}  // namespace clifft::sampling
