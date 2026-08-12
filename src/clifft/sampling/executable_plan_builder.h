#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/util/runtime_isa.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft::sampling {

// Construction-only state for lowering a validated SamplingPlan into the
// fixed storage consumed by Executor.
class ExecutablePlanBuilder {
  public:
    static void build(ExecutablePlan& output, const SamplingPlan& source);

  private:
    ExecutablePlanBuilder(ExecutablePlan& output, const SamplingPlan& source);

    void compile();
    void initialize_program();
    void prepare_noise_and_boundaries();
    void lower_actions();
    void lower_action(const PlannedAction& planned);
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
    std::vector<uint32_t> expression_terms_;
    std::vector<uint32_t> expression_term_begins_;
    std::vector<uint32_t> boundary_noise_starts_;
    size_t boundary_index_ = 0;
};

}  // namespace clifft::sampling
