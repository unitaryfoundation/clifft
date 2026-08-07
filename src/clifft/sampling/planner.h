#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

#include <span>

namespace clifft::sampling {

// Builds the executor-independent sampling plan for the supported optimized
// HIR subset. The planner performs all stabilizer-coordinate changes and
// symbolic dependency discovery before execution.
//
struct SamplingPlanOptions {
    std::span<const uint8_t> postselection_mask;
    std::span<const uint8_t> expected_detectors;
    std::span<const uint8_t> expected_observables;
};

// Exact-state probes remain outside this sampling-only plan. Instruments lower
// to explicit state actions and continuation boundaries.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options = {});

}  // namespace clifft::sampling
