#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

namespace clifft::sampling {

// Builds the executor-independent sampling plan for the supported optimized
// HIR subset. The planner performs all stabilizer-coordinate changes and
// symbolic dependency discovery before execution.
//
// The initial subset accepts T gates, phase rotations, and measurements. It
// throws std::invalid_argument for every other HIR operation.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir);

}  // namespace clifft::sampling
