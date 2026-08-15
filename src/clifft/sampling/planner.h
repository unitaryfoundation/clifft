#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

#include <span>

namespace clifft::sampling {

// Builds the executor-independent sampling plan for the supported optimized
// HIR subset. The planner performs all stabilizer-coordinate changes and
// symbolic dependency discovery before execution.
struct SamplingPlanOptions {
    std::span<const uint8_t> postselection_mask;
    std::span<const uint8_t> expected_detectors;
    std::span<const uint8_t> expected_observables;
    // Tooling can request an action-to-source sidecar when the HIR carries a
    // complete parallel source map. Sampling and ordinary lowering do not
    // require or retain it.
    bool retain_source_map = false;
};

// EXP_VAL probes become plan actions. Exact final-state queries instead retain
// a coordinate map only for eligible unitary HIR. Instruments become explicit
// state actions followed by continuation boundaries.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options = {});

}  // namespace clifft::sampling
