#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

#include <span>

namespace clifft::sampling {

// Sampling pipeline:
//   optimized HirModule -> SamplingPlan -> ExecutablePlan -> Executor -> results
// The planner produces the executor-independent semantic action stream. CPU
// lowering then prepares fixed descriptors, and an Executor owns mutable state
// for one shot at a time. The sampler entry points drive repeated shots and
// collect their outputs.
struct SamplingPlanOptions {
    std::span<const uint8_t> postselection_mask;
    std::span<const uint8_t> expected_detectors;
    std::span<const uint8_t> expected_observables;
};

// EXP_VAL probes become plan actions. Exact final-state queries instead retain
// a coordinate map only for eligible unitary HIR. Instruments become explicit
// state actions followed by continuation boundaries.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options = {});

}  // namespace clifft::sampling
