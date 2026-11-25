#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

#include <span>
#include <string_view>

namespace clifft::sampling {

// Builds the executor-independent sampling plan for the supported optimized
// HIR subset. The planner performs all stabilizer-coordinate changes and
// symbolic dependency discovery before execution.
struct SamplingPlanOptions {
    std::span<const uint8_t> postselection_mask{};
    std::span<const uint8_t> expected_detectors{};
    std::span<const uint8_t> expected_observables{};
    // Tooling can request an action-to-source sidecar when the HIR carries a
    // complete parallel source map. Sampling and ordinary lowering do not
    // require or retain it.
    bool retain_source_map = false;
    // Plan commuting T regions whose rotation signs depend on the shot on a
    // runtime Clifford frame when that needs fewer active coordinates than
    // signed rotations. Noise, random outcomes, and readout flips then change
    // a region by Cliffords instead of widening the coefficient state. HIR
    // with instruments or expectation probes keeps the static plan.
    bool runtime_frame = false;
};

// EXP_VAL probes become plan actions. Exact final-state queries instead retain
// a coordinate map only for eligible unitary HIR. Instruments become explicit
// state actions followed by continuation boundaries.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options = {});

// Why SamplingPlanOptions::runtime_frame cannot apply to this circuit, or an
// empty view when it can.
[[nodiscard]] std::string_view runtime_frame_unsupported_reason(const HirModule& hir) noexcept;

}  // namespace clifft::sampling
