#pragma once
#include "clifft/circuit/circuit.h"
#include "clifft/sampling/planner.h"

#include <optional>
#include <string>

namespace clifft::sampling {
struct CssPlanningResult {
    std::optional<SamplingPlan> plan;
    std::string reason;
};
// Conservative source certification. A decline leaves ordinary compilation
// available; no state allocation or per-shot specialization occurs here.
[[nodiscard]] CssPlanningResult try_plan_css_blocks(const Circuit& circuit,
                                                    SamplingPlanOptions options = {},
                                                    unsigned minimum_data_width = 21);
// Compares certified factor work with the ordinary active-state traversals.
// This is a conservative routing heuristic, not a cross-family timing guarantee.
[[nodiscard]] bool prefer_css_blocks(const SamplingPlan& ordinary, const SamplingPlan& candidate);
}  // namespace clifft::sampling
