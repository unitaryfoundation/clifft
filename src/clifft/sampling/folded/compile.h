#pragma once

#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"

namespace clifft::sampling::folded {

// Recognition supplies a terminal gate pattern and unbound contraction tables.
// Boundary certification and integration with ordinary records/noise happen here.
std::optional<ExecutablePlan> compile_region(const std::string& source, const std::string& prefix,
                                             const std::string& block,
                                             const std::vector<std::string>& contractions,
                                             const std::vector<size_t>& dimensions,
                                             SamplingPlanOptions options, bool normalize_syndromes,
                                             HirPassManager* passes);

}  // namespace clifft::sampling::folded
