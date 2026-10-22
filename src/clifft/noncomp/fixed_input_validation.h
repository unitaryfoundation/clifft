#pragma once

#include "clifft/frontend/frontend.h"
#include "clifft/sampling/plan.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::detail {

struct FixedReadoutSite {
    size_t node;
    uint32_t record;
};

// External inputs are ordered by lowered circuit sites. Check both boundaries
// before installing them: equal counts or equal record targets do not establish
// identity, particularly for consecutive channels on the same record.
void validate_fixed_input_bindings(std::span<const size_t> noise_nodes,
                                   std::span<const FixedReadoutSite> readout_sites,
                                   const TraceSiteOrigins& origins, const HirModule& hir,
                                   const sampling::SamplingPlan& plan);

}  // namespace clifft::detail
