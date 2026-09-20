#pragma once

#include "clifft/frontend/hir.h"

#if __has_include("clifft/optimizer/active_width_schedule_pass.h")
#include "clifft/optimizer/active_width_schedule_pass.h"
#endif

#include <stdexcept>
#include <string>

// Research builds can compare an optional optimizer without changing defaults.
inline void apply_study_schedule(clifft::HirModule& hir, const std::string& mode) {
    if (mode == "off")
        return;
    if (mode != "budgeted" && mode != "unbounded")
        throw std::invalid_argument("schedule must be off, budgeted, or unbounded");
#if __has_include("clifft/optimizer/active_width_schedule_pass.h")
    clifft::ActiveWidthScheduleOptions options;
    if (mode == "unbounded")
        options.search_budget = std::nullopt;
    clifft::ActiveWidthSchedulePass pass(options);
    pass.run(hir);
#else
    (void)hir;
    throw std::invalid_argument("this build does not contain the scheduling pass");
#endif
}
