#pragma once

// Query-private phase-aware extensions to sampling planning.

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/sampling/planner.h"

#include <complex>

namespace clifft::sampling {

struct PhaseAwareSamplingPlan {
    SamplingPlan plan;
    PhaseAwareCliffordFrame final_clifford_frame;
    std::complex<double> scalar{1.0, 0.0};
};

// Pure-unitary amplitude planning retains scalar rotations that ordinary
// projective planning can discard on known dormant eigenstates.
[[nodiscard]] PhaseAwareSamplingPlan plan_sampling_phase_aware(
    const HirModule& hir, PhaseAwareCliffordFrame final_clifford_frame);

}  // namespace clifft::sampling
