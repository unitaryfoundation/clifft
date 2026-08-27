#pragma once

// Query-private phase-aware extensions to sampling planning.

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/sampling/planner.h"

#include <complex>
#include <optional>
#include <vector>

namespace clifft::sampling {

struct PhaseAwareScalarRotation {
    double half_turns = 0.0;
    AffineBool sign;
};

struct PhaseAwareSamplingPlan {
    SamplingPlan plan;
    PhaseAwareCliffordFrame final_clifford_frame;
    std::optional<Tableau> final_tableau;
    std::complex<double> scalar{1.0, 0.0};
    std::vector<PhaseAwareScalarRotation> scalar_rotations;
};

// Exact amplitude planning retains scalar rotations that ordinary projective
// planning can discard on known dormant eigenstates.
[[nodiscard]] PhaseAwareSamplingPlan plan_sampling_phase_aware(
    const HirModule& hir, PhaseAwareCliffordFrame final_clifford_frame);

}  // namespace clifft::sampling
