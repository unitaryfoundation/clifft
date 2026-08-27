#pragma once

// Query-private phase-aware extensions to sampling planning.

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/sampling/planner.h"

#include <complex>
#include <cstdint>
#include <span>

namespace clifft::sampling {

struct PhaseAwareSamplingPlan {
    SamplingPlan plan;
    PhaseAwareCliffordFrame final_clifford_frame;
    Tableau final_tableau;
    std::complex<double> scalar{1.0, 0.0};
};

// Exact amplitude planning retains forced branches and scalar rotations that
// ordinary projective planning can discard.
[[nodiscard]] PhaseAwareSamplingPlan plan_sampling_phase_aware(
    const HirModule& hir, PhaseAwareCliffordFrame final_clifford_frame,
    std::span<const uint8_t> forced_effect_records);

}  // namespace clifft::sampling
