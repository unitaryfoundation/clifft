#pragma once

#include "clifft/sampling/plan.h"
#include "clifft/sampling/state.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

// The X masks partition computational-basis states into sets
// {|b xor x> : x is in their binary span}; each such set is called an orbit.
// This descriptor applies one precomposed unitary within every orbit generated
// by at most two X masks. Z parities of the orbit representative select the
// matrix, so each coefficient is still loaded and stored exactly once.
struct PreparedFusedRotation {
    uint32_t active_width = 0;
    uint32_t orbit_rank = 0;
    // The orbit and selector masks are reduced GF(2) bases in ascending pivot
    // order; orbit_pivots identifies the bits omitted from representatives.
    std::array<uint64_t, 2> orbit_masks{};
    std::array<uint32_t, 2> orbit_pivots{};
    std::vector<uint64_t> selector_masks;
    std::vector<std::complex<double>> matrices;
};

// Describes the maximal rank-two-eligible constant-sign rotation run beginning
// at the supplied action. A populated rotation replaces all action_count
// inputs; otherwise the caller lowers that many actions individually.
struct FusedRotationRun {
    size_t action_count = 0;
    std::optional<PreparedFusedRotation> rotation;
};

// A low-rank rotation run whose matrix depends on a small basis of per-shot
// affine signs. Each sign assignment selects one ordinary prepared fused
// rotation, so execution can reuse the existing kernels.
struct PreparedDynamicFusedRotation {
    std::vector<AffineBool> sign_basis;
    std::vector<PreparedFusedRotation> variants;
};

struct DynamicFusedRotationRun {
    size_t action_count = 0;
    std::optional<PreparedDynamicFusedRotation> rotation;
};

[[nodiscard]] FusedRotationRun prepare_fused_rotation_run(std::span<const PlannedAction> actions);
[[nodiscard]] DynamicFusedRotationRun prepare_dynamic_fused_rotation_run(
    std::span<const PlannedAction> actions);
void apply_fused_rotation(State& state, const PreparedFusedRotation& rotation) noexcept;

}  // namespace clifft::sampling
