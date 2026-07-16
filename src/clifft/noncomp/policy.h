#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Exact includes the no-jump back-action from source-dependent transition
// rates. Neglect omits that back-action, which avoids the need to grow the
// active state vector (increase the active dimension) in cliffts factored
// state representation, but this ignores a shift of |p_g - p_e| between the
// computational ground and excited states that would result from exact back-action.
//
// Both policies are actually exact when the total transition rate is
// source-independent.
//
// Fired transitions retain their source and destination correlations under both
// settings.
enum class DampingPolicy : uint8_t {
    Exact = 0,
    Neglect = 1,
};

struct NonComputationalPolicy {
    // When true, a reset (R/RX/RY) on a lost qubit restores it to a
    // computational state. When false (default), the reset acts on a
    // vacated site and is dropped (identity on the surviving operands).
    bool reset_restores_lost = false;

    DampingPolicy damping = DampingPolicy::Exact;
};

}  // namespace clifft
