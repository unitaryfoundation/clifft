#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Exact applies the no-jump back-action for source-dependent transition
// rates. On a coherent dormant qubit, this may promote the qubit into the
// active state and increase the active width. Neglect avoids that cost by
// omitting the back-action, introducing a survivorship tilt of order
// |p_g - p_e|. It remains exact when the total transition rate is
// source-independent. Fired transitions retain their source and destination
// correlations under both settings.
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
