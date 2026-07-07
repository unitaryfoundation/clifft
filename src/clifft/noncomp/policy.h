#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Damping policy for exact-mode compilation at sites where the no-fire
// back-action is genuinely non-Clifford (a source-dependent transition on
// a dormant qubit with a random outcome). Exact expands the qubit into
// the amplitude array (+1 to the circuit's k at that site) and applies
// the damp there. Neglect keeps k stable by omitting the no-fire
// back-action -- a pure survivorship tilt of order |p_g - p_e|, with no
// effect at all on source-independent rates -- and that omission is its
// only approximation. Every fire at such a site traps with the drawn
// source recorded and its destination still undrawn, and the exact-mode
// driver's continuation collapses the carrier onto that source (a
// trace-out forced to the reported outcome) before applying the drawn
// destination's effects, keeping fire-side correlations exact. Sites
// where the qubit is active or deterministic are exact under both
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
