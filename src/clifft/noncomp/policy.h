#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Policy for source-dependent transitions that fire on a
// ComputationalUnknown qubit. Reject refuses them: ahead-of-time
// sampling cannot pick a source column for a qubit with no definite
// level, so the run throws rather than approximate.
enum class UnknownSourcePolicy : uint8_t {
    Reject = 0,

    // Exact: transition firing moves to runtime. The circuit compiles
    // once with every annotation materialized as an instrument site; fire
    // probabilities are evaluated on the live state, and a fire that
    // cannot resolve in-line traps to the driver, which recompiles the
    // remaining circuit under the now-known status and resumes. Exact for
    // every source context; see DampingPolicy for the one knob that
    // trades exactness for rank at dormant-random sites.
    Exact = 1,
};

// Policy for operations touching a leaked or lost operand that have no
// representable effect there. Reject refuses them. Drop excises the whole
// operation (identity on the surviving operands): the physical
// interaction cannot happen at a vacated or leaked site. Measurements are
// never dropped -- their visible record slot is preserved and the
// classifier supplies the outcome.
enum class LostLeakedOpsPolicy : uint8_t {
    Reject = 0,
    Drop = 1,
};

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
    // When true, lost-qubit reset (R/RX/RY) restores the qubit to a
    // computational state. When false (default), lost-qubit reset
    // rejects (or drops, under LostLeakedOpsPolicy::Drop).
    bool reset_restores_lost = false;

    UnknownSourcePolicy unknown_source_policy = UnknownSourcePolicy::Reject;

    LostLeakedOpsPolicy lost_leaked_ops = LostLeakedOpsPolicy::Reject;

    DampingPolicy damping = DampingPolicy::Exact;
};

}  // namespace clifft
