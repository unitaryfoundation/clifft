#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Policy for source-dependent transitions that fire on a
// ComputationalUnknown qubit. Reject refuses them (the exact behavior is
// not representable without runtime branching). EqualizeRates approximates
// them: every computational column is padded with a diagonal pseudo-jump
// so all columns share the maximum computational jump rate, the source is
// drawn uniformly over the computational levels, and the destination from
// that padded column. The pseudo-jump collapses the carrier without
// changing its level (pure dephasing). The approximation matches
// unbiased unknown-source computational marginals exactly; it discards
// the correlation between the destination and the collapse outcome, and
// a state that is gate-determined but tracked as unknown remains
// approximate (the source is still drawn uniformly).
enum class UnknownSourcePolicy : uint8_t {
    Reject = 0,
    EqualizeRates = 1,
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

struct NonComputationalPolicy {
    // When true, lost-qubit reset (R/RX/RY) restores the qubit to a
    // computational state. When false (default), lost-qubit reset
    // rejects (or drops, under LostLeakedOpsPolicy::Drop).
    bool reset_restores_lost = false;

    UnknownSourcePolicy unknown_source_policy = UnknownSourcePolicy::Reject;

    LostLeakedOpsPolicy lost_leaked_ops = LostLeakedOpsPolicy::Reject;
};

}  // namespace clifft
