#pragma once

// Policy knobs for the noncomputational trajectory model.

#include <cstdint>

namespace clifft {

// Policy for transitions that fire on a ComputationalUnknown qubit.
enum class UnknownSourcePolicy : uint8_t {
    Reject = 0,
};

struct NonComputationalPolicy {
    // When true, lost-qubit reset (R/RX/RY) restores the qubit to a
    // computational state. When false (default), lost-qubit reset
    // rejects.
    bool reset_restores_lost = false;

    UnknownSourcePolicy unknown_source_policy = UnknownSourcePolicy::Reject;
};

}  // namespace clifft
