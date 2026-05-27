#pragma once

// Policy knobs for the noncomputational trajectory MVP. See
// design/noncomputational-mvp.md sections 4.2 and 5.2.

#include <cstdint>

namespace clifft {

// How the runtime sampler handles a source-dependent transition that
// fires on a ComputationalUnknown qubit. The MVP exposes only Reject;
// the EqualizeRates compat mode (section 9 of the design note) is
// future work but the enum slot is reserved now so adding it later
// does not break the ABI.
enum class UnknownSourcePolicy : uint8_t {
    Reject = 0,
    // EqualizeRates = 1,  // reserved for future sqale-compat approximation
};

struct NonComputationalPolicy {
    // When true, lost-qubit reset (R/RX/RY) restores the qubit to a
    // computational state per the section 5.2 table. When false
    // (default), lost-qubit reset rejects.
    bool reset_restores_lost = false;

    UnknownSourcePolicy unknown_source_policy = UnknownSourcePolicy::Reject;
};

}  // namespace clifft
