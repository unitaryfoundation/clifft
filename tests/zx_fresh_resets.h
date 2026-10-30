#pragma once

// Remove qubit reuse: continue every reset on a fresh qubit.
//
// A reset discards a qubit and prepares a new one, so after `R q` (and the
// reset half of `MR q`, and their X and Y variants) the circuit can continue
// on a new qubit in the same state while the old one is never touched again.
// This is the same channel on the rest of the system and leaves the
// measurement record unchanged.
//
// The ZX builder performs the same rewrite on traced HIR, where a reset of a
// used qubit is a measurement plus a correction; this circuit-level version is
// the independent reference its tests compare against.

#include "clifft/circuit/circuit.h"

namespace zx_test {

using clifft::Circuit;

// Throws std::invalid_argument for leakage and loss annotations, whose state
// survives a reset.
[[nodiscard]] Circuit with_fresh_qubit_per_reset(const Circuit& circuit);

}  // namespace zx_test
