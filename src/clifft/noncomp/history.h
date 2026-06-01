#pragma once

// NonComputationalHistory: the canonical, minimal record of one sampled
// trajectory through a circuit.
//
// It stores only the *random* facts a replay needs to be deterministic:
//   - the initial sampled status of each qubit, and
//   - the outcome of every transition instrument that was consulted, in
//     circuit-walk order.
//
// Everything else (each qubit's status at each operation) is a derived
// view: replay the circuit through the same status stepper, consuming
// these outcomes in order. Per-operation statuses are intentionally not
// materialized here, and transition outcomes are not part of the
// user-facing sample result -- they belong to optional sidecar/debug
// output.

#include "clifft/noncomp/qubit_status.h"

#include <cstdint>
#include <vector>

namespace clifft {

// One consulted transition instrument: the sampled outcome for a single
// (operation, qubit operand). Recorded whether or not a jump occurred,
// so a replay consumes exactly one record per consulted operand.
struct TransitionRecord {
    uint32_t op_index;          // index into Circuit::nodes
    uint32_t qubit;             // qubit operand id
    bool jumped;                // true if the instrument fired a jump
    uint8_t destination_level;  // jump target; kInvalidLevel when !jumped
};

struct NonComputationalHistory {
    // Sampled initial status, one entry per qubit (index == qubit id).
    std::vector<QubitStatus> initial_status;

    // Consulted transition outcomes, in circuit-walk order.
    std::vector<TransitionRecord> transitions;
};

}  // namespace clifft
