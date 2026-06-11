#pragma once

// Circuit rewriter for the noncomputational history layer.
//
// rewrite(original, history, model) replays one sampled trajectory and
// produces a new ordinary clifft::Circuit ready for the existing compile
// pipeline -- it introduces no new instruction kinds. It does three things:
//
//   1. Initial-state prep: prepend an X on each qubit whose sampled initial
//      status is ComputationalKnown with a basis_bit == One level, so the
//      SVM's |0...0> initial state matches the sampled known level.
//   2. Per-op policy: replay each operation's per-qubit status through the
//      shared status stepper and keep, drop, or reject the operation. A
//      measurement is always kept so the visible measurement record (and
//      every rec[-k] reference into it) is preserved; reinterpreting a
//      measurement's outcome for a leaked or lost qubit is a sampling /
//      sidecar concern for the orchestrator, not a circuit edit. An
//      ambiguous operation on a leaked or lost operand rejects by default;
//      under LostLeakedOpsPolicy::Drop it is excised whole (identity on the
//      surviving operands), whose statuses then keep their entry values.
//   3. Hidden trace-out: when a coherent qubit jumps to a Leaked or Lost
//      level, insert an R on that qubit immediately after the operation.
//      The existing reset lowering turns it into a hidden measurement plus a
//      corrective Pauli; it adds no visible measurement and shifts no record
//      index. "Coherent" means the carrier state the base operation would
//      leave with no jump is ComputationalUnknown -- a qubit a gate has just
//      made coherent still needs trace-out even if it entered known.
//   4. Carrier materialization: when a jump lands on a computational level,
//      insert an R (plus an X for the basis_bit == One level) immediately
//      after the operation, so the SVM carrier is prepared at the definite
//      destination level. This is done for every carrier state: it is the
//      collapse unraveling for a coherent carrier, a deterministic re-prep
//      for a known one, and it rezeros a stale residual when a leaked or
//      lost qubit is recaptured. Like the trace-out, it shifts no record
//      index.
//
// Transition outcomes are consumed from history.transitions in the same
// order the sampler produced them (one record per Physical operand of a gate
// that declares a transition). The rewriter does not sample, compile, or run
// the SVM.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/history.h"
#include "clifft/noncomp/model.h"

namespace clifft {

// Produce the rewritten circuit for `original` under `history` and `model`.
// Throws std::invalid_argument when the trajectory policy rejects an
// operation (naming the operation index, qubit, gate, and status) or when
// `history` does not describe `original` (qubit count or transition count
// mismatch).
//
// Precondition: `original` is a parser-normalized circuit -- each
// single-qubit operation is a single-target node and each two-qubit
// operation is a single pair. The keep / drop / reject decision is made per
// node, so a hand-built node that packs several single-qubit operands would
// be dropped or rejected as a whole rather than per operand.
Circuit rewrite(const Circuit& original, const NonComputationalHistory& history,
                const NonComputationalModel& model);

}  // namespace clifft
