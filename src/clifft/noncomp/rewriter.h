#pragma once

// Circuit rewriter for the noncomputational history layer.
//
// rewrite(original, history, model) replays one sampled trajectory and
// produces a new ordinary clifft::Circuit ready for the existing compile
// pipeline -- it introduces no new instruction kinds. It does five things:
//
//   1. Initial-state prep: prepend an X on each qubit whose sampled initial
//      status is ComputationalKnown with the |1> level, so the
//      SVM's |0...0> initial state matches the sampled known level.
//   2. Per-op policy: replay each operation's per-qubit status through the
//      shared status stepper and keep, drop, or reject the operation. An
//      ambiguous operation on a leaked or lost operand rejects by default;
//      under LostLeakedOpsPolicy::Drop it is excised whole (identity on the
//      surviving operands), whose statuses then keep their entry values.
//   3. Classifier record write: a measurement on a leaked or lost qubit is
//      not a physical Born measurement -- the model's classifier defines its
//      record bit. The measurement node is replaced by an MPAD writing the
//      same visible record slot, so the record layout and every rec[-k]
//      reference are preserved and the bit reaches the SVM's
//      detector/observable evaluation. A stochastic classifier column adds a
//      READOUT_NOISE on that slot so the bit is drawn at sample time inside
//      the VM; a deterministic column pads the literal bit with no draw. A
//      measure-and-reset additionally keeps its reset as a separate node.
//      Each such replacement is reported in the result's
//      classified_measurements (in slot order) so callers can post-process
//      per-slot classifier behavior -- e.g. the herald pass for three-symbol
//      classifiers, which re-points a heralded slot's flip probability at
//      one half. For a three-symbol column the emitted probability is the
//      bit's not-heralded conditional, and a READOUT_NOISE node is always
//      emitted so a heralded slot has a node to patch.
//   4. Hidden trace-out: when a coherent qubit jumps to a Leaked or Lost
//      level, insert an R on that qubit immediately after the operation.
//      The existing reset lowering turns it into a hidden measurement plus a
//      corrective Pauli; it adds no visible measurement and shifts no record
//      index. "Coherent" means the carrier state the base operation would
//      leave with no jump is ComputationalUnknown -- a qubit a gate has just
//      made coherent still needs trace-out even if it entered known.
//   5. Carrier materialization: when a jump lands on a computational level,
//      insert an R (plus an X for the |1> level) immediately
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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

// One measurement the rewrite replaced with a classifier record write.
struct ClassifiedMeasurement {
    // Visible measurement record index of the replaced measurement.
    uint32_t slot = 0;

    // The measured qubit's level id at the measurement.
    uint8_t level = 0;

    // Index into the rewritten circuit's nodes of the READOUT_NOISE node
    // drawing this slot's bit, or SIZE_MAX when the column is deterministic
    // and the bit is the MPAD literal itself.
    size_t noise_node = SIZE_MAX;
};

// Result of a rewrite: the circuit plus the classifier record writes it
// contains, in ascending slot order.
struct RewriteResult {
    Circuit circuit;
    std::vector<ClassifiedMeasurement> classified_measurements;
};

// Produce the rewritten circuit for `original` under `history` and `model`.
// Throws std::invalid_argument when the trajectory policy rejects an
// operation (naming the operation index, qubit, gate, and status), when a
// measurement on a leaked/lost qubit needs a classifier the model does not
// provide (or one that is not a two- or three-symbol stochastic column), or
// when `history` does not describe `original` (qubit count or transition
// count mismatch).
//
// Precondition: `original` is a parser-normalized circuit -- each
// single-qubit operation is a single-target node and each two-qubit
// operation is a single pair. The keep / drop / reject decision is made per
// node, so a hand-built node that packs several single-qubit operands would
// be dropped or rejected as a whole rather than per operand.
RewriteResult rewrite(const Circuit& original, const NonComputationalHistory& history,
                      const NonComputationalModel& model);

}  // namespace clifft
