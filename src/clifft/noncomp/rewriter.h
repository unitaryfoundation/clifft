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
//      emitted so a heralded slot has a node to patch. A kept computational
//      Z-basis measurement (M, MR) additionally receives the classifier's
//      computational readout confusion as an asymmetric READOUT_NOISE on
//      its slot -- the misreport probabilities P(1 | zero level) and
//      P(0 | one level) -- when those columns are not the identity.
//   4. Hidden trace-out: when a coherent qubit jumps to a Leaked or Lost
//      level, insert an R on that qubit at the annotation's position.
//      The existing reset lowering turns it into a hidden measurement plus a
//      corrective Pauli; it adds no visible measurement and shifts no record
//      index. "Coherent" means the qubit's status at the annotation's
//      position is ComputationalUnknown; a definite atom needs no
//      unraveling.
//   5. Carrier materialization: when a jump lands on a computational level,
//      insert an R (plus an X for the |1> level) at the annotation's
//      position, so the SVM carrier is prepared at the definite
//      destination level. This is done for every carrier state: it is the
//      collapse unraveling for a coherent carrier, a deterministic re-prep
//      for a known one, and it rezeros a stale residual when a leaked or
//      lost qubit is recaptured. Like the trace-out, it shifts no record
//      index.
//
// Transition consults happen only at LEVEL_TRANSITION and LOSS annotations
// (gate hooks are expanded by annotate() before sampling and rewriting).
// The rewriter consumes those annotations -- replaying outcomes from
// history.transitions in sampler order, one record per annotation target --
// and emits only their carrier edits; annotation nodes never reach the
// rewritten circuit. It does not sample, compile, or run the SVM.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/history.h"
#include "clifft/noncomp/model.h"

#include <cstddef>
#include <cstdint>
#include <utility>
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

// =============================================================================
// Exact-mode continuation rewrite
// =============================================================================
//
// In exact mode, transition firing happens at runtime and a fire that
// cannot resolve in-line halts execution at its instrument site. The
// continuation is the full circuit recompiled under the now-known status
// outcomes: its prefix (everything up to and including the trapped
// annotation) is emitted verbatim so it compiles bit-identically to the
// code that already ran, and only the suffix is rewritten. Unlike the AOT
// rewrite above, annotation nodes are kept wherever their qubit is still
// computational -- they stay runtime instruments, including on a
// recaptured qubit -- and are consumed only where the source is a
// classical (leaked/lost) status, using outcomes the host pre-drew.

// One resolved jump in a shot's trap chain, in circuit order. `op_index`
// and `qubit` name the annotation target that trapped; the destination
// level is the host's draw from the transition column.
struct ResolvedJump {
    uint32_t op_index = 0;
    uint32_t qubit = 0;
    uint8_t destination_level = 0;
};

// One pre-drawn outcome for an annotation target whose source status is
// classical at that point (seepage, restore, lost-stays-lost). Records
// no-jump outcomes too: the stream must cover every classical-source
// consult after the last trap, in circuit order, so the rewrite can
// validate it describes this circuit.
struct ClassicalOutcome {
    uint32_t op_index = 0;
    uint32_t qubit = 0;
    bool jumped = false;
    uint8_t destination_level = 0;
    // The level the qubit held when the outcome was drawn. The emitted
    // nodes do not depend on it; it exists so every reuse and
    // consumption can check the outcome is not being replayed against a
    // different source. Today a qubit's noncomputational level moves
    // only through its own consults, so a mismatch is unreachable --
    // this check is what keeps that invariant explicit, and loud if a
    // future cross-qubit transition breaks it.
    uint8_t source_level = 0;
};

// The status-outcome delta a continuation is compiled under: the shot's
// initial statuses, the trap chain so far, and the pre-drawn
// classical-source outcomes for the remaining annotations. This struct is
// also the continuation cache key's content: two shots with equal events
// share one compiled module. The one field the key omits is
// ClassicalOutcome::source_level -- it is derived from everything else
// in the events and exists only to validate replays, so it cannot vary
// within a key.
struct ExactShotEvents {
    std::vector<QubitStatus> initial_status;
    std::vector<ResolvedJump> jumps;
    std::vector<ClassicalOutcome> classical_outcomes;
};

// A compiled-continuation rewrite: the circuit, the classifier record
// writes in its suffix, and -- when the *last* jump in the chain traps at
// a site whose collapse could not happen in-line (a neglect-form site on
// a coherent carrier) -- the hidden record slot of that jump's carrier
// reset, which the driver forces to the trap's reported source.
struct ContinuationRewrite {
    Circuit circuit;
    std::vector<ClassifiedMeasurement> classified_measurements;

    // Hidden record slot of the reset that collapses the last jump's
    // carrier -- the trace-out R of a noncomputational destination, or
    // the materializing R of a computational one -- or SIZE_MAX when the
    // caller did not request forcing. Requesting forcing for a jump that
    // emits no reset (the carrier's level was already definite) throws.
    size_t forced_traceout_slot = SIZE_MAX;

    // Annotation target of each kept (runtime-instrument) site, in
    // emission order -- which is trace()'s materialization order, so the
    // vector maps a trap's site_id to its (op_index, qubit) in the
    // annotated circuit's coordinates.
    std::vector<std::pair<uint32_t, uint32_t>> site_targets;

    // Every qubit's status at the end of the walk: the shot's final
    // statuses once execution reaches the end of this continuation.
    std::vector<QubitStatus> final_status;
};

// Rewrite `annotated` (the hook-expanded circuit the main line was
// compiled from) into the continuation for `events`. Every annotation
// target before or at the last jump must be covered by the walk (no-fire
// for targets not in `jumps`); classical_outcomes must list, in circuit
// order, exactly the classical-source consults after the last jump.
// `force_last_traceout` marks the last jump as a neglect-form trap whose
// carrier arrives uncollapsed. Throws std::invalid_argument on policy
// rejects and on events that do not describe this circuit.
ContinuationRewrite rewrite_continuation(const Circuit& annotated, const ExactShotEvents& events,
                                         bool force_last_traceout,
                                         const NonComputationalModel& model);

}  // namespace clifft
