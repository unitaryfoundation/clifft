#pragma once

// Circuit rewriter for the noncomputational layer: turns an annotated
// circuit plus a shot's resolved events into an ordinary clifft::Circuit
// for the existing compile pipeline -- it introduces no new instruction
// kinds. The per-node semantics:
//
//   1. Per-op policy: replay each operation's per-qubit status through the
//      shared status stepper and keep, drop, or reject the operation. An
//      operation with no representable effect on a leaked or lost operand is
//      excised whole (identity on the surviving operands), whose statuses
//      then keep their entry values. A non-reset X/Y-basis measurement
//      (MX/MY) or a multi-qubit-parity measurement (MPP) of such an operand
//      has no faithful single-bit form and is rejected -- a representability
//      limit, not a policy choice. A measure-and-reset (MR/MRX/MRY) is kept.
//   2. Classifier record write: a measurement on a leaked or lost qubit is
//      not a physical Born measurement -- the model's classifier defines its
//      record bit. The measurement node is replaced by an MPAD writing the
//      same visible record slot, so the record layout and every rec[-k]
//      reference are preserved and the bit reaches the SVM's
//      detector/observable evaluation. A stochastic classifier column adds a
//      READOUT_NOISE on that slot so the bit is drawn at sample time inside
//      the VM; a deterministic column pads the literal bit with no draw. A
//      measure-and-reset additionally keeps its reset as a separate node.
//      Each such replacement is reported in classified_measurements (in
//      slot order) so callers can post-process per-slot classifier behavior
//      -- e.g. the driver's herald patching for three-symbol classifiers,
//      which re-points a heralded slot's flip probability at one half. For
//      a three-symbol column the emitted probability is the bit's
//      not-heralded conditional, and a READOUT_NOISE node is always emitted
//      so a heralded slot has a node to patch. A kept computational Z-basis
//      measurement (M, MR) additionally receives the classifier's
//      computational readout confusion as an asymmetric READOUT_NOISE on
//      its slot -- the misreport probabilities P(1 | zero level) and
//      P(0 | one level) -- when those columns are not the identity.
//   3. Carrier trace-out / re-prep: every recorded jump inserts an R on
//      its qubit at the annotation's position, plus an X when the
//      destination is the |1> level. For a Leaked or Lost destination the
//      reset is the trace-out unraveling: the existing reset lowering
//      turns it into a hidden measurement plus a corrective Pauli, adding
//      no visible measurement and shifting no record index, and the
//      site's collapse-before-trap makes that hidden measurement
//      deterministic. For a computational destination the same reset
//      prepares the SVM carrier at the definite destination level -- the
//      collapse unraveling for a coherent carrier, and a rezero of stale
//      residual when a leaked or lost qubit is recaptured.
//
// Transition consults happen only at LEVEL_TRANSITION and LOSS annotations
// (gate hooks are expanded by annotate() before rewriting). A consult on a
// leaked or lost qubit is consumed against its pre-drawn outcome; one on a
// computational qubit stays a runtime instrument site. The rewriter does
// not sample, compile, or run the SVM.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace clifft {

// One measurement the rewrite replaced with a classifier record write.
struct ClassifiedMeasurement {
    // Visible measurement record index of the replaced measurement.
    uint32_t slot = 0;

    // The measured qubit's level at the measurement.
    Level level = Level::G;

    // Index into the rewritten circuit's nodes of the READOUT_NOISE node
    // drawing this slot's bit, or SIZE_MAX when the column is deterministic
    // and the bit is the MPAD literal itself.
    size_t noise_node = SIZE_MAX;
};

// =============================================================================
// Exact-mode continuation rewrite
// =============================================================================
//
// Transition firing happens at runtime, and a fire that cannot resolve
// in-line halts execution at its instrument site. The continuation is
// the full circuit recompiled under the now-known status outcomes: its
// prefix (everything up to and including the trapped annotation) is
// emitted verbatim so it compiles bit-identically to the code that
// already ran, and only the suffix is rewritten. Annotation nodes are
// kept wherever their qubit is still computational -- they stay runtime
// instruments, including on a recaptured qubit -- and are consumed only
// where the source is a classical (leaked/lost) status, using outcomes
// the driver pre-drew.

// One resolved jump in a shot's trap chain, in circuit order. `op_index`
// and `qubit` name the annotation target that trapped; the destination
// level is the host's draw from the transition column.
struct ResolvedJump {
    uint32_t op_index = 0;
    uint32_t qubit = 0;
    Level destination_level = Level::G;
};

// One pre-drawn outcome for an annotation target whose source status is
// classical at that point (seepage, restore, lost-stays-lost). Records
// no-jump outcomes too: the stream must cover every classical-source
// consult after the last trap, in circuit order, so the rewrite can
// validate it describes this circuit.
struct ClassicalOutcome {
    uint32_t op_index = 0;
    uint32_t qubit = 0;
    // The jump destination; nullopt records a no-jump outcome.
    std::optional<Level> destination;
    // The level the qubit held when the outcome was drawn. The emitted
    // nodes do not depend on it; it exists so every reuse and
    // consumption can check the outcome is not being replayed against a
    // different source. Today a qubit's noncomputational level moves
    // only through its own consults, so a mismatch is unreachable --
    // this check is what keeps that invariant explicit, and loud if a
    // future cross-qubit transition breaks it.
    Level source_level = Level::G;
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
// a coherent carrier) -- the node index (in the rewritten stream) of the
// last recorded jump's trace-out reset, which the driver hands to trace()
// to obtain the hidden slot. Every jump emits a reset, so the forced form
// always has one to point at.
struct ContinuationRewrite {
    Circuit circuit;
    std::vector<ClassifiedMeasurement> classified_measurements;

    // Node index (in the rewritten stream) of the last recorded jump's
    // trace-out reset, or nullopt when the caller did not request forcing.
    // The driver hands this to trace() via
    // InstrumentTraceOptions::forced_traceout_node; trace() reports the
    // hidden slot it assigns through HirModule::forced_traceout_slot.
    std::optional<size_t> forced_traceout_node;

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
