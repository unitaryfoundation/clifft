#pragma once

// Rewrites an annotated circuit plus one shot's resolved events into an
// ordinary clifft::Circuit for the existing compile pipeline; no new
// instruction kinds. The user-level semantics (what drops, what
// classifies, what restores) are documented in
// docs/theory/noncomputational.md. The mechanisms owned here:
//
// - A classified measurement becomes an MPAD writing its original record
//   slot, so the record layout and every rec[-k] reference are preserved.
//   A stochastic classifier column adds a READOUT_NOISE drawing the bit at
//   sample time; a deterministic column pads the literal bit. With a
//   three-symbol classifier the emitted flip probability is the bit's
//   not-heralded conditional, and a READOUT_NOISE is always emitted so a
//   heralded slot has a node to patch. A kept computational M/MR receives
//   the classifier's readout confusion as an asymmetric READOUT_NOISE when
//   those columns are not the identity. A measure-and-reset keeps its
//   reset as a separate node only when the stepper restores the site.
// - Every recorded jump inserts an R at the annotation's position (plus an
//   X for an |1> destination): the trace-out unraveling for a
//   noncomputational destination, the carrier re-preparation for a
//   computational one. Reset lowering adds no visible measurement and
//   shifts no record index.
// - Consults happen only at LEVEL_TRANSITION and LOSS nodes (annotate()
//   expands gate hooks first). A computational-source consult stays a
//   runtime instrument site; a classical-source (leaked/lost) consult is
//   consumed against its pre-drawn outcome. The rewriter does not sample,
//   compile, or run the SVM.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace clifft {

// One qubit target of a transition annotation in the annotated circuit.
struct AnnotationTarget {
    uint32_t op_index = 0;
    uint32_t qubit = 0;

    constexpr auto operator<=>(const AnnotationTarget&) const = default;
};

// One measurement the rewrite replaced with a classifier record write.
struct ClassifiedMeasurement {
    // Visible measurement record index of the replaced measurement.
    uint32_t slot = 0;

    // The measured qubit's level at the measurement.
    Level level = Level::G;

    // Index into the rewritten circuit's nodes of the READOUT_NOISE node
    // drawing this slot's bit, or absent when the column is deterministic
    // and the bit is the MPAD literal itself.
    std::optional<size_t> noise_node;
};

// One resolved jump in a shot's trap chain, in circuit order. `op_index`
// `target` names the annotation target that trapped; the destination
// level is the driver's draw from the transition column.
struct ResolvedJump {
    AnnotationTarget target;
    Level destination_level = Level::G;
};

// One pre-drawn outcome for an annotation target whose source status is
// classical at that point. Records no-jump outcomes too: the stream must
// cover every classical-source consult after the last trap, in circuit
// order, so the rewrite can validate it describes this circuit.
struct ClassicalOutcome {
    AnnotationTarget target;
    // The jump destination; nullopt records a no-jump outcome.
    std::optional<Level> destination;
    // The level the qubit held when the outcome was drawn. Validation
    // only: consumption checks the outcome is not replayed against a
    // different source. Derived from the other event fields, so the
    // continuation cache key omits it.
    Level source_level = Level::G;
};

// The status-outcome delta a continuation is compiled under: the shot's
// initial statuses, the trap chain so far, and the pre-drawn
// classical-source outcomes for the remaining annotations. Also the
// continuation cache key's content (source_level excepted): two shots
// with equal events share one compiled module.
struct ExactShotEvents {
    std::vector<QubitStatus> initial_status;
    std::vector<ResolvedJump> jumps;
    std::vector<ClassicalOutcome> classical_outcomes;
};

// A continuation rewrite and its bookkeeping for the driver.
struct ContinuationRewrite {
    Circuit circuit;
    std::vector<ClassifiedMeasurement> classified_measurements;

    // Node index (in the rewritten stream) of the last recorded jump's
    // trace-out reset, or nullopt when the caller did not request forcing.
    // The driver hands this to trace() via
    // InstrumentTraceOptions::forced_traceout_node; trace() reports the
    // hidden slot it assigns through HirModule::forced_traceout_slot.
    // Every jump emits a reset, so the forced form always has one.
    std::optional<size_t> forced_traceout_node;

    // Annotation target of each kept (runtime-instrument) site, in
    // emission order -- which is trace()'s materialization order, so the
    // vector maps a trap's site_id to its (op_index, qubit) in the
    // annotated circuit's coordinates.
    std::vector<AnnotationTarget> site_targets;

    // Every qubit's status at the end of this continuation's walk: the
    // final statuses the driver reports for the shot.
    std::vector<QubitStatus> final_status;
};

// Rewrite `annotated` (the hook-expanded circuit) into the continuation
// for `events`. The prefix up to and including the last jump's annotation
// is emitted verbatim, so it compiles bit-identically to the code the
// shot already ran; only the suffix is rewritten. Every annotation target
// before or at the last jump must be covered by the walk (no-fire for
// targets not in `jumps`); classical_outcomes must list, in circuit
// order, exactly the classical-source consults after the last jump.
// `force_last_traceout` marks the last jump as a trap-form fire whose
// carrier arrives uncollapsed. Throws std::invalid_argument on policy
// rejects and on events that do not describe this circuit.
ContinuationRewrite rewrite_continuation(const Circuit& annotated, const ExactShotEvents& events,
                                         bool force_last_traceout,
                                         const NonComputationalModel& model);

}  // namespace clifft
