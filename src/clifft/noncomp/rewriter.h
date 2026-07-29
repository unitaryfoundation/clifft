#pragma once

// Rewrites a hook-expanded circuit for one shot so it can use Clifft's
// existing compiler and VM. User-facing behavior is documented in
// docs/theory/noncomputational.md.
//
// The rewrite preserves the visible measurement record while applying the
// noncomputational events already resolved for the shot:
//
// - A measurement on a leaked or lost qubit becomes an MPAD in the same
//   record slot. READOUT_NOISE supplies a random classifier result when
//   needed. Kept computational M/MR measurements receive the classifier's
//   readout error when its computational columns are not the identity.
// - A recorded jump inserts a reset immediately after the transition
//   annotation. For a leaked or lost destination, the reset implements the
//   trace-out. For a return to g or e, it prepares the qubit at that level,
//   with an X after the reset for e.
// - A transition on a computational qubit remains a runtime instrument. A
//   transition on a leaked or lost qubit is removed after applying the
//   driver's pre-drawn outcome.
//
// The rewriter does not sample, compile, or run the VM.

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

// One measurement replaced with a classifier-generated record bit.
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

// One recorded jump, in circuit order. `target` identifies the transition
// annotation and `destination_level` is the level selected by the driver.
struct ResolvedJump {
    AnnotationTarget target;
    Level destination_level = Level::G;
};

// A pre-drawn outcome for a transition reached while its qubit is leaked
// or lost. No-jump outcomes are included so the rewriter can verify that
// every such transition has a matching outcome, in circuit order.
struct ClassicalOutcome {
    AnnotationTarget target;
    // The destination level, or nullopt when no jump occurred.
    std::optional<Level> destination;
    // The qubit's level when the outcome was drawn. The rewriter checks this
    // to prevent reusing an outcome for a different source level when a later
    // trap causes the circuit to be walked again.
    Level source_level = Level::G;
};

// Everything needed to rewrite one shot: its initial qubit statuses,
// recorded jumps, and pre-drawn outcomes for transitions reached on leaked or
// lost qubits.
struct TrajectoryEvents {
    std::vector<QubitStatus> initial_status;
    std::vector<ResolvedJump> jumps;
    std::vector<ClassicalOutcome> classical_outcomes;
};

// The rewritten circuit and the information the driver needs from it.
struct ContinuationRewrite {
    Circuit circuit;
    std::vector<ClassifiedMeasurement> classified_measurements;

    // Index in the rewritten circuit of the reset inserted for the last
    // recorded jump. It is absent unless the caller requests a forced
    // trace-out. The driver passes the index to trace(), which reports the
    // hidden measurement slot assigned during reset lowering.
    std::optional<size_t> forced_traceout_node;

    // Original annotation target for each transition kept as a runtime
    // instrument. Entries follow circuit emission order, which is also the
    // order in which trace() assigns site IDs. This lets the driver map a
    // trapped site back to its operation and qubit in `annotated`.
    std::vector<AnnotationTarget> site_targets;

    // Every qubit's status after the rewritten circuit. The driver reports
    // these as the shot's final statuses.
    std::vector<QubitStatus> final_status;
};

// Rewrite `annotated` (the hook-expanded circuit) for `events`. The portion
// through the last recorded jump keeps the same compiled prefix as the code
// that encountered that jump; only the later suffix is changed. Recorded
// jumps must match transition annotations in the circuit, and
// `classical_outcomes` must list exactly the transitions reached while their
// qubits are leaked or lost, in circuit order.
//
// When `force_last_traceout` is true, the reset inserted for the last jump is
// recorded in `forced_traceout_node` so the driver can force its hidden
// measurement to the source level reported by the trap. Throws
// std::invalid_argument when the policy rejects an operation or the events do
// not describe this circuit.
ContinuationRewrite rewrite_continuation(const Circuit& annotated, const TrajectoryEvents& events,
                                         bool force_last_traceout,
                                         const NonComputationalModel& model);

}  // namespace clifft
