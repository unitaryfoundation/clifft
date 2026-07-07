#pragma once

// Static pre-sampling validation for circuit/model pairs.
//
// validate_static rejects a circuit/model combination before any shot is
// drawn when sampling could ever encounter an unrepresentable operation: an
// X/Y-basis or parity measurement that a reachable leaked or lost qubit
// could reach, or a measurement that requires a classifier the model lacks.
//
// The check runs an abstract interpretation over per-qubit SETS of reachable
// QubitStatus values (a uint8_t bitmask), walking the annotated circuit with
// the same policy primitives the rewriter applies to concrete statuses.
// Because the two share primitives, they can never disagree: any path this
// scan allows, the rewriter can handle, and any path this scan forbids,
// the rewriter would reject at runtime.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

namespace clifft {

// Reject circuit/model pairs whose sampling could ever hit an
// unrepresentable operation, before any shot is drawn. Walks the
// annotated circuit with a per-qubit set of reachable statuses and
// applies the same per-status policy primitives the rewrite applies to
// concrete statuses, so the two can never disagree.
//
// Throws std::invalid_argument when a rejectable path is reachable.
// The annotated circuit must be the result of annotate(original, model).
void validate_static(const Circuit& annotated, const NonComputationalModel& model);

}  // namespace clifft
