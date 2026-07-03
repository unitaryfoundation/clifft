#pragma once

// Exact-mode driver: the runtime half of exact state-dependent jumps.
//
// Under UnknownSourcePolicy::Exact, transition firing moves into the VM:
// the annotated circuit compiles once as a shared main line with every
// annotation materialized as an instrument site, each shot preloads its
// sampled initial levels into the Pauli frame and executes, and a fire
// that cannot resolve in-line traps back here. The driver resolves the
// destination, extends the shot's event record, fetches or compiles the
// continuation (cached by the status-outcome delta plus herald flags),
// and resumes past the site. sample_noncomputational() routes here; this
// header is internal.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/orchestrator.h"

#include <cstdint>
#include <optional>

namespace clifft {

NonComputationalSample sample_noncomputational_exact(const Circuit& circuit,
                                                     const NonComputationalModel& model,
                                                     uint32_t shots, uint64_t global_seed,
                                                     std::optional<uint32_t> max_rank);

}  // namespace clifft
