#pragma once

// Exact-mode driver: resolves level transitions (leak, loss, decay) at
// sample time, with no sampling approximation.
//
// A transition whose outcome depends on a qubit's quantum state -- a
// jump out of superposition, where which level the qubit leaves from is
// not yet decided -- cannot be drawn before that state exists. Under
// UnknownSourcePolicy::Exact the annotated circuit therefore compiles
// once with every annotation kept as a runtime instrument site; each
// shot preloads its sampled initial levels and executes; a fire the VM
// cannot resolve in-line traps back to this driver, which draws the
// destination, extends the shot's event record, fetches or compiles the
// matching continuation (cached by the status-outcome delta plus herald
// flags), and resumes past the site. DampingPolicy::Neglect also runs
// through this driver: it changes how a trapped fire resolves (the
// carrier hands over uncollapsed and the continuation's trace-out is
// forced to the reported source), not whether the mode applies.
// sample_noncomputational() routes here; this header is internal. The
// driver's working vocabulary is defined at the top of exact_driver.cc.

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
