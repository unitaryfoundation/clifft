#pragma once

// Trajectory driver for level transitions such as leak, loss, and decay.
//
// Each shot samples its initial levels, then runs transitions on computational
// qubits as executor instruments. If the executor cannot finish a transition,
// it stops and reports a trap. The driver records the outcome, compiles a
// circuit rewritten for those events, and resumes after the trapped transition.
// Transitions reached while a qubit is leaked or lost are drawn by the driver
// before compilation. DampingPolicy::Neglect uses the same process but omits
// source-dependent no-jump back-action.
//
// PreparedNonComputationalRequest calls this internal entry point.

#include "clifft/noncomp/preparation.h"
#include "clifft/noncomp/sample.h"
#include "clifft/util/shot_seed.h"

#include <cstdint>

namespace clifft {

NonComputationalSample run_trajectory_driver(const PreparedNonComputationalRequest& request,
                                             uint32_t shots, const SeedRoot& root,
                                             uint32_t threads);

}  // namespace clifft
