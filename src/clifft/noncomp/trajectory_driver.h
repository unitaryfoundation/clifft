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
// sample_noncomputational() calls this internal entry point.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/sample.h"
#include "clifft/util/shot_seed.h"

#include <cstdint>
#include <optional>

namespace clifft {

NonComputationalSample run_trajectory_driver(const Circuit& circuit,
                                             const NonComputationalModel& model, uint32_t shots,
                                             const SeedRoot& root,
                                             std::optional<uint32_t> max_active_width_cap,
                                             uint32_t threads);

}  // namespace clifft
