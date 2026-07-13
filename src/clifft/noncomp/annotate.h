#pragma once

// Annotation expansion for the noncomputational trajectory layer.
//
// annotate(circuit, model) expands the model's gate hooks -- transition
// keys that name a gate -- into explicit LEVEL_TRANSITION[key]
// annotations: one single-target annotation per physical qubit operand,
// inserted immediately after each hooked operation in operand order. The
// sampler and rewriter therefore consume only explicit annotations. Circuits
// may also carry hand-written LEVEL_TRANSITION and LOSS annotations;
// expansion leaves them in place.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

namespace clifft {

Circuit annotate(const Circuit& circuit, const NonComputationalModel& model);

}  // namespace clifft
