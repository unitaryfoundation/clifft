#pragma once

// Annotation expansion for the noncomputational trajectory layer.
//
// annotate(circuit, model) expands the model's gate hooks -- transition
// keys that name a gate -- into explicit LEVEL_TRANSITION[key] annotations in the
// circuit: one single-target annotation per Physical qubit operand,
// inserted immediately after each hooked operation, in operand order. The
// result carries every transition consult point as a first-class
// instruction, positioned where it fires, so the sampler and rewriter
// consume only annotations and the expanded circuit is a complete audit of
// the applied noise model. Circuits may also carry hand-written LEVEL_TRANSITION
// and LOSS annotations; expansion leaves them where they are.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

namespace clifft {

Circuit annotate(const Circuit& circuit, const NonComputationalModel& model);

}  // namespace clifft
