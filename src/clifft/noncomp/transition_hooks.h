#pragma once

// Expands gate hooks into explicit circuit annotations.
//
// Each physical gate is followed by its configured partner interactions and
// then its per-operand LEVEL_TRANSITION hooks, before the next user node.
// Explicit annotations are copied unchanged and compose sequentially with
// generated effects. Record-controlled feedback receives neither kind of
// hook. Later stages therefore only need to handle explicit annotations.
// Direct AST inputs need one physical pair per node only when that gate has
// an enabled partner effect. Otherwise existing node grouping and post-node
// transition placement are preserved.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

namespace clifft {

Circuit expand_transition_hooks(const Circuit& circuit, const NonComputationalModel& model);

}  // namespace clifft
