#pragma once

// Expands gate hooks into explicit circuit annotations.
//
// A gate hook is a model transition named after a gate, such as "CZ". For
// every occurrence of that gate, expand_transition_hooks() inserts a
// LEVEL_TRANSITION immediately afterward for each qubit the gate acts on
// directly. Existing LEVEL_TRANSITION, LEAKAGE, and LOSS annotations are
// copied unchanged. Record-controlled feedback receives no transition because
// it does not physically execute the gate. Later stages therefore only need
// to handle explicit annotations.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

namespace clifft {

Circuit expand_transition_hooks(const Circuit& circuit, const NonComputationalModel& model);

}  // namespace clifft
