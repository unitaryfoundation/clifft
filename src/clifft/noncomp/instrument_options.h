#pragma once

// Bridge from a NonComputationalModel to the front-end's instrument
// materialization: resolve every named transition into the plain
// per-computational-source probabilities trace() consumes, and bake in
// the model's damping policy. The front-end stays model-free -- level
// tables, transition matrices, and policy enums never cross this
// boundary, only InstrumentSpec numbers.

#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/model.h"

namespace clifft {

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model);

}  // namespace clifft
