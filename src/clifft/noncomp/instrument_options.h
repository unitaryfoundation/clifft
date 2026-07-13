#pragma once

// Bridge from a NonComputationalModel to the front-end's instrument
// materialization: compress each named five-level transition matrix into
// InstrumentSpec's two quantum-source columns, and carry the model-wide
// damping setting. The front-end stays model-free -- level tables,
// transition matrices, and policy enums never cross this boundary, only
// InstrumentSpec numbers.

#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/model.h"

namespace clifft {

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model);

}  // namespace clifft
