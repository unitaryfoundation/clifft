#pragma once

#include "clifft/noncomp/instrument_options.h"

namespace clifft::test {

// Source-dependent fixture shared by front-end, planner, and executor tests.
// The unassigned fire mass is the noncomputational trap remainder.
inline InstrumentTraceOptions source_dependent_jump_options(bool neglect_damping = false) {
    InstrumentTraceOptions options;
    InstrumentProbabilities probabilities;
    probabilities.p_fire[0] = 0.1;
    probabilities.p_computational_dest[0][0] = 0.02;
    probabilities.p_computational_dest[0][1] = 0.03;
    probabilities.p_fire[1] = 0.4;
    options.transitions.emplace("jump", probabilities);
    options.neglect_instrument_damping = neglect_damping;
    return options;
}

}  // namespace clifft::test
