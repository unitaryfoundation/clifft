#include "clifft/noncomp/instrument_options.h"

namespace clifft {

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model) {
    const LevelSet& levels = model.levels();
    const uint8_t level_id[2] = {levels.computational_zero_id(), levels.computational_one_id()};

    InstrumentTraceOptions options;
    options.neglect_damping = model.policy().damping == DampingPolicy::Neglect;

    for (const auto& [name, instrument] : model.transitions()) {
        InstrumentSpec spec;
        for (int s = 0; s < 2; ++s) {
            spec.p_total[s] = instrument.column_sum(level_id[s]);
            for (int d = 0; d < 2; ++d) {
                spec.p_dest[s][d] = instrument.prob(level_id[d], level_id[s]);
            }
        }
        options.transitions.emplace(name, spec);
    }
    return options;
}

}  // namespace clifft
