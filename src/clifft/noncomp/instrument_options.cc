#include "clifft/noncomp/instrument_options.h"

namespace clifft {

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model) {
    constexpr Level computational[2] = {Level::G, Level::E};

    InstrumentTraceOptions options;
    options.neglect_damping = model.policy().damping == DampingPolicy::Neglect;

    for (const auto& [name, instrument] : model.transitions()) {
        InstrumentSpec spec;
        for (int s = 0; s < 2; ++s) {
            spec.p_total[s] = instrument.column_sum(computational[s]);
            for (int d = 0; d < 2; ++d) {
                spec.p_dest[s][d] = instrument.prob(computational[d], computational[s]);
            }
        }
        options.transitions.emplace(name, spec);
    }
    return options;
}

}  // namespace clifft
