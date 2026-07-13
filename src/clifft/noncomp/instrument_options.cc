#include "clifft/noncomp/instrument_options.h"

namespace clifft {

InstrumentTraceOptions instrument_trace_options(const NonComputationalModel& model) {
    constexpr Level computational[2] = {Level::G, Level::E};

    InstrumentTraceOptions options;
    options.neglect_instrument_damping = model.policy().damping == DampingPolicy::Neglect;

    for (const auto& [name, instrument] : model.transitions()) {
        InstrumentProbabilities probabilities;
        // Compress only the columns whose source remains quantum. p_fire
        // retains each full five-level column sum, while
        // p_computational_dest retains its G/E destination entries. Their
        // difference is the aggregate LeakG/LeakE/Lost trap probability.
        // Columns sourced at a noncomputational level stay in the model and
        // are consulted classically by the exact-mode driver.
        for (int s = 0; s < 2; ++s) {
            probabilities.p_fire[s] = instrument.column_sum(computational[s]);
            for (int d = 0; d < 2; ++d) {
                probabilities.p_computational_dest[s][d] =
                    instrument.prob(computational[d], computational[s]);
            }
        }
        options.transitions.emplace(name, probabilities);
    }
    return options;
}

}  // namespace clifft
