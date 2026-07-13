#pragma once

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <cstdint>
#include <string_view>

namespace clifft {
namespace test {

// Source-dependent fixture used to follow one instrument through tracing and
// lowering. The unassigned fire mass is the noncomputational trap remainder.
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

inline CompiledModule compile_instruments_raw(std::string_view text,
                                              const InstrumentTraceOptions& options) {
    return lower(trace(parse(text), &options));
}

inline CompiledModule compile_instruments_full(std::string_view text,
                                               const InstrumentTraceOptions& options) {
    auto hir = trace(parse(text), &options);
    auto hir_passes = default_hir_pass_manager();
    hir_passes.run(hir);
    auto module = lower(hir);
    auto bytecode_passes = default_bytecode_pass_manager();
    bytecode_passes.run(module);
    return module;
}

inline SchrodingerState make_shot_state(const CompiledModule& module, uint64_t seed) {
    return SchrodingerState(StateConfig{.peak_rank = module.peak_rank,
                                        .num_measurements = module.total_meas_slots,
                                        .num_qubits = module.num_qubits,
                                        .num_detectors = module.num_detectors,
                                        .num_observables = module.num_observables,
                                        .num_exp_vals = module.num_exp_vals,
                                        .seed = seed});
}

}  // namespace test
}  // namespace clifft
