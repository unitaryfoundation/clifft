#include "clifft/api/basis_amplitudes.h"

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/phase_aware_planner.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/util/mask_view.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace clifft::sampling {
namespace {

void validate_output_basis(uint32_t num_qubits, std::span<const uint64_t> basis) {
    if (basis.size() != mask_word_count(num_qubits)) {
        throw std::invalid_argument("amplitude output basis must have ceil(num_qubits / 64) words");
    }
    if (!mask_has_only_bits(MaskView{basis}, num_qubits)) {
        throw std::invalid_argument("amplitude output basis sets unused high bits");
    }
}

double exact_half_amplitude_scale(uint32_t count) {
    const int exponent = -static_cast<int>(count / 2U);
    const double odd_factor = count % 2U == 0 ? 1.0 : 1.0 / std::numbers::sqrt2;
    return std::ldexp(odd_factor, exponent);
}

uint32_t final_active_width(const SamplingPlan& plan) {
    return plan.actions.empty() ? plan.initial_active_width : plan.actions.back().active_after;
}

Circuit append_terminal_measurements(const Circuit& source) {
    Circuit result(source);
    if (source.num_qubits == 0) {
        return result;
    }
    std::vector<Target> targets;
    targets.reserve(source.num_qubits);
    for (uint32_t q = 0; q < source.num_qubits; ++q) {
        targets.push_back(Target::qubit(q));
    }
    result.nodes.push_back(AstNode{.gate = GateType::M,
                                   .targets = std::move(targets),
                                   .args = {},
                                   .source_line = 0,
                                   .tag = {}});
    result.num_measurements = source.num_qubits;
    return result;
}

}  // namespace

BasisAmplitudeQuery::Prepared BasisAmplitudeQuery::prepare(const Circuit& circuit,
                                                           std::span<const uint64_t> output_basis,
                                                           std::complex<double> input_phase) {
    validate_output_basis(circuit.num_qubits, output_basis);
    std::vector<uint8_t> output_records(circuit.num_qubits);
    for (uint32_t q = 0; q < circuit.num_qubits; ++q) {
        output_records[q] = static_cast<uint8_t>((output_basis[q / 64U] >> (q % 64U)) & 1U);
    }
    PhaseAwareHir traced =
        trace_phase_aware_terminal_measurements(append_terminal_measurements(circuit));
    // Output effects move left while non-Clifford expansions move right. The
    // target-aware planner retains forced-branch phases across the resulting
    // coordinate changes and scalar rotations.
    StatevectorSqueezePass{}.run(traced.hir);
    PhaseAwareSamplingPlan planned = plan_sampling_phase_aware(
        traced.hir, std::move(traced.final_clifford_frame), output_records);
    if (final_active_width(planned.plan) != 0) {
        throw std::logic_error("terminal effects did not eliminate every active coordinate");
    }
    std::complex<double> phase = input_phase * traced.source_scalar * planned.scalar;
    phase *= internal::clifford_row_phase(planned.final_tableau, planned.final_clifford_frame,
                                          output_basis);
    return Prepared{.plan = std::move(planned.plan),
                    .output_records = std::move(output_records),
                    .phase = phase};
}

BasisAmplitudeQuery::BasisAmplitudeQuery(const Circuit& circuit,
                                         std::span<const uint64_t> output_basis,
                                         std::complex<double> input_phase)
    : BasisAmplitudeQuery(prepare(circuit, output_basis, input_phase)) {}

BasisAmplitudeQuery::BasisAmplitudeQuery(Prepared prepared)
    : plan_(std::move(prepared.plan)),
      output_records_(std::move(prepared.output_records)),
      phase_(prepared.phase) {}

std::complex<double> BasisAmplitudeQuery::evaluate() const {
    Executor executor(plan_);
    const ReplayResult replay = executor.replay_effect(output_records_);
    if (!replay.reachable) {
        return {0.0, 0.0};
    }
    const State& state = executor.state();
    assert(state.active_width() == 0 && "terminal effects must eliminate every active coordinate");

    const std::complex<double> normalized{state.real_data()[0], state.imag_data()[0]};
    return phase_ * std::exp(0.5 * replay.log_probability) *
           exact_half_amplitude_scale(replay.exact_half_probability_factors) * normalized;
}

}  // namespace clifft::sampling
