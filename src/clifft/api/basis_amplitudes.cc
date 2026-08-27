#include "clifft/api/basis_amplitudes.h"

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/phase_aware_planner.h"
#include "clifft/sampling/state_queries.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace clifft::sampling {
namespace {

[[nodiscard]] size_t basis_word_count(uint32_t num_qubits) {
    return (static_cast<size_t>(num_qubits) + 63U) / 64U;
}

void validate_output_basis(uint32_t num_qubits, std::span<const uint64_t> basis) {
    const size_t expected_words = basis_word_count(num_qubits);
    if (basis.size() != expected_words) {
        throw std::invalid_argument("amplitude output basis must have ceil(num_qubits / 64) words");
    }
    const uint32_t used_bits = num_qubits % 64U;
    if (!basis.empty() && used_bits != 0) {
        const uint64_t valid = (uint64_t{1} << used_bits) - 1U;
        if ((basis.back() & ~valid) != 0) {
            throw std::invalid_argument("amplitude output basis sets unused high bits");
        }
    }
}

Circuit append_terminal_measurements(const Circuit& source) {
    for (const AstNode& node : source.nodes) {
        if (!is_unitary(node.gate) && node.gate != GateType::TICK) {
            throw std::invalid_argument("basis amplitude query requires a pure-unitary circuit");
        }
        const size_t arity = node.targets.size();
        if ((gate_arity(node.gate) == GateArity::PAIR && arity % 2U != 0) ||
            (gate_arity(node.gate) == GateArity::TRIPLE && arity % 3U != 0)) {
            throw std::invalid_argument("basis amplitude query received malformed gate targets");
        }
    }

    Circuit result(source);
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

struct PlannedAmplitude {
    SamplingPlan plan;
    std::complex<double> phase;
};

PlannedAmplitude plan_amplitude(HirModule hir, PhaseAwareCliffordFrame final_clifford_frame,
                                std::span<const uint8_t> output_records,
                                std::span<const uint64_t> output_basis,
                                std::complex<double> source_phase,
                                StatevectorSqueezePass squeeze_pass) {
    squeeze_pass.run(hir);
    PhaseAwareSamplingPlan planned =
        plan_sampling_phase_aware(hir, std::move(final_clifford_frame), output_records);
    std::complex<double> phase = source_phase * planned.scalar;
    phase *= internal::clifford_row_phase(planned.final_tableau, planned.final_clifford_frame,
                                          output_basis);
    return PlannedAmplitude{.plan = std::move(planned.plan), .phase = phase};
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
    HirModule alternate_hir = traced.hir;
    PhaseAwareCliffordFrame alternate_frame = traced.final_clifford_frame;
    const std::complex<double> source_phase = input_phase * traced.source_scalar;

    // Commuting expansions admit multiple legal schedules around the fixed
    // output effects. Plan both stable extremes and retain the cheaper dense
    // state without changing the ordinary sampling pipeline.
    PlannedAmplitude canonical =
        plan_amplitude(std::move(traced.hir), std::move(traced.final_clifford_frame),
                       output_records, output_basis, source_phase, StatevectorSqueezePass{});
    PlannedAmplitude alternate = plan_amplitude(
        std::move(alternate_hir), std::move(alternate_frame), output_records, output_basis,
        source_phase, StatevectorSqueezePass::with_reversed_commuting_expansions());
    PlannedAmplitude& selected =
        alternate.plan.peak_active_width < canonical.plan.peak_active_width ? alternate : canonical;
    return Prepared{.plan = std::move(selected.plan),
                    .output_records = std::move(output_records),
                    .phase = selected.phase};
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
    return phase_ * std::exp(0.5 * replay.log_probability) * normalized;
}

}  // namespace clifft::sampling
