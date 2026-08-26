#include "clifft/api/basis_amplitudes.h"

#include "clifft/frontend/phase_aware_frontend.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/phase_aware_planner.h"
#include "clifft/sampling/state_queries.h"

#include <algorithm>
#include <cassert>
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
        throw std::invalid_argument(
            "amplitude output basis must have ceil(num_qubits / 64) words");
    }
    const uint32_t used_bits = num_qubits % 64U;
    if (!basis.empty() && used_bits != 0) {
        const uint64_t valid = (uint64_t{1} << used_bits) - 1U;
        if ((basis.back() & ~valid) != 0) {
            throw std::invalid_argument("amplitude output basis sets unused high bits");
        }
    }
}

void reverse_operation_targets(AstNode& node) {
    switch (gate_arity(node.gate)) {
        case GateArity::SINGLE:
            std::ranges::reverse(node.targets);
            return;
        case GateArity::PAIR: {
            std::vector<Target> reversed;
            reversed.reserve(node.targets.size());
            for (size_t end = node.targets.size(); end >= 2; end -= 2) {
                reversed.push_back(node.targets[end - 2]);
                reversed.push_back(node.targets[end - 1]);
            }
            node.targets = std::move(reversed);
            return;
        }
        case GateArity::MULTI:
        case GateArity::ANNOTATION:
            return;
        case GateArity::TRIPLE:
            throw std::invalid_argument(
                "amplitude adjoint does not support triple-target circuit operations");
    }
}

AstNode adjoint_node(const AstNode& source) {
    AstNode result(source);
    switch (source.gate) {
        case GateType::T:
            result.gate = GateType::T_DAG;
            break;
        case GateType::T_DAG:
            result.gate = GateType::T;
            break;
        case GateType::TPP:
            result.gate = GateType::TPP_DAG;
            break;
        case GateType::TPP_DAG:
            result.gate = GateType::TPP;
            break;
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
            result.args.at(0) = -result.args.at(0);
            break;
        case GateType::U3: {
            const double theta = result.args.at(0);
            const double phi = result.args.at(1);
            const double lambda = result.args.at(2);
            result.args = {-theta, -lambda, -phi};
            break;
        }
        case GateType::TICK:
            break;
        default:
            if (!is_clifford(source.gate)) {
                throw std::invalid_argument("amplitude adjoint does not support circuit gate " +
                                            std::string(gate_name(source.gate)));
            }
            result.gate = inverse_clifford_gate(source.gate);
            break;
    }
    reverse_operation_targets(result);
    return result;
}

Circuit adjoint_with_basis_input(const Circuit& source, std::span<const uint64_t> input_basis) {
    Circuit result = source.metadata_only_copy();
    result.nodes.reserve(source.nodes.size() + source.num_qubits);
    for (uint32_t q = 0; q < source.num_qubits; ++q) {
        if (((input_basis[q / 64U] >> (q % 64U)) & 1U) != 0) {
            result.nodes.push_back(AstNode{.gate = GateType::X,
                                           .targets = {Target::qubit(q)}});
        }
    }
    for (auto it = source.nodes.rbegin(); it != source.nodes.rend(); ++it) {
        result.nodes.push_back(adjoint_node(*it));
    }
    return result;
}

}  // namespace

BasisAmplitudeQuery::Prepared BasisAmplitudeQuery::prepare(
    const Circuit& circuit, std::span<const uint64_t> output_basis,
    std::complex<double> input_phase) {
    validate_output_basis(circuit.num_qubits, output_basis);

    const auto compile_orientation = [](const Circuit& oriented,
                                        std::span<const uint64_t> execution_basis,
                                        std::complex<double> phase,
                                        bool conjugate_result) -> Prepared {
        PhaseAwareHir traced = trace_phase_aware(oriented);
        if (!traced.hir.final_tableau.has_value()) {
            throw std::runtime_error("phase-aware trace did not retain its final Clifford frame");
        }

        // This pass only reorders commuting coherent operations. The ordinary
        // peephole pass is deliberately excluded until each projective fusion
        // has a phase-ledger counterpart.
        StatevectorSqueezePass{}.run(traced.hir);
        PhaseAwareSamplingPlan planned = plan_sampling_phase_aware(
            traced.hir, std::move(traced.final_clifford_frame));
        if (!planned.plan.final_tableau.has_value()) {
            throw std::runtime_error(
                "phase-aware planner did not retain its final Clifford frame");
        }
        const std::complex<double> frame_phase = internal::clifford_row_phase(
            *planned.plan.final_tableau, planned.final_clifford_frame, execution_basis);
        return Prepared{
            .plan = std::move(planned.plan),
            .execution_basis = {execution_basis.begin(), execution_basis.end()},
            .phase = phase * traced.source_scalar * frame_phase * planned.scalar,
            .conjugate_result = conjugate_result,
        };
    };

    Prepared forward = compile_orientation(circuit, output_basis, input_phase, false);

    // The output effect becomes a computational-basis input under the adjoint:
    // <x|U|0> = conj(<0|U^dagger|x>). Planning both orientations lets the
    // compiler choose the smaller exact contraction without changing kernels.
    const Circuit adjoint = adjoint_with_basis_input(circuit, output_basis);
    const std::vector<uint64_t> zero_basis(basis_word_count(circuit.num_qubits), 0);
    Prepared backward =
        compile_orientation(adjoint, zero_basis, std::conj(input_phase), true);
    if (backward.plan.peak_active_width < forward.plan.peak_active_width) {
        return backward;
    }
    return forward;
}

BasisAmplitudeQuery::BasisAmplitudeQuery(const Circuit& circuit,
                                         std::span<const uint64_t> output_basis,
                                         std::complex<double> input_phase)
    : BasisAmplitudeQuery(circuit, prepare(circuit, output_basis, input_phase)) {}

BasisAmplitudeQuery::BasisAmplitudeQuery(const Circuit& circuit, Prepared prepared)
    : plan_(std::move(prepared.plan)), output_basis_(std::move(prepared.execution_basis)),
      phase_(prepared.phase), conjugate_result_(prepared.conjugate_result) {
    assert(circuit.num_qubits == plan_.num_qubits());
}

std::complex<double> BasisAmplitudeQuery::evaluate() const {
    auto amplitudes = internal::basis_amplitudes(plan_, phase_, output_basis_, 1,
                                                 output_basis_.size());
    return conjugate_result_ ? std::conj(amplitudes.front()) : amplitudes.front();
}

}  // namespace clifft::sampling
