#include "clifft/sampling/executor.h"
#include "clifft/svm/svm.h"
#include "clifft/svm/svm_math.h"

#include <bit>
#include <cassert>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

constexpr uint32_t kMaxStatevectorQubits = 10;

void validate_statevector_size(uint32_t num_qubits) {
    if (num_qubits > kMaxStatevectorQubits) {
        throw std::runtime_error(
            "Statevector expansion limited to 10 qubits (dense U_C matrix is 4^n)");
    }
}

template <typename CoefficientAt>
std::vector<std::complex<double>> expand_factored_state(
    uint32_t num_qubits, uint32_t active_width, uint64_t active_size, uint64_t pauli_x,
    uint64_t pauli_z, const stim::Tableau<kStimWidth>* final_tableau, std::complex<double> scale,
    CoefficientAt coefficient_at) {
    validate_statevector_size(num_qubits);
    const uint64_t dimension = uint64_t{1} << num_qubits;
    assert(active_width <= num_qubits && "active width exceeds physical qubit count");
    assert(active_size == (uint64_t{1} << active_width) &&
           "active coefficient count does not match active width");

    // Active coordinates occupy the low bits and dormant coordinates are
    // |0>. Apply the virtual Pauli frame while embedding so the dense
    // expansion needs only the framed and physical arrays.
    std::vector<std::complex<double>> framed(dimension, {0.0, 0.0});
    for (uint64_t index = 0; index < active_size; ++index) {
        const uint64_t target = index ^ pauli_x;
        const double sign = (std::popcount(index & pauli_z) & 1U) != 0 ? -1.0 : 1.0;
        framed[target] = coefficient_at(index) * sign;
    }

    std::vector<std::complex<double>> physical;
    if (final_tableau == nullptr) {
        physical = std::move(framed);
    } else {
        physical.assign(dimension, {0.0, 0.0});
        const auto unitary = final_tableau->to_flat_unitary_matrix(true);
        for (uint64_t row = 0; row < dimension; ++row) {
            std::complex<double> amplitude{0.0, 0.0};
            for (uint64_t col = 0; col < dimension; ++col) {
                const auto factor = unitary[row * dimension + col];
                amplitude += std::complex<double>{factor.real(), factor.imag()} * framed[col];
            }
            physical[row] = amplitude;
        }
    }

    for (std::complex<double>& amplitude : physical) {
        amplitude *= scale;
    }
    return physical;
}

}  // namespace

std::vector<std::complex<double>> get_statevector(const CompiledModule& program,
                                                  const SchrodingerState& state) {
    validate_statevector_size(program.num_qubits);
    uint64_t pauli_x = 0;
    uint64_t pauli_z = 0;
    for (uint32_t q = 0; q < program.num_qubits; ++q) {
        if (bit_get(state.p_x, q)) {
            pauli_x |= uint64_t{1} << q;
        }
        if (bit_get(state.p_z, q)) {
            pauli_z |= uint64_t{1} << q;
        }
    }
    const stim::Tableau<kStimWidth>* final_tableau =
        program.constant_pool.final_tableau ? &*program.constant_pool.final_tableau : nullptr;
    return expand_factored_state(program.num_qubits, state.active_k, state.v_size(), pauli_x,
                                 pauli_z, final_tableau,
                                 state.gamma() * program.constant_pool.global_weight,
                                 [&](uint64_t index) { return state.v()[index]; });
}

namespace sampling {

std::vector<std::complex<double>> get_statevector(const ExecutablePlan& plan) {
    const stim::Tableau<kStimWidth>* final_tableau = plan.final_state_tableau();
    if (final_tableau == nullptr) {
        throw std::invalid_argument(
            "get_statevector() requires pure-state unitary evolution: measurements, feedback, "
            "noise, readout noise, transition instruments, detectors, postselection, and "
            "observables are not supported. EXP_VAL probes are allowed but their outputs are "
            "ignored. Use DropNonUnitaryPass only if you intentionally want to inspect the "
            "unitary skeleton of a mixed circuit.");
    }

    Executor executor(plan);
    executor.run_shot();
    const State& state = executor.state();
    return expand_factored_state(plan.num_qubits(), state.active_width(), state.size(), 0, 0,
                                 final_tableau, state.global_scalar(), [&](uint64_t index) {
                                     return std::complex<double>{state.real_data()[index],
                                                                 state.imag_data()[index]};
                                 });
}

}  // namespace sampling
}  // namespace clifft
