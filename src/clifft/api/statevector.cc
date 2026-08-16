#include "clifft/sampling/executor.h"
#include "clifft/sampling/state_queries.h"

#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

constexpr uint32_t kMaxStatevectorQubits = 10;

std::complex<double> exact_clifford_factor(std::complex<float> factor, double support_magnitude) {
    if (factor == std::complex<float>{0.0F, 0.0F}) {
        return {0.0, 0.0};
    }

    // Stim canonicalizes a stabilizer state to a uniform magnitude times one
    // of {1, -1, i, -i}. Recover those exact classes before multiplying the
    // double-precision active state instead of preserving float rounding in
    // the public result.
    assert((factor.real() == 0.0F) != (factor.imag() == 0.0F));
    if (factor.real() > 0.0F) {
        return {support_magnitude, 0.0};
    }
    if (factor.real() < 0.0F) {
        return {-support_magnitude, 0.0};
    }
    return factor.imag() > 0.0F ? std::complex<double>{0.0, support_magnitude}
                                : std::complex<double>{0.0, -support_magnitude};
}

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
    static_cast<void>(active_width);

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
            uint64_t support = 0;
            for (uint64_t col = 0; col < dimension; ++col) {
                support += unitary[row * dimension + col] != std::complex<float>{0.0F, 0.0F};
            }
            assert(support != 0 && "Clifford unitary row has empty support");
            const double support_magnitude = 1.0 / std::sqrt(static_cast<double>(support));
            std::complex<double> amplitude{0.0, 0.0};
            for (uint64_t col = 0; col < dimension; ++col) {
                amplitude +=
                    exact_clifford_factor(unitary[row * dimension + col], support_magnitude) *
                    framed[col];
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
