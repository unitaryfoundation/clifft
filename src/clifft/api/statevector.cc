#include "clifft/sampling/executor.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/sampling/state_query_limits.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

static_assert(sampling::kMaxExpandedStatevectorQubits <= 64,
              "Dense Pauli application reads one mask word");

void validate_statevector_size(uint32_t num_qubits) {
    if (num_qubits > sampling::kMaxExpandedStatevectorQubits) {
        throw std::runtime_error("Statevector expansion limited to 10 qubits");
    }
}

void apply_pauli(PauliStringView pauli, std::span<const std::complex<double>> input,
                 std::span<std::complex<double>> output) {
    assert(input.size() == output.size());
    const uint64_t x = pauli.x().words.empty() ? 0 : pauli.x().words[0];
    const uint64_t z = pauli.z().words.empty() ? 0 : pauli.z().words[0];
    const std::complex<double> phase = i_power(pauli.phase());
    for (uint64_t basis = 0; basis < input.size(); ++basis) {
        const double z_sign = (std::popcount(basis & z) & 1U) != 0 ? -1.0 : 1.0;
        output[basis ^ x] = phase * z_sign * input[basis];
    }
}

std::vector<std::complex<double>> zero_state_image(const Tableau& tableau) {
    const uint64_t dimension = uint64_t{1} << tableau.num_qubits();
    std::vector<std::complex<double>> state(dimension);
    std::vector<std::complex<double>> transformed(dimension);

    // At least one computational basis vector has nonzero overlap with the
    // stabilizer state. The first survivor fixes a deterministic global phase
    // with its first nonzero amplitude positive real.
    for (uint64_t seed = 0; seed < dimension; ++seed) {
        std::ranges::fill(state, std::complex<double>{0.0, 0.0});
        state[seed] = {1.0, 0.0};
        bool survives = true;
        for (uint32_t q = 0; q < tableau.num_qubits(); ++q) {
            apply_pauli(tableau.z_output(q), state, transformed);
            double norm = 0.0;
            for (uint64_t basis = 0; basis < dimension; ++basis) {
                state[basis] += transformed[basis];
                norm += std::norm(state[basis]);
            }
            // A stabilizer projection has exact squared norm 0, 2, or 4 before
            // normalization. Leave a wide gap so cancellation residue cannot
            // turn a dead seed into a numerically amplified state.
            if (norm < 1.0) {
                survives = false;
                break;
            }
            const double scale = 1.0 / std::sqrt(norm);
            for (auto& amplitude : state) {
                amplitude *= scale;
            }
        }
        if (survives) {
            return state;
        }
    }
    throw std::logic_error("Clifford tableau has no stabilized state");
}

std::vector<std::complex<double>> apply_tableau(const Tableau& tableau,
                                                std::span<const std::complex<double>> input) {
    const uint64_t dimension = uint64_t{1} << tableau.num_qubits();
    assert(input.size() == dimension);
    const std::vector<std::complex<double>> zero_image = zero_state_image(tableau);
    std::vector<std::complex<double>> column(dimension);
    std::vector<std::complex<double>> output(dimension);
    for (uint64_t basis = 0; basis < dimension; ++basis) {
        if (input[basis] == std::complex<double>{0.0, 0.0}) {
            continue;
        }
        PauliString virtual_x(tableau.num_qubits());
        for (uint32_t q = 0; q < tableau.num_qubits(); ++q) {
            if (((basis >> q) & 1U) != 0) {
                virtual_x.set_pauli(q, true, false);
            }
        }
        const PauliString physical_x = tableau.apply(virtual_x.view());
        apply_pauli(physical_x.view(), zero_image, column);
        for (uint64_t row = 0; row < dimension; ++row) {
            output[row] += column[row] * input[basis];
        }
    }
    return output;
}

template <typename CoefficientAt>
std::vector<std::complex<double>> expand_factored_state(uint32_t num_qubits, uint32_t active_width,
                                                        uint64_t active_size, uint64_t pauli_x,
                                                        uint64_t pauli_z,
                                                        const Tableau* final_tableau,
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
        physical = apply_tableau(*final_tableau, framed);
    }

    return physical;
}

}  // namespace

namespace sampling {

std::vector<std::complex<double>> get_statevector(const ExecutablePlan& plan) {
    const Tableau* final_tableau = plan.final_state_tableau();
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
                                 final_tableau, [&](uint64_t index) {
                                     return std::complex<double>{state.real_data()[index],
                                                                 state.imag_data()[index]};
                                 });
}

}  // namespace sampling
}  // namespace clifft
