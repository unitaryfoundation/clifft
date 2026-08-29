#include "clifft/frontend/phase_aware_frontend.h"

#include "clifft/util/mask_view.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace clifft {

PhaseAwareCliffordFrame::NamedOperation::NamedOperation(GateType gate,
                                                        std::span<const uint32_t> targets)
    : gate_(gate) {
    const GateArity arity = gate_arity(gate);
    const size_t expected = arity == GateArity::SINGLE ? 1U : 2U;
    if (!is_clifford(gate) || (arity != GateArity::SINGLE && arity != GateArity::PAIR) ||
        targets.size() != expected) {
        throw std::invalid_argument(
            "phase-aware named operation requires a fixed one- or two-qubit Clifford gate");
    }
    std::ranges::copy(targets, targets_.begin());
}

PhaseAwareCliffordFrame::NamedOperation::NamedOperation(GateType gate,
                                                        std::initializer_list<uint32_t> targets)
    : NamedOperation(gate, std::span<const uint32_t>(targets.begin(), targets.size())) {}

std::span<const uint32_t> PhaseAwareCliffordFrame::NamedOperation::targets() const {
    const size_t count = gate_arity(gate_) == GateArity::SINGLE ? 1U : 2U;
    return std::span<const uint32_t>(targets_).first(count);
}

PhaseAwareCliffordFrame::PhaseAwareCliffordFrame(uint32_t num_qubits) : num_qubits_(num_qubits) {}

void PhaseAwareCliffordFrame::apply_named_gate(GateType gate, std::span<const uint32_t> targets) {
    source_operations_.emplace_back(NamedOperation(gate, targets));
}

void PhaseAwareCliffordFrame::apply_pauli_rotation(PauliStringView axis, bool dagger) {
    source_operations_.push_back(PauliRotation{PauliString(axis), dagger});
}

void PhaseAwareCliffordFrame::compose_input(std::span<const NamedOperation> operations) {
    input_operations_reversed_.reserve(input_operations_reversed_.size() + operations.size());
    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
        input_operations_reversed_.push_back(*it);
    }
}

StabilizerChForm PhaseAwareCliffordFrame::inverse_on_basis(std::span<const uint64_t> basis) const {
    if (basis.size() != mask_word_count(num_qubits_)) {
        throw std::invalid_argument("Clifford-frame basis width does not match the operator");
    }
    if (!mask_has_only_bits(MaskView{basis}, num_qubits_)) {
        throw std::invalid_argument("Clifford-frame basis sets unused high bits");
    }

    StabilizerChForm state(num_qubits_);
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (((basis[q / 64U] >> (q % 64U)) & 1U) != 0) {
            state.apply_x(q);
        }
    }

    auto apply_inverse = [&](const auto& operation) {
        using Operation = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<Operation, NamedOperation>) {
            state.apply_named_gate(inverse_fixed_clifford_gate(operation.gate()),
                                   operation.targets());
        } else {
            state.apply_pauli_rotation(operation.axis.view(), !operation.dagger);
        }
    };
    for (auto it = source_operations_.rbegin(); it != source_operations_.rend(); ++it) {
        std::visit(apply_inverse, *it);
    }
    for (const NamedOperation& operation : input_operations_reversed_) {
        apply_inverse(operation);
    }
    return state;
}

}  // namespace clifft
