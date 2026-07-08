#include "clifft/noncomp/op_role.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/status_step.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace clifft {

std::vector<QubitOperand> qubit_operands(const AstNode& node) {
    std::vector<QubitOperand> operands;

    // MPAD pads the measurement record; its targets are classical 0/1
    // literals, not qubit indices.
    if (node.gate == GateType::MPAD) {
        return operands;
    }

    bool has_rec = false;
    for (const Target& target : node.targets) {
        if (target.is_rec()) {
            has_rec = true;
            break;
        }
    }

    OperandRole role = OperandRole::Physical;
    if (has_rec) {
        // A record target accompanies a qubit only as CX/CZ classical
        // feedback (conditional X can flip g<->e; conditional Z is
        // phase-only). Otherwise it is a rec-only annotation -- DETECTOR,
        // OBSERVABLE_INCLUDE, READOUT_NOISE -- which has no qubit operands.
        // A record alongside a qubit on any other gate is a shape this
        // layer does not model.
        if (node.gate == GateType::CX || node.gate == GateType::CZ) {
            role = OperandRole::Feedback;
        } else {
            for (const Target& target : node.targets) {
                if (!target.is_rec()) {
                    throw std::invalid_argument(
                        "qubit_operands: record-controlled qubit target on gate '" +
                        std::string(gate_name(node.gate)) + "'; expected CX/CZ feedback");
                }
            }
            return operands;  // rec-only annotation: no qubit operands
        }
    }

    for (const Target& target : node.targets) {
        if (target.is_rec()) {
            continue;  // record reference, not a qubit operand
        }
        operands.push_back(QubitOperand{target.value(), role});
    }
    return operands;
}

OrdinaryStep advance_ordinary_node(const AstNode& node, uint32_t op_index,
                                   std::vector<QubitStatus>& status,
                                   const NonComputationalPolicy& policy, std::string_view caller) {
    const GateType gate = node.gate;

    // Policy pre-scan over entry statuses: any rejecting operand rejects
    // the whole operation; otherwise any dropping operand drops it whole
    // (identity on the surviving operands).
    bool drop_op = false;
    for (const QubitOperand& operand : qubit_operands(node)) {
        const uint32_t qubit = operand.qubit;
        if (qubit >= status.size()) {
            throw std::invalid_argument(std::string(caller) + ": operand qubit " +
                                        std::to_string(qubit) + " is out of range at op " +
                                        std::to_string(op_index));
        }
        switch (operand_action(gate, status[qubit], policy)) {
            case OperandAction::Reject:
                throw std::invalid_argument(
                    std::string(caller) + ": operation '" + std::string(gate_name(gate)) +
                    "' on a " + status_name(status[qubit]) + " qubit " + std::to_string(qubit) +
                    " at op " + std::to_string(op_index) + " is not representable; rejecting");
            case OperandAction::Drop:
                drop_op = true;
                break;
            case OperandAction::Apply:
                break;
        }
    }

    // Set when this (single-qubit Z-basis) measurement reads a leaked or
    // lost qubit: the classifier, not the SVM, defines its record bit.
    std::optional<Level> classified_level;

    for (const QubitOperand& operand : qubit_operands(node)) {
        const uint32_t qubit = operand.qubit;
        const QubitStatus pre = status[qubit];

        if (is_measurement(gate) && !is_computational(pre)) {
            classified_level = noncomp_level(pre);
        }

        status[qubit] = drop_op ? pre : normal_post_op_status(pre, gate, operand.role, policy);
    }

    return OrdinaryStep{drop_op, classified_level};
}

}  // namespace clifft
