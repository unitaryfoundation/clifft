#include "clifft/noncomp/op_role.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"

#include <stdexcept>
#include <string>

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

}  // namespace clifft
