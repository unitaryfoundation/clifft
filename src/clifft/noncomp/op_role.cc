#include "clifft/noncomp/op_role.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"

namespace clifft {

std::vector<QubitOperand> qubit_operands(const AstNode& node) {
    std::vector<QubitOperand> operands;

    // MPAD pads the measurement record; its targets are classical 0/1
    // literals, not qubit indices.
    if (node.gate == GateType::MPAD) {
        return operands;
    }

    // A record control marks a classically-controlled feedback node; the
    // parser only allows that on CX/CZ, with the record target first.
    bool feedback = false;
    for (const Target& target : node.targets) {
        if (target.is_rec()) {
            feedback = true;
            break;
        }
    }

    const OperandRole role = feedback ? OperandRole::Feedback : OperandRole::Physical;
    for (const Target& target : node.targets) {
        if (target.is_rec()) {
            continue;  // record reference, not a qubit operand
        }
        operands.push_back(QubitOperand{target.value(), role});
    }
    return operands;
}

}  // namespace clifft
