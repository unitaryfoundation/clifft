#include "clifft/noncomp/annotate.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/op_role.h"

namespace clifft {

Circuit annotate(const Circuit& circuit, const NonComputationalModel& model) {
    const auto& hooks = model.transition_hooks();

    Circuit out = circuit;
    out.nodes.clear();
    out.nodes.reserve(circuit.nodes.size() * 2);

    for (const AstNode& node : circuit.nodes) {
        out.nodes.push_back(node);
        const auto hook = hooks.find(node.gate);
        if (hook == hooks.end()) {
            continue;
        }
        // One annotation per Physical operand: feedback corrections are
        // virtual and fire no transition.
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand.role != OperandRole::Physical) {
                continue;  // Feedback corrections are virtual; no transition fires
            }
            out.nodes.push_back(AstNode{GateType::LEVEL_TRANSITION,
                                        {Target::qubit(operand.qubit)},
                                        {},
                                        node.source_line,
                                        hook->second});
        }
    }
    return out;
}

}  // namespace clifft
