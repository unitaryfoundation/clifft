#include "clifft/noncomp/transition_hooks.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/status_walk.h"

#include <stdexcept>

namespace clifft {

namespace {
bool has_enabled_partner_effect(const NonComputationalModel& model, GateType gate) {
    for (uint32_t source = 0; source < 2; ++source) {
        for (const auto status : {InteractionSource::Leaked, InteractionSource::Lost}) {
            const PartnerEffect* effect = model.partner_effect(gate, source, status);
            if (effect != nullptr && !effect->empty()) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

Circuit expand_transition_hooks(const Circuit& circuit, const NonComputationalModel& model) {
    const auto& hooks = model.transition_hooks();

    Circuit out = circuit.metadata_only_copy();
    out.nodes.reserve(circuit.nodes.size() * 2);

    for (const AstNode& node : circuit.nodes) {
        out.nodes.push_back(node);
        if (supports_partner_effect(node.gate) && has_enabled_partner_effect(model, node.gate)) {
            const auto operands = qubit_operands(node);
            if (operands.size() == 2 && operands[0].role == OperandRole::Physical &&
                operands[1].role == OperandRole::Physical) {
                for (uint32_t source = 0; source < 2; ++source) {
                    for (const auto status : {InteractionSource::Leaked, InteractionSource::Lost}) {
                        const PartnerEffect* effect =
                            model.partner_effect(node.gate, source, status);
                        if (effect == nullptr || effect->empty()) {
                            continue;
                        }
                        const bool leakage = status == InteractionSource::Leaked;
                        std::vector<double> args(effect->pauli.begin(), effect->pauli.end());
                        if (leakage) {
                            args.push_back(effect->spread_probability);
                        }
                        out.nodes.push_back(AstNode{
                            leakage ? GateType::LEAKAGE_INTERACTION : GateType::LOSS_INTERACTION,
                            {node.targets[source], node.targets[1 - source]},
                            std::move(args),
                            node.source_line});
                    }
                }
            } else if (operands.size() != 1 || operands[0].role != OperandRole::Feedback) {
                throw std::invalid_argument(
                    "configured partner effects require one physical gate pair per node");
            }
        }
        const auto hook = hooks.find(node.gate);
        if (hook == hooks.end()) {
            continue;
        }
        // Add a transition for each qubit the gate acts on directly.
        // Record-controlled feedback does not physically execute the gate.
        for (const QubitOperand& operand : qubit_operands(node)) {
            if (operand.role != OperandRole::Physical) {
                continue;
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
