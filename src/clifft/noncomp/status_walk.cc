#include "clifft/noncomp/status_walk.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/util/numeric.h"

#include <stdexcept>
#include <string>

namespace clifft {

double inline_transition_probability(GateType gate, const std::vector<double>& args,
                                     uint32_t op_index, std::string_view caller) {
    if (gate != GateType::LEAKAGE && gate != GateType::LOSS) {
        throw std::invalid_argument(
            std::string(caller) + ": inline transition probability requested for '" +
            std::string(gate_name(gate)) + "' at op " + std::to_string(op_index));
    }
    const std::string name{gate_name(gate)};
    if (args.size() != 1) {
        throw std::invalid_argument(std::string(caller) + ": " + name + " at op " +
                                    std::to_string(op_index) +
                                    " requires exactly one argument (the probability)");
    }
    const double p = args[0];
    if (!is_probability(p)) {
        throw std::invalid_argument(std::string(caller) + ": " + name + " probability at op " +
                                    std::to_string(op_index) + " = " + std::to_string(p) +
                                    " is not finite or is out of [0, 1]");
    }
    return p;
}

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
        // feedback. Otherwise it is a rec-only annotation.
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
            return operands;
        }
    }

    for (const Target& target : node.targets) {
        if (!target.is_rec()) {
            operands.push_back(QubitOperand{target.value(), role});
        }
    }
    return operands;
}

OperandAction operand_action(GateType gate, QubitStatus status,
                             const NonComputationalPolicy& policy) {
    if (is_computational(status)) {
        return OperandAction::Apply;
    }

    const bool lost = is_lost(status);
    if (is_identity_noop(gate)) {
        return OperandAction::Apply;
    }
    if (is_measure_reset(gate)) {
        return OperandAction::Apply;
    }
    if (is_measurement(gate)) {
        return (gate == GateType::M || gate == GateType::MX || gate == GateType::MY)
                   ? OperandAction::Apply
                   : OperandAction::Reject;
    }
    if (is_reset(gate)) {
        return !lost || policy.reset_restores_lost ? OperandAction::Apply : OperandAction::Drop;
    }
    if (gate == GateType::CORRELATED_ERROR || gate == GateType::ELSE_CORRELATED_ERROR) {
        return OperandAction::Apply;
    }
    return OperandAction::Drop;
}

QubitStatus normal_post_op_status(QubitStatus entry, GateType gate, OperandRole role,
                                  const NonComputationalPolicy& policy) {
    if (role == OperandRole::Feedback) {
        return entry;
    }

    if (is_leaked(entry) || is_lost(entry)) {
        const bool reset = is_reset(gate) || is_measure_reset(gate);
        const bool restorable = is_leaked(entry) || policy.reset_restores_lost;
        return reset && restorable ? QubitStatus::Computational : entry;
    }
    return entry;
}

OrdinaryStep advance_ordinary_node(const AstNode& node, uint32_t op_index,
                                   std::vector<QubitStatus>& status,
                                   const NonComputationalPolicy& policy, std::string_view caller) {
    const GateType gate = node.gate;
    const std::vector<QubitOperand> operands = qubit_operands(node);

    // Decide whether the whole operation applies before updating any status:
    // one dropped operand makes a multi-qubit operation a no-op for all.
    bool drop_op = false;
    for (const QubitOperand& operand : operands) {
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

    std::optional<ClassifiedOperand> classified;
    for (const QubitOperand& operand : operands) {
        const uint32_t qubit = operand.qubit;
        const QubitStatus pre = status[qubit];
        if (is_measurement(gate) && !is_computational(pre)) {
            classified = ClassifiedOperand{qubit, noncomp_level(pre)};
        }
        status[qubit] = drop_op ? pre : normal_post_op_status(pre, gate, operand.role, policy);
    }

    return OrdinaryStep{drop_op, classified};
}

}  // namespace clifft
