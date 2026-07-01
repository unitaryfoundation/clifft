#include "clifft/noncomp/rewriter.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

AstNode single_qubit_op(GateType gate, uint32_t qubit) {
    return AstNode{gate, {Target::qubit(qubit)}, {}, 0};
}

// A hidden carrier edit appended after an op for one operand's jump: an R
// collapses and rezeros the carrier, and an X then prepares |1> when the
// jump lands on the |1> computational level.
struct CarrierEdit {
    uint32_t qubit;
    bool prepare_one;
};

}  // namespace

Circuit rewrite(const Circuit& original, const NonComputationalHistory& history,
                const NonComputationalModel& model) {
    const LevelSet& levels = model.levels();
    const NonComputationalPolicy& policy = model.policy();

    if (history.initial_status.size() != original.num_qubits) {
        throw std::invalid_argument("rewrite: history has " +
                                    std::to_string(history.initial_status.size()) +
                                    " initial statuses but circuit declares " +
                                    std::to_string(original.num_qubits) + " qubits");
    }

    // Copy so every circuit-level field carries over (and any field added to
    // Circuit later is not silently dropped); only the node list is rebuilt.
    // The inserted X-prep, hidden R, and destination-prep X ops are not
    // visible measurements, so the record-layout counts stay valid.
    Circuit out = original;
    out.nodes.clear();
    out.nodes.reserve(original.nodes.size() + original.num_qubits);

    std::vector<QubitStatus> status = history.initial_status;

    // Initial-state prep: a sampled known |1> initial level needs a leading X
    // so the SVM's |0> matches it.
    for (uint32_t q = 0; q < original.num_qubits; ++q) {
        const QubitStatus& s = status[q];
        if (s.kind() == QubitStatusKind::ComputationalKnown &&
            s.level_id() == levels.computational_one_id()) {
            out.nodes.push_back(single_qubit_op(GateType::X, q));
        }
    }

    size_t trans_cursor = 0;

    for (uint32_t op_index = 0; op_index < original.nodes.size(); ++op_index) {
        const AstNode& node = original.nodes[op_index];
        const GateType gate = node.gate;
        const TransitionInstrument* instrument = model.transition_for(gate);

        // Policy pre-scan over entry statuses: any rejecting operand rejects
        // the whole operation; otherwise any dropping operand drops it whole
        // (identity on the surviving operands). The scan precedes the
        // transition replay so a surviving operand's stepping can know the
        // base operation is gone.
        bool drop_op = false;
        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            if (qubit >= status.size()) {
                throw std::invalid_argument("rewrite: operand qubit " + std::to_string(qubit) +
                                            " is out of range at op " + std::to_string(op_index));
            }
            switch (operand_action(gate, status[qubit].kind(), policy)) {
                case OperandAction::Reject:
                    throw std::invalid_argument(
                        "rewrite: operation '" + std::string(gate_name(gate)) + "' on a " +
                        kind_name(status[qubit].kind()) + " qubit " + std::to_string(qubit) +
                        " at op " + std::to_string(op_index) + " is not representable; rejecting");
                case OperandAction::Drop:
                    drop_op = true;
                    break;
                case OperandAction::Apply:
                    break;
            }
        }

        std::vector<CarrierEdit> carrier_edits;  // hidden edits appended after this op

        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            const QubitStatus pre = status[qubit];

            // Replay the recorded transition for this operand, if the sampler
            // consulted one (a Physical operand of a gate declaring a
            // transition). Records are consumed in sampler order.
            TransitionOutcome outcome;
            if (operand.role == OperandRole::Physical && instrument != nullptr) {
                if (trans_cursor >= history.transitions.size()) {
                    throw std::invalid_argument(
                        "rewrite: circuit consults more transitions than the history records; "
                        "history does not describe this circuit");
                }
                const TransitionRecord& record = history.transitions[trans_cursor++];
                if (record.op_index != op_index || record.qubit != qubit) {
                    throw std::invalid_argument(
                        "rewrite: transition record (op " + std::to_string(record.op_index) +
                        ", qubit " + std::to_string(record.qubit) +
                        ") does not match consult (op " + std::to_string(op_index) + ", qubit " +
                        std::to_string(qubit) + "); history does not describe this circuit");
                }
                outcome.jumped = record.jumped;
                outcome.destination_level = record.destination_level;
            }

            // Carrier-edit decision for a jump. A jump into the computational
            // subspace materializes the carrier at the destination level: the
            // R collapses whatever the base op left -- a coherent
            // superposition, a definite level, or a stale residual on a
            // recaptured leaked/lost qubit -- and the X then prepares |1> for
            // a One-level destination. A jump out of the computational
            // subspace needs a hidden Z-basis unraveling (trace-out) only
            // when the carrier the base op would leave with no jump is
            // coherent; a dropped operation leaves the entry carrier.
            if (outcome.jumped) {
                const Level& dest = levels.at(outcome.destination_level);
                if (dest.category == LevelCategory::Computational) {
                    carrier_edits.push_back(
                        {qubit, outcome.destination_level == levels.computational_one_id()});
                } else {
                    const QubitStatus post_if_no_jump =
                        drop_op ? pre
                                : normal_post_op_status(pre, gate, operand.role, policy, levels);
                    if (post_if_no_jump.kind() == QubitStatusKind::ComputationalUnknown) {
                        carrier_edits.push_back({qubit, false});
                    }
                }
            }

            status[qubit] = drop_op ? step_status_dropped(pre, outcome, levels)
                                    : step_status(pre, gate, operand.role, outcome, policy, levels);
        }

        if (!drop_op) {
            out.nodes.push_back(node);
        }
        for (const CarrierEdit& edit : carrier_edits) {
            out.nodes.push_back(single_qubit_op(GateType::R, edit.qubit));
            if (edit.prepare_one) {
                out.nodes.push_back(single_qubit_op(GateType::X, edit.qubit));
            }
        }
    }

    if (trans_cursor != history.transitions.size()) {
        throw std::invalid_argument(
            "rewrite: history records more transitions than the circuit consults; "
            "history does not describe this circuit");
    }

    return out;
}

}  // namespace clifft
