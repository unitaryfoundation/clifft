#include "clifft/noncomp/rewriter.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

// How the base operation is handled for one qubit operand.
enum class OpAction { Apply, Drop, Reject };

const char* kind_name(QubitStatusKind k) {
    switch (k) {
        case QubitStatusKind::ComputationalUnknown:
            return "ComputationalUnknown";
        case QubitStatusKind::ComputationalKnown:
            return "ComputationalKnown";
        case QubitStatusKind::Leaked:
            return "Leaked";
        case QubitStatusKind::Lost:
            return "Lost";
    }
    return "unknown";
}

// Trajectory policy for one operand, keyed on the operand's status at op
// entry. A computational qubit (known or unknown) runs every operation; the
// table below only governs leaked and lost operands. Aggregated across an
// operation's operands by the caller: any Reject rejects the whole op, and
// Drop only ever arises for a single-operand op.
OpAction operand_action(GateType gate, QubitStatusKind kind, const NonComputationalPolicy& policy) {
    if (kind == QubitStatusKind::ComputationalKnown ||
        kind == QubitStatusKind::ComputationalUnknown) {
        return OpAction::Apply;
    }

    const bool lost = kind == QubitStatusKind::Lost;

    // An identity no-op is harmless to keep on any qubit.
    if (is_identity_noop(gate)) {
        return OpAction::Apply;
    }
    // Measurements are kept so the visible record and its rec[-k] references
    // do not shift; the leaked/lost outcome is reinterpreted downstream.
    if (is_measurement(gate)) {
        return OpAction::Apply;
    }
    // A reset restores a leaked qubit always, a lost qubit only by policy;
    // a lost-qubit reset otherwise rejects.
    if (is_reset(gate)) {
        return (!lost || policy.reset_restores_lost) ? OpAction::Apply : OpAction::Reject;
    }
    // A single-qubit Pauli noise channel drops on a leaked or lost qubit; a
    // single-qubit unitary gate drops on a lost qubit (no carrier remains).
    if (gate_arity(gate) == GateArity::SINGLE) {
        if (is_noise_gate(gate)) {
            return OpAction::Drop;
        }
        if (lost) {
            return OpAction::Drop;
        }
        // A single-qubit gate on a leaked qubit rejects by default.
        return OpAction::Reject;
    }
    // Anything else touching a leaked or lost operand -- a two-qubit gate,
    // a multi-qubit measurement-class op, classical feedback onto a vacated
    // site -- is ambiguous and rejects by default.
    return OpAction::Reject;
}

AstNode single_qubit_op(GateType gate, uint32_t qubit) {
    return AstNode{gate, {Target::qubit(qubit)}, {}, 0};
}

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

    Circuit out;
    out.num_qubits = original.num_qubits;
    // Only X-prep and trace-out R ops are inserted, neither of which is a
    // visible measurement, so the record-layout counts carry over unchanged.
    out.num_measurements = original.num_measurements;
    out.num_detectors = original.num_detectors;
    out.num_observables = original.num_observables;
    out.num_exp_vals = original.num_exp_vals;
    out.nodes.reserve(original.nodes.size() + original.num_qubits);

    std::vector<QubitStatus> status = history.initial_status;

    // Initial-state prep: a sampled known |1> initial level needs a leading X
    // so the SVM's |0> matches it.
    for (uint32_t q = 0; q < original.num_qubits; ++q) {
        const QubitStatus& s = status[q];
        if (s.kind() == QubitStatusKind::ComputationalKnown &&
            levels.at(s.level_id()).basis_bit == BasisBit::One) {
            out.nodes.push_back(single_qubit_op(GateType::X, q));
        }
    }

    size_t trans_cursor = 0;

    for (uint32_t op_index = 0; op_index < original.nodes.size(); ++op_index) {
        const AstNode& node = original.nodes[op_index];
        const GateType gate = node.gate;
        const TransitionInstrument* instrument = model.transition_for(gate);

        bool drop_op = false;
        std::vector<uint32_t> trace_out;  // qubits needing a trace-out R after this op

        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            if (qubit >= status.size()) {
                throw std::invalid_argument("rewrite: operand qubit " + std::to_string(qubit) +
                                            " is out of range at op " + std::to_string(op_index));
            }
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

            switch (operand_action(gate, pre.kind(), policy)) {
                case OpAction::Reject:
                    throw std::invalid_argument(
                        "rewrite: operation '" + std::string(gate_name(gate)) + "' on a " +
                        kind_name(pre.kind()) + " qubit " + std::to_string(qubit) + " at op " +
                        std::to_string(op_index) + " is not representable; rejecting");
                case OpAction::Drop:
                    drop_op = true;  // single-operand op
                    break;
                case OpAction::Apply:
                    break;
            }

            // Trace-out decision: the carrier state the base op would leave
            // with no jump. A jump to a noncomputational level from a coherent
            // carrier needs a hidden Z-basis unraveling.
            if (outcome.jumped) {
                const QubitStatus post_if_no_jump =
                    normal_post_op_status(pre, gate, operand.role, policy, levels);
                const QubitStatusKind dest = levels.status_for(outcome.destination_level).kind();
                if ((dest == QubitStatusKind::Leaked || dest == QubitStatusKind::Lost) &&
                    post_if_no_jump.kind() == QubitStatusKind::ComputationalUnknown) {
                    trace_out.push_back(qubit);
                }
            }

            status[qubit] = step_status(pre, gate, operand.role, outcome, policy, levels);
        }

        if (!drop_op) {
            out.nodes.push_back(node);
        }
        for (uint32_t qubit : trace_out) {
            out.nodes.push_back(single_qubit_op(GateType::R, qubit));
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
