#include "clifft/noncomp/rewriter.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/numeric.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"
#include "clifft/noncomp/transition_instrument.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace clifft {

namespace {

AstNode single_qubit_op(GateType gate, uint32_t qubit) {
    return AstNode{gate, {Target::qubit(qubit)}, {}, 0};
}

// MPAD's target is the padded record bit itself (a 0/1 literal).
AstNode mpad_op(uint8_t bit) {
    return AstNode{GateType::MPAD, {Target::qubit(bit)}, {}, 0};
}

// Classical bit-flip on an absolute visible record slot, drawn at sample
// time inside the VM.
AstNode readout_noise_op(uint32_t slot, double prob) {
    return AstNode{GateType::READOUT_NOISE, {Target::rec(slot)}, {prob}, 0};
}

GateType reset_for(GateType measure_reset) {
    switch (measure_reset) {
        case GateType::MRX:
            return GateType::RX;
        case GateType::MRY:
            return GateType::RY;
        default:
            return GateType::R;  // MR
    }
}

// Validate that the model's classifier can supply the record bit for a
// measurement on a noncomputational qubit at `level`: it must exist, have
// two or three symbols, and offer a stochastic (non-reject) column for the
// level. Returns the classifier on success.
const MeasurementClassifier& classifier_for(const NonComputationalModel& model, uint8_t level,
                                            GateType gate, uint32_t op_index, uint32_t qubit) {
    const MeasurementClassifier* classifier = model.classifier();
    if (classifier == nullptr) {
        throw std::invalid_argument("rewrite: measurement '" + std::string(gate_name(gate)) +
                                    "' on a noncomputational qubit " + std::to_string(qubit) +
                                    " at op " + std::to_string(op_index) +
                                    " requires a classifier, but the model has none");
    }
    // The record write is one visible bit. Symbols 0/1 are that bit; a third
    // symbol is the herald. A still-richer alphabet has no defined mapping
    // onto (bit, herald) and is not representable.
    if (classifier->num_symbols() != 2 && classifier->num_symbols() != 3) {
        throw std::invalid_argument(
            "rewrite: a measurement on a noncomputational qubit requires a two- or three-symbol "
            "classifier, but the model's has " +
            std::to_string(classifier->num_symbols()) + " symbols (measurement '" +
            std::string(gate_name(gate)) + "' on qubit " + std::to_string(qubit) + " at op " +
            std::to_string(op_index) + ")");
    }
    // A substochastic column reserves probability for a reject (heralded
    // abort) outcome that has no binary record bit. That path is not wired
    // through this entry point yet, so the column must sum to one within
    // tolerance; a reject column is an explicit error rather than a silently
    // dropped or aborted shot.
    const double reject = classifier->reject_probability(level);
    if (reject > kProbTolerance) {
        throw std::invalid_argument(
            "rewrite: classifier reject columns are not supported yet; the column for level " +
            std::to_string(level) + " sums to less than one (reject probability " +
            std::to_string(reject) + ", measurement '" + std::string(gate_name(gate)) +
            "' on qubit " + std::to_string(qubit) + " at op " + std::to_string(op_index) + ")");
    }
    return *classifier;
}

// A hidden carrier edit appended after an op for one operand's jump: an R
// collapses and rezeros the carrier, and an X then prepares |1> when the
// jump lands on the |1> computational level.
struct CarrierEdit {
    uint32_t qubit;
    bool prepare_one;
};

}  // namespace

RewriteResult rewrite(const Circuit& original, const NonComputationalHistory& history,
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
    // visible measurements, and a classifier record write pads the same slot
    // its measurement occupied, so the record-layout counts stay valid.
    RewriteResult result;
    Circuit& out = result.circuit;
    out = original;
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
    uint32_t slot = 0;  // visible measurement record index

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

        // Set when this (single-qubit Z-basis) measurement reads a leaked or
        // lost qubit: the classifier, not the SVM, defines its record bit.
        std::optional<uint8_t> classified_level;

        for (const QubitOperand& operand : qubit_operands(node)) {
            const uint32_t qubit = operand.qubit;
            const QubitStatus pre = status[qubit];

            if (is_measurement(gate) &&
                (pre.kind() == QubitStatusKind::Leaked || pre.kind() == QubitStatusKind::Lost)) {
                classified_level = pre.level_id();
            }

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
            if (classified_level.has_value()) {
                // The policy pre-scan admits only single-qubit Z-basis
                // measurement forms here (M and the measure-and-resets);
                // X/Y-basis and multi-target measurements on a
                // noncomputational operand reject above.
                const uint32_t qubit = qubit_operands(node).front().qubit;
                const MeasurementClassifier& classifier =
                    classifier_for(model, *classified_level, gate, op_index, qubit);
                const bool ternary = classifier.num_symbols() == 3;

                // The record bit's flip probability on top of MPAD 0:
                // P(symbol 1 | level) for a two-symbol column; for a ternary
                // column, the bit's not-heralded conditional -- the herald
                // pass re-points heralded slots at one half. An always-herald
                // column has no not-heralded conditional; one half stands in
                // (every draw heralds, so the pass always overwrites it).
                double flip;
                if (ternary) {
                    const double p_herald = classifier.prob(2, *classified_level);
                    const double denom = 1.0 - p_herald;
                    flip = denom > 0.0 ? classifier.prob(1, *classified_level) / denom : 0.5;
                    flip = std::min(1.0, std::max(0.0, flip));
                } else {
                    flip = classifier.prob(1, *classified_level);
                }

                size_t noise_node = SIZE_MAX;
                if (!ternary && (flip == 0.0 || flip == 1.0)) {
                    // Deterministic column: the bit is the padding literal
                    // itself, no sample-time draw.
                    out.nodes.push_back(mpad_op(flip == 1.0 ? 1 : 0));
                } else {
                    out.nodes.push_back(mpad_op(0));
                    out.nodes.push_back(readout_noise_op(slot, flip));
                    noise_node = out.nodes.size() - 1;
                }
                if (is_measure_reset(gate)) {
                    out.nodes.push_back(single_qubit_op(reset_for(gate), qubit));
                }
                result.classified_measurements.push_back({slot, *classified_level, noise_node});
            } else {
                out.nodes.push_back(node);
            }
        }
        for (const CarrierEdit& edit : carrier_edits) {
            out.nodes.push_back(single_qubit_op(GateType::R, edit.qubit));
            if (edit.prepare_one) {
                out.nodes.push_back(single_qubit_op(GateType::X, edit.qubit));
            }
        }
        if (is_measurement(gate)) {
            ++slot;
        }
    }

    if (trans_cursor != history.transitions.size()) {
        throw std::invalid_argument(
            "rewrite: history records more transitions than the circuit consults; "
            "history does not describe this circuit");
    }

    return result;
}

}  // namespace clifft
