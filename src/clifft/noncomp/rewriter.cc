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
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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

// The model's classifier, which supplies the record bit for a measurement
// on a noncomputational qubit. Everything about the classifier's shape
// (symbol count, stochastic columns) is validated at its construction;
// only its presence is checked here, where the op context makes a good
// error message.
const MeasurementClassifier& classifier_for(const NonComputationalModel& model, GateType gate,
                                            uint32_t op_index, uint32_t qubit) {
    const MeasurementClassifier* classifier = model.classifier();
    if (classifier == nullptr) {
        throw std::invalid_argument("rewrite: measurement '" + std::string(gate_name(gate)) +
                                    "' on a noncomputational qubit " + std::to_string(qubit) +
                                    " at op " + std::to_string(op_index) +
                                    " requires a classifier, but the model has none");
    }
    return *classifier;
}

// Classifier readout confusion for a kept computational Z-basis measurement.
// The classifier's computational columns give the probability of misreporting
// each true outcome, so the record bit gets an asymmetric flip conditioned on
// its value: p01 = P(symbol 1 | zero level), p10 = P(symbol 0 | one level).
// Real readout error is a misreport -- the qubit still collapses to its true
// state -- so only the record slot is touched and the layout is unchanged.
// Identity columns emit nothing. An inverted measurement target reports the
// complement bit, and confusion physically precedes the reporting convention,
// so the emitted probabilities are swapped for it.
void append_computational_confusion(Circuit& out, const NonComputationalModel& model, bool inverted,
                                    uint32_t slot) {
    const MeasurementClassifier* classifier = model.classifier();
    if (classifier == nullptr) {
        return;
    }
    double p01 = classifier->prob(1, Level::G);  // true 0 misread as 1
    double p10 = classifier->prob(0, Level::E);  // true 1 misread as 0
    if (inverted) {
        std::swap(p01, p10);
    }
    if (p01 == 0.0 && p10 == 0.0) {
        return;  // identity columns: no draw, no node
    }
    out.nodes.push_back(AstNode{GateType::READOUT_NOISE, {Target::rec(slot)}, {p01, p10}, 0});
}

// Per-node processing for every non-annotation operation: the policy
// scan (drop/reject), classifier record writes for measurements on
// leaked/lost qubits, computational readout confusion for kept Z-basis
// measurements, status stepping, and the visible-slot cursor.
void process_ordinary_node(const AstNode& node, uint32_t op_index,
                           const NonComputationalModel& model, std::vector<QubitStatus>& status,
                           Circuit& out, uint32_t& slot,
                           std::vector<ClassifiedMeasurement>& classified) {
    const NonComputationalPolicy& policy = model.policy();
    const GateType gate = node.gate;

    // Policy pre-scan over entry statuses: any rejecting operand rejects
    // the whole operation; otherwise any dropping operand drops it whole
    // (identity on the surviving operands).
    bool drop_op = false;
    for (const QubitOperand& operand : qubit_operands(node)) {
        const uint32_t qubit = operand.qubit;
        if (qubit >= status.size()) {
            throw std::invalid_argument("rewrite: operand qubit " + std::to_string(qubit) +
                                        " is out of range at op " + std::to_string(op_index));
        }
        switch (operand_action(gate, status[qubit], policy)) {
            case OperandAction::Reject:
                throw std::invalid_argument(
                    "rewrite: operation '" + std::string(gate_name(gate)) + "' on a " +
                    status_name(status[qubit]) + " qubit " + std::to_string(qubit) + " at op " +
                    std::to_string(op_index) + " is not representable; rejecting");
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

    if (!drop_op) {
        if (classified_level.has_value()) {
            // The policy pre-scan admits only single-qubit Z-basis
            // measurement forms here (M and the measure-and-resets);
            // X/Y-basis and multi-target measurements on a
            // noncomputational operand reject above.
            const uint32_t qubit = qubit_operands(node).front().qubit;
            const MeasurementClassifier& classifier = classifier_for(model, gate, op_index, qubit);
            const bool ternary = classifier.has_herald();

            // The record bit's flip probability on top of MPAD 0:
            // P(symbol 1 | level) for a two-symbol column; for a ternary
            // column, the bit's not-heralded conditional -- the herald
            // pass re-points heralded slots at one half. An always-herald
            // column has no not-heralded conditional; one half stands in
            // (every draw heralds, so the pass always overwrites it).
            double flip;
            if (ternary) {
                const double p_herald =
                    classifier.prob(MeasurementClassifier::kHeraldSymbol, *classified_level);
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
            classified.push_back({slot, *classified_level, noise_node});
        } else {
            out.nodes.push_back(node);
            // The classifier's computational columns apply to Z-basis
            // level readouts (M, MR); other measurement bases are not
            // level readouts and carry no confusion.
            if (gate == GateType::M || gate == GateType::MR) {
                append_computational_confusion(out, model, node.targets.front().is_inverted(),
                                               slot);
            }
        }
    }
    if (is_measurement(gate)) {
        ++slot;
    }
}

// The hidden measurement-record slot trace() assigns to the reset at
// `reset_node` in the rewritten stream. This mirrors trace()'s hidden
// numbering (frontend.cc: hidden_meas_idx starts at the visible-measurement
// count and increments by one per pure-reset target, in circuit order).
// The two counts must agree -- the driver forces exactly this slot -- so
// the contract is stated in both places and cross-checked at runtime by
// swap_traceout_to_forced, which fails loudly if the slot names anything
// other than exactly one forced-capable measurement. Threading the slot out
// of trace() directly (so it is assigned in one place) is a worthwhile
// follow-up; it needs the HIR measure op to carry its source node.
size_t forced_reset_hidden_slot(const Circuit& rewritten, size_t reset_node,
                                uint32_t num_visible_measurements) {
    uint32_t hidden_before = 0;
    for (size_t i = 0; i < reset_node; ++i) {
        if (is_reset(rewritten.nodes[i].gate)) {
            hidden_before += static_cast<uint32_t>(rewritten.nodes[i].targets.size());
        }
    }
    return static_cast<size_t>(num_visible_measurements) + hidden_before;
}

}  // namespace

ContinuationRewrite rewrite_continuation(const Circuit& annotated, const ExactShotEvents& events,
                                         bool force_last_traceout,
                                         const NonComputationalModel& model) {
    if (events.initial_status.size() != annotated.num_qubits) {
        throw std::invalid_argument("rewrite_continuation: events carry " +
                                    std::to_string(events.initial_status.size()) +
                                    " initial statuses but circuit declares " +
                                    std::to_string(annotated.num_qubits) + " qubits");
    }
    if (force_last_traceout && events.jumps.empty()) {
        throw std::invalid_argument(
            "rewrite_continuation: force_last_traceout requires at least one jump");
    }

    // Jumps index by their annotation target. The chain arrives in trap
    // order, which is circuit order; visitation below validates coverage.
    std::map<std::pair<uint32_t, uint32_t>, Level> jump_dest;
    for (const ResolvedJump& jump : events.jumps) {
        if (!jump_dest.emplace(std::make_pair(jump.op_index, jump.qubit), jump.destination_level)
                 .second) {
            throw std::invalid_argument("rewrite_continuation: duplicate jump for op " +
                                        std::to_string(jump.op_index) + ", qubit " +
                                        std::to_string(jump.qubit));
        }
    }
    size_t jumps_seen = 0;
    size_t classical_cursor = 0;

    ContinuationRewrite result;
    Circuit& out = result.circuit;
    out = annotated;
    out.nodes.clear();
    out.nodes.reserve(annotated.nodes.size() + events.jumps.size() * 2);

    // No initial X-prep here: in exact mode a known |1> initial level is a
    // per-shot Pauli-frame preload, so every module in a chain compiles
    // from the same node stream regardless of the shot's initials.
    std::vector<QubitStatus> status = events.initial_status;

    // Node index (into `out`) of the last jump's trace-out R, when one is
    // emitted; converted to a hidden record slot after the walk.
    size_t traceout_node = SIZE_MAX;

    uint32_t slot = 0;  // visible measurement record index

    for (uint32_t op_index = 0; op_index < annotated.nodes.size(); ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        const GateType gate = node.gate;

        if (gate == GateType::LEVEL_TRANSITION || gate == GateType::LOSS) {
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                if (qubit >= status.size()) {
                    throw std::invalid_argument("rewrite_continuation: operand qubit " +
                                                std::to_string(qubit) + " is out of range at op " +
                                                std::to_string(op_index));
                }
                const QubitStatus pre = status[qubit];

                if (!is_computational(pre)) {
                    // Classical-source consult: no runtime instrument; the
                    // driver pre-drew the outcome and the annotation node
                    // is consumed.
                    if (classical_cursor >= events.classical_outcomes.size()) {
                        throw std::invalid_argument(
                            "rewrite_continuation: circuit consults more classical-source "
                            "transitions than the events record (op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) + ")");
                    }
                    const ClassicalOutcome& outcome = events.classical_outcomes[classical_cursor++];
                    if (outcome.op_index != op_index || outcome.qubit != qubit) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome (op " +
                            std::to_string(outcome.op_index) + ", qubit " +
                            std::to_string(outcome.qubit) + ") does not match consult (op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) + ")");
                    }
                    if (outcome.source_level != noncomp_level(pre)) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome at op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                            " was drawn at level '" + level_name(outcome.source_level) +
                            "' but the walk holds level '" + level_name(noncomp_level(pre)) + "'");
                    }
                    if (!outcome.destination.has_value()) {
                        continue;
                    }
                    const Level dest = *outcome.destination;
                    if (category(dest) == LevelCategory::Computational) {
                        // Recapture: materialize the carrier at the definite
                        // destination level.
                        out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                        if (dest == Level::E) {
                            out.nodes.push_back(single_qubit_op(GateType::X, qubit));
                        }
                    }
                    status[qubit] = status_for(dest);
                    continue;
                }

                // Quantum source: the annotation stays a runtime instrument.
                // Split multi-target nodes so a sibling target with a
                // classical status is not re-materialized.
                result.site_targets.emplace_back(op_index, qubit);
                out.nodes.push_back(
                    AstNode{gate, {Target::qubit(qubit)}, node.args, node.source_line, node.tag});

                const auto jump = jump_dest.find({op_index, qubit});
                if (jump == jump_dest.end()) {
                    continue;  // no fire recorded here; the site runs live
                }
                ++jumps_seen;
                const bool is_last = jumps_seen == events.jumps.size() &&
                                     op_index == events.jumps.back().op_index &&
                                     qubit == events.jumps.back().qubit;

                // Every jump resets its carrier at the site. For a
                // noncomputational destination the R is the trace-out
                // unraveling (a hidden measurement plus corrective Pauli
                // under reset lowering -- deterministic here, because the
                // site collapsed the carrier before trapping); for a
                // computational destination it re-prepares the carrier at
                // the destination level, with an X appended for |1>. A
                // forced neglect trace-out points at the same reset.
                const size_t r_node = out.nodes.size();
                out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                if (jump->second == Level::E) {
                    out.nodes.push_back(single_qubit_op(GateType::X, qubit));
                }
                if (is_last && force_last_traceout) {
                    traceout_node = r_node;
                }
                status[qubit] = status_for(jump->second);
            }
            continue;
        }

        process_ordinary_node(node, op_index, model, status, out, slot,
                              result.classified_measurements);
    }

    if (jumps_seen != events.jumps.size()) {
        throw std::invalid_argument(
            "rewrite_continuation: events record jumps the circuit never consults");
    }
    if (classical_cursor != events.classical_outcomes.size()) {
        throw std::invalid_argument(
            "rewrite_continuation: events record more classical outcomes than the circuit "
            "consults");
    }

    if (traceout_node != SIZE_MAX) {
        result.forced_traceout_slot =
            forced_reset_hidden_slot(out, traceout_node, annotated.num_measurements);
    }

    result.final_status = std::move(status);
    return result;
}

}  // namespace clifft
