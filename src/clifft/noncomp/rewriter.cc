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

// Classifier readout confusion for a kept computational Z-basis measurement.
// The classifier's computational columns give the probability of misreporting
// each true outcome, so the record bit gets an asymmetric flip conditioned on
// its value: p01 = P(symbol 1 | zero level), p10 = P(symbol 0 | one level).
// Real readout error is a misreport -- the qubit still collapses to its true
// state -- so only the record slot is touched and the layout is unchanged.
// Identity columns emit nothing. An inverted measurement target reports the
// complement bit, and confusion physically precedes the reporting convention,
// so the emitted probabilities are swapped for it.
void append_computational_confusion(Circuit& out, const NonComputationalModel& model, GateType gate,
                                    bool inverted, uint32_t slot, uint32_t op_index,
                                    uint32_t qubit) {
    const MeasurementClassifier* classifier = model.classifier();
    if (classifier == nullptr) {
        return;
    }
    const LevelSet& levels = model.levels();
    const uint8_t zero_id = levels.computational_zero_id();
    const uint8_t one_id = levels.computational_one_id();
    for (uint8_t level : {zero_id, one_id}) {
        const double reject = classifier->reject_probability(level);
        if (reject > kProbTolerance) {
            throw std::invalid_argument(
                "rewrite: classifier reject columns are not supported yet; the column for "
                "computational level " +
                std::to_string(level) + " sums to less than one (reject probability " +
                std::to_string(reject) + ", measurement '" + std::string(gate_name(gate)) +
                "' on qubit " + std::to_string(qubit) + " at op " + std::to_string(op_index) + ")");
        }
        double beyond_bit = 0.0;
        for (size_t symbol = 2; symbol < classifier->num_symbols(); ++symbol) {
            beyond_bit += classifier->prob(static_cast<uint8_t>(symbol), level);
        }
        if (beyond_bit > kProbTolerance) {
            throw std::invalid_argument(
                "rewrite: a computational measurement's classifier column must place all its "
                "probability on the record symbols 0 and 1; the column for level " +
                std::to_string(level) + " puts " + std::to_string(beyond_bit) +
                " beyond the bit (measurement '" + std::string(gate_name(gate)) + "' on qubit " +
                std::to_string(qubit) + " at op " + std::to_string(op_index) + ")");
        }
    }
    double p01 = classifier->prob(1, zero_id);  // true 0 misread as 1
    double p10 = classifier->prob(0, one_id);   // true 1 misread as 0
    if (inverted) {
        std::swap(p01, p10);
    }
    if (p01 == 0.0 && p10 == 0.0) {
        return;  // identity columns: no draw, no node
    }
    out.nodes.push_back(AstNode{GateType::READOUT_NOISE, {Target::rec(slot)}, {p01, p10}, 0});
}

// Shared per-node processing for every non-annotation operation: the
// policy scan (drop/reject), classifier record writes for measurements on
// leaked/lost qubits, computational readout confusion for kept Z-basis
// measurements, status stepping, and the visible-slot cursor. Both the
// AOT rewrite and the exact-mode continuation rewrite go through this, so
// the two paths cannot drift.
void process_ordinary_node(const AstNode& node, uint32_t op_index,
                           const NonComputationalModel& model, std::vector<QubitStatus>& status,
                           Circuit& out, uint32_t& slot,
                           std::vector<ClassifiedMeasurement>& classified) {
    const LevelSet& levels = model.levels();
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

        status[qubit] =
            drop_op ? pre : normal_post_op_status(pre, gate, operand.role, policy, levels);
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
                append_computational_confusion(out, model, gate, node.targets.front().is_inverted(),
                                               slot, op_index, qubit_operands(node).front().qubit);
            }
        }
    }
    if (is_measurement(gate)) {
        ++slot;
    }
}

}  // namespace

ContinuationRewrite rewrite_continuation(const Circuit& annotated, const ExactShotEvents& events,
                                         bool force_last_traceout,
                                         const NonComputationalModel& model) {
    const LevelSet& levels = model.levels();

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
    std::map<std::pair<uint32_t, uint32_t>, uint8_t> jump_dest;
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
                const bool classical_source =
                    pre.kind() == QubitStatusKind::Leaked || pre.kind() == QubitStatusKind::Lost;

                if (classical_source) {
                    // Classical-source consult: no runtime instrument; the
                    // host pre-drew the outcome. The annotation node is
                    // consumed, exactly as in the AOT rewrite.
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
                    if (outcome.source_level != pre.level_id()) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome at op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                            " was drawn at level " + std::to_string(outcome.source_level) +
                            " but the walk holds level " + std::to_string(pre.level_id()));
                    }
                    if (!outcome.jumped) {
                        continue;
                    }
                    const Level& dest = levels.at(outcome.destination_level);
                    if (dest.category == LevelCategory::Computational) {
                        // Recapture: materialize the carrier at the definite
                        // destination level.
                        out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                        if (outcome.destination_level == levels.computational_one_id()) {
                            out.nodes.push_back(single_qubit_op(GateType::X, qubit));
                        }
                    }
                    status[qubit] = levels.status_for(outcome.destination_level);
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

                const Level& dest = levels.at(jump->second);
                size_t r_node = SIZE_MAX;
                if (dest.category == LevelCategory::Computational) {
                    r_node = out.nodes.size();
                    out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                    if (jump->second == levels.computational_one_id()) {
                        out.nodes.push_back(single_qubit_op(GateType::X, qubit));
                    }
                } else if (pre.kind() == QubitStatusKind::ComputationalUnknown) {
                    r_node = out.nodes.size();
                    out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                }
                if (is_last && force_last_traceout) {
                    if (r_node == SIZE_MAX) {
                        throw std::invalid_argument(
                            "rewrite_continuation: force_last_traceout on a jump that emits no "
                            "carrier reset (the qubit's level is already definite at op " +
                            std::to_string(op_index) + ")");
                    }
                    traceout_node = r_node;
                }
                status[qubit] = levels.status_for(jump->second);
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
        // The forced trace-out's hidden record slot. Hidden slots are
        // assigned by trace() sequentially in circuit order, one per pure
        // reset target, starting after the visible slots; mirror that count
        // over the rewritten node stream.
        uint32_t hidden_before = 0;
        for (size_t i = 0; i < traceout_node; ++i) {
            if (is_reset(out.nodes[i].gate)) {
                hidden_before += static_cast<uint32_t>(out.nodes[i].targets.size());
            }
        }
        result.forced_traceout_slot = annotated.num_measurements + hidden_before;
    }

    result.final_status = std::move(status);
    return result;
}

}  // namespace clifft
