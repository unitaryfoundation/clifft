#include "clifft/noncomp/rewriter.h"

#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/status_walk.h"
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
                           std::vector<ClassifiedMeasurement>& classified_measurements) {
    const NonComputationalPolicy& policy = model.policy();
    const GateType gate = node.gate;

    const OrdinaryStep step = advance_ordinary_node(node, op_index, status, policy, "rewrite");
    bool drop_op = step.dropped;
    const std::optional<ClassifiedOperand>& classified = step.classified_measurement;

    if (!drop_op) {
        if (classified.has_value()) {
            // The policy pre-scan admits single-qubit measurement forms
            // M, MX, MY, and the measure-and-resets; multi-qubit parity
            // measurements (MPP) on a noncomputational operand reject above.
            const uint32_t qubit = classified->qubit;
            const Level classified_level = classified->level;
            const MeasurementClassifier& classifier = classifier_for(model, gate, op_index, qubit);
            const bool ternary = classifier.has_herald();
            const bool inverted = node.targets.front().is_inverted();

            // MPAD starts the visible record at 0, then READOUT_NOISE flips it
            // with probability `flip`. Binary classifiers use P(symbol 1 |
            // level). Ternary classifiers use P(symbol 1 | level, no herald),
            // because heralded slots are later patched to an unbiased bit.
            // Target inversion complements only this visible bit; the herald
            // flag still means the classifier's third symbol. If every draw
            // heralds, the placeholder flip is irrelevant and 0.5 is used.
            double flip;
            if (ternary) {
                const double p_herald =
                    classifier.prob(MeasurementClassifier::kHeraldSymbol, classified_level);
                const double denom = 1.0 - p_herald;
                flip = denom > 0.0 ? classifier.prob(1, classified_level) / denom : 0.5;
                flip = std::min(1.0, std::max(0.0, flip));
            } else {
                flip = classifier.prob(1, classified_level);
            }
            if (inverted) {
                flip = 1.0 - flip;
            }

            std::optional<size_t> noise_node;
            if (!ternary && (flip == 0.0 || flip == 1.0)) {
                // Deterministic column: the bit is the padding literal
                // itself, no sample-time draw.
                out.nodes.push_back(mpad_op(flip == 1.0 ? 1 : 0));
            } else {
                out.nodes.push_back(mpad_op(0));
                out.nodes.push_back(readout_noise_op(slot, flip));
                noise_node = out.nodes.size() - 1;
            }
            // The reset half runs only when the stepper restored the site;
            // a non-restoring lost qubit keeps its vacated carrier.
            if (is_measure_reset(gate) && is_computational(status[qubit])) {
                out.nodes.push_back(single_qubit_op(reset_for(gate), qubit));
            }
            classified_measurements.push_back({slot, classified_level, noise_node});
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
    std::map<AnnotationTarget, Level> jump_dest;
    for (const ResolvedJump& jump : events.jumps) {
        if (!jump_dest.emplace(jump.target, jump.destination_level).second) {
            throw std::invalid_argument("rewrite_continuation: duplicate jump for op " +
                                        std::to_string(jump.target.op_index) + ", qubit " +
                                        std::to_string(jump.target.qubit));
        }
    }
    size_t jumps_seen = 0;
    size_t classical_cursor = 0;

    ContinuationRewrite result;
    Circuit& out = result.circuit;
    out = annotated.metadata_only_copy();
    out.nodes.reserve(annotated.nodes.size() + events.jumps.size() * 2);

    // No initial X-prep here: in exact mode a known |1> initial level is a
    // per-shot Pauli-frame preload, so every module in a chain compiles
    // from the same node stream regardless of the shot's initials.
    std::vector<QubitStatus> status = events.initial_status;

    // Node index (into `out`) of the last jump's trace-out R, when one is
    // emitted; converted to a hidden record slot after the walk.
    std::optional<size_t> traceout_node;

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
                    const AnnotationTarget target{op_index, qubit};
                    if (outcome.target != target) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome (op " +
                            std::to_string(outcome.target.op_index) + ", qubit " +
                            std::to_string(outcome.target.qubit) + ") does not match consult (op " +
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
                    if (is_computational(dest)) {
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

                // Quantum source: if the channel can never fire from any
                // computational level, the site is the identity on this qubit
                // and trace() will elide it. The site table and trace()'s
                // materialization must elide identically, so skip both the
                // node emission and the site_targets entry.
                // LOSS(p): fires iff p != 0. LEVEL_TRANSITION[tag]: fires iff
                // column_sum(G) != 0 or column_sum(E) != 0. These predicates
                // are exact 0.0 comparisons, matching frontend.cc.
                if (gate == GateType::LOSS) {
                    if (loss_probability(node.args, op_index, "rewrite_continuation") == 0.0) {
                        continue;
                    }
                } else {
                    const TransitionInstrument* instr = model.transition_named(node.tag);
                    if (instr == nullptr) {
                        throw std::invalid_argument(
                            "rewrite_continuation: unknown transition tag '" + node.tag +
                            "' at op " + std::to_string(op_index));
                    }
                    if (instr->column_sum(Level::G) == 0.0 && instr->column_sum(Level::E) == 0.0) {
                        continue;
                    }
                }

                // The annotation stays a runtime instrument.
                // Split multi-target nodes so a sibling target with a
                // classical status is not re-materialized.
                const AnnotationTarget site_target{op_index, qubit};
                result.site_targets.push_back(site_target);
                out.nodes.push_back(
                    AstNode{gate, {Target::qubit(qubit)}, node.args, node.source_line, node.tag});

                const auto jump = jump_dest.find(site_target);
                if (jump == jump_dest.end()) {
                    continue;  // no fire recorded here; the site runs live
                }
                ++jumps_seen;
                const bool is_last =
                    jumps_seen == events.jumps.size() && site_target == events.jumps.back().target;

                // Every jump resets its carrier at the site. For a
                // noncomputational destination the R is the trace-out
                // unraveling (a hidden measurement plus corrective Pauli
                // under reset lowering -- deterministic here, because the
                // site collapsed the carrier before trapping); for a
                // computational destination it re-prepares the carrier at
                // the destination level, with an X appended for |1>. A
                // forced trap-form trace-out points at the same reset.
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

    result.forced_traceout_node = traceout_node;

    result.final_status = std::move(status);
    return result;
}

}  // namespace clifft
