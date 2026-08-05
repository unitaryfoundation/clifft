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

// Returns the classifier used for a measurement on a leaked or lost qubit.
// The classifier itself was validated at construction. This function only
// checks that it exists and includes the operation location in the error.
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

// Applies classifier readout error to a kept computational Z-basis
// measurement. The computational columns give p01, the probability of
// reporting 1 for a true 0, and p10, the probability of reporting 0 for a
// true 1. Only the record bit changes; the qubit still collapses to its true
// state. Identity columns add no node. An inverted target reports the
// complement bit, so its two error probabilities are swapped.
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

// Process one ordinary circuit operation: apply the policy, replace
// measurements on leaked or lost qubits, add computational readout error,
// update qubit statuses, and advance the measurement record index.
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
            // Only single-qubit measurements reach this branch. MPP on a
            // leaked or lost operand is rejected by the policy above.
            const uint32_t qubit = classified->qubit;
            const Level classified_level = classified->level;
            const MeasurementClassifier& classifier = classifier_for(model, gate, op_index, qubit);
            const bool ternary = classifier.has_herald();
            const bool inverted = node.targets.front().is_inverted();

            // MPAD writes 0 into the original record slot, then
            // READOUT_NOISE changes it with probability `flip`. A binary
            // classifier uses P(symbol 1 | level). A three-symbol classifier
            // uses P(symbol 1 | level, not heralded), because the driver later
            // replaces a heralded slot with a uniformly random bit. Target
            // inversion complements the record bit but not the herald flag. If
            // the classifier always heralds, the temporary bit is irrelevant,
            // so use 0.5.
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
            // Keep the reset half only when the policy restored the qubit.
            // Otherwise a lost site remains empty.
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

ContinuationRewrite rewrite_continuation(const Circuit& annotated, const TrajectoryEvents& events,
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

    // Index recorded jumps by their annotation targets. The caller supplies
    // them in circuit order; the traversal below verifies that each one
    // matches a transition site.
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

    // Do not emit an X for an initial |1>. The driver supplies it through the
    // per-shot Pauli frame, keeping the circuit nodes identical across shots
    // and continuations.
    std::vector<QubitStatus> status = events.initial_status;

    // Index in `out` of the reset inserted for the last jump. If forcing is
    // requested, trace() later reports the hidden measurement slot created
    // when this reset is lowered.
    std::optional<size_t> traceout_node;

    uint32_t slot = 0;  // visible measurement record index

    for (uint32_t op_index = 0; op_index < annotated.nodes.size(); ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        const GateType gate = node.gate;

        if (is_noncomputational_annotation(gate)) {
            for (const Target& target : node.targets) {
                const uint32_t qubit = target.value();
                if (qubit >= status.size()) {
                    throw std::invalid_argument("rewrite_continuation: operand qubit " +
                                                std::to_string(qubit) + " is out of range at op " +
                                                std::to_string(op_index));
                }
                const QubitStatus pre = status[qubit];

                if (!is_computational(pre)) {
                    // The qubit is leaked or lost, so the VM has no
                    // transition to execute. Remove the annotation and apply
                    // the outcome already drawn by the driver.
                    if (classical_cursor >= events.classical_outcomes.size()) {
                        throw std::invalid_argument(
                            "rewrite_continuation: circuit has more transitions on leaked or "
                            "lost qubits than the events record (op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) + ")");
                    }
                    const ClassicalOutcome& outcome = events.classical_outcomes[classical_cursor++];
                    const AnnotationTarget target{op_index, qubit};
                    if (outcome.target != target) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome (op " +
                            std::to_string(outcome.target.op_index) + ", qubit " +
                            std::to_string(outcome.target.qubit) +
                            ") does not match transition (op " + std::to_string(op_index) +
                            ", qubit " + std::to_string(qubit) + ")");
                    }
                    if (outcome.source_level != noncomp_level(pre)) {
                        throw std::invalid_argument(
                            "rewrite_continuation: classical outcome at op " +
                            std::to_string(op_index) + ", qubit " + std::to_string(qubit) +
                            " was drawn at level '" + level_name(outcome.source_level) +
                            "' but the qubit has level '" + level_name(noncomp_level(pre)) + "'");
                    }
                    if (!outcome.destination.has_value()) {
                        continue;
                    }
                    const Level dest = *outcome.destination;
                    if (is_computational(dest)) {
                        // A transition back to g or e recreates the qubit in
                        // that definite computational state.
                        out.nodes.push_back(single_qubit_op(GateType::R, qubit));
                        if (dest == Level::E) {
                            out.nodes.push_back(single_qubit_op(GateType::X, qubit));
                        }
                    }
                    status[qubit] = status_for(dest);
                    continue;
                }

                // The qubit is computational. Omit a transition whose firing
                // probability is exactly zero for both g and e. trace() omits
                // the same sites, so skipping both the node and site_targets
                // entry keeps their site IDs aligned.
                //
                // Inline LEAKAGE/LOSS can fire when p != 0.
                // LEVEL_TRANSITION[tag] can fire when either column_sum(G) or
                // column_sum(E) is nonzero. These exact 0.0 comparisons match
                // frontend.cc.
                if (is_inline_noncomputational_annotation(gate)) {
                    if (inline_transition_probability(gate, node.args, op_index,
                                                      "rewrite_continuation") == 0.0) {
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

                // Keep one runtime instrument for this computational target.
                // Splitting a multi-target annotation prevents a leaked or
                // lost sibling target from being emitted again.
                const AnnotationTarget site_target{op_index, qubit};
                result.site_targets.push_back(site_target);
                out.nodes.push_back(
                    AstNode{gate, {Target::qubit(qubit)}, node.args, node.source_line, node.tag});

                const auto jump = jump_dest.find(site_target);
                if (jump == jump_dest.end()) {
                    continue;  // no recorded jump; the VM executes this transition
                }
                ++jumps_seen;
                const bool is_last =
                    jumps_seen == events.jumps.size() && site_target == events.jumps.back().target;

                // Insert a reset after every recorded jump. For a leaked or
                // lost destination, reset lowering traces the qubit out of the
                // quantum state. For destination g or e, the reset prepares
                // |0>, followed by X for |1>. When the last jump came from a
                // trap, remember this reset so the driver can force its hidden
                // measurement to the source level reported by the VM.
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
            "rewrite_continuation: events record jumps that do not match transition "
            "annotations on computational qubits");
    }
    if (classical_cursor != events.classical_outcomes.size()) {
        throw std::invalid_argument(
            "rewrite_continuation: events record more leaked or lost transition outcomes than "
            "the circuit uses");
    }

    result.forced_traceout_node = traceout_node;

    result.final_status = std::move(status);
    return result;
}

}  // namespace clifft
