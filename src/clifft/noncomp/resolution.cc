#include "clifft/noncomp/resolution.h"

#include "clifft/noncomp/status_walk.h"

#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace clifft {
namespace {

void validate_annotation_targets(const AstNode& node, uint32_t op_index, uint32_t num_qubits,
                                 std::string_view caller) {
    if (node.targets.empty()) {
        throw std::invalid_argument(
            std::string(caller) + ": annotation '" + std::string(gate_name(node.gate)) +
            "' at op " + std::to_string(op_index) + " requires at least one qubit target");
    }
    std::unordered_set<uint32_t> seen_qubits;
    for (const Target& target : node.targets) {
        if (target.is_rec() || target.has_pauli() || target.is_inverted()) {
            throw std::invalid_argument(std::string(caller) + ": annotation '" +
                                        std::string(gate_name(node.gate)) + "' at op " +
                                        std::to_string(op_index) + " requires plain qubit targets");
        }
        if (target.value() >= num_qubits) {
            throw std::invalid_argument(std::string(caller) + ": annotation '" +
                                        std::string(gate_name(node.gate)) + "' target qubit " +
                                        std::to_string(target.value()) + " is out of range at op " +
                                        std::to_string(op_index));
        }
        if (!seen_qubits.insert(target.value()).second) {
            throw std::invalid_argument(std::string(caller) + ": annotation '" +
                                        std::string(gate_name(node.gate)) +
                                        "' repeats target qubit " + std::to_string(target.value()) +
                                        " at op " + std::to_string(op_index));
        }
    }
}

void validate_measurement_shape(const AstNode& node, uint32_t op_index, std::string_view caller) {
    if (is_measurement(node.gate) && node.gate != GateType::MPP && node.targets.size() != 1) {
        throw std::invalid_argument(std::string(caller) + ": measurement '" +
                                    std::string(gate_name(node.gate)) + "' at op " +
                                    std::to_string(op_index) + " carries " +
                                    std::to_string(node.targets.size()) + " targets; the parser" +
                                    " emits one measurement node per target and the rewrite's" +
                                    " record accounting relies on that shape; split the node" +
                                    " into single-target measurements");
    }
}

}  // namespace

bool ResolvedInteraction::applies(std::span<const QubitStatus> status) const {
    assert(source < status.size() && partner < status.size());
    const bool matches = source_status == InteractionSource::Leaked ? is_leaked(status[source])
                                                                    : is_lost(status[source]);
    return matches && is_computational(status[partner]);
}

ResolvedNonComputationalCircuit::ResolvedNonComputationalCircuit(
    Circuit annotated, const NonComputationalModel& model, std::string_view caller,
    std::vector<std::optional<uint32_t>> generated_by)
    : circuit_(std::move(annotated)) {
    assert(generated_by.empty() || generated_by.size() == circuit_.nodes.size());
    operations_.reserve(circuit_.nodes.size());
    std::map<std::string, TransitionId, std::less<>> instrument_ids;
    for (uint32_t op_index = 0; op_index < circuit_.nodes.size(); ++op_index) {
        const AstNode& node = circuit_.nodes[op_index];
        ResolvedNonComputationalOperation op;
        if (!generated_by.empty()) {
            op.generated_by = generated_by[op_index];
            assert(!op.generated_by || *op.generated_by < op_index);
        }
        if (is_noncomputational_interaction(node.gate)) {
            const PartnerEffect effect =
                validate_interaction(node, op_index, circuit_.num_qubits, caller);
            op.annotation = ResolvedInteraction{node.targets[0].value(), node.targets[1].value(),
                                                node.gate == GateType::LEAKAGE_INTERACTION
                                                    ? InteractionSource::Leaked
                                                    : InteractionSource::Lost,
                                                effect};
        } else if (is_noncomputational_annotation(node.gate)) {
            if (is_noncomputational_herald(node.gate)) {
                validate_herald_arguments(node, op_index, caller);
            } else if (is_inline_noncomputational_annotation(node.gate)) {
                const double probability =
                    inline_transition_probability(node.gate, node.args, op_index, caller);
                op.annotation = node.gate == GateType::LEAKAGE
                                    ? ResolvedTransition{LeakageTransition{probability}}
                                    : ResolvedTransition{LossTransition{probability}};
            } else if (node.gate == GateType::LEVEL_TRANSITION) {
                auto it = instrument_ids.find(node.tag);
                if (it == instrument_ids.end()) {
                    const TransitionInstrument* instrument = model.transition_named(node.tag);
                    if (instrument == nullptr) {
                        throw std::invalid_argument(
                            std::string(caller) + ": LEVEL_TRANSITION[" + node.tag + "] at op " +
                            std::to_string(op_index) + " does not name a transition in the model");
                    }
                    const TransitionId id{static_cast<uint32_t>(instruments_.size())};
                    instruments_.push_back(*instrument);
                    it = instrument_ids.emplace(node.tag, id).first;
                }
                op.annotation = ResolvedTransition{it->second};
            } else {
                throw std::invalid_argument(std::string(caller) + ": unsupported annotation '" +
                                            std::string(gate_name(node.gate)) + "' at op " +
                                            std::to_string(op_index));
            }
            validate_annotation_targets(node, op_index, circuit_.num_qubits, caller);
        }
        validate_measurement_shape(node, op_index, caller);
        if (is_noncomputational_herald(node.gate)) {
            op.annotation = ResolvedStatusProbe{node.targets[0].value(),
                                                node.gate == GateType::HERALD_LEAKAGE_EVENT
                                                    ? InteractionSource::Leaked
                                                    : InteractionSource::Lost};
        }
        operations_.push_back(std::move(op));
    }
}

ResolvedTransition ResolvedNonComputationalCircuit::transition(uint32_t op_index) const {
    const auto& annotation = operation(op_index).annotation;
    if (const auto* channel = std::get_if<ResolvedTransition>(&annotation)) {
        return *channel;
    }
    const auto* interaction = std::get_if<ResolvedInteraction>(&annotation);
    assert(interaction != nullptr && interaction->source_status == InteractionSource::Leaked);
    return LeakageTransition{interaction->effect.spread_probability};
}

}  // namespace clifft
