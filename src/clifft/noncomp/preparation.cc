#include "clifft/noncomp/preparation.h"

#include "clifft/noncomp/interaction.h"
#include "clifft/noncomp/status_walk.h"
#include "clifft/noncomp/trajectory_driver.h"
#include "clifft/noncomp/transition_hooks.h"
#include "clifft/util/shot_seed.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace clifft {

namespace {

void validate_annotation(const AstNode& node, const NonComputationalModel& model, uint32_t op_index,
                         uint32_t num_qubits) {
    if (is_noncomputational_interaction(node.gate)) {
        (void)validate_interaction(node, op_index, num_qubits, "sample_noncomputational");
        return;
    }
    if (is_noncomputational_herald(node.gate)) {
        validate_herald_arguments(node, op_index, "sample_noncomputational");
    } else if (is_inline_noncomputational_annotation(node.gate)) {
        (void)inline_transition_probability(node.gate, node.args, op_index,
                                            "sample_noncomputational");
    } else if (node.gate != GateType::LEVEL_TRANSITION) {
        throw std::invalid_argument("sample_noncomputational: unsupported annotation '" +
                                    std::string(gate_name(node.gate)) + "' at op " +
                                    std::to_string(op_index));
    } else if (model.transition_named(node.tag) == nullptr) {
        throw std::invalid_argument("sample_noncomputational: LEVEL_TRANSITION[" + node.tag +
                                    "] at op " + std::to_string(op_index) +
                                    " does not name a transition in the model");
    }
    if (node.targets.empty()) {
        throw std::invalid_argument(
            "sample_noncomputational: annotation '" + std::string(gate_name(node.gate)) +
            "' at op " + std::to_string(op_index) + " requires at least one qubit target");
    }
    std::unordered_set<uint32_t> seen_qubits;
    for (const Target& target : node.targets) {
        if (target.is_rec() || target.has_pauli() || target.is_inverted()) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) + "' at op " +
                                        std::to_string(op_index) + " requires plain qubit targets");
        }
        if (target.value() >= num_qubits) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) + "' target qubit " +
                                        std::to_string(target.value()) + " is out of range at op " +
                                        std::to_string(op_index));
        }
        if (!seen_qubits.insert(target.value()).second) {
            throw std::invalid_argument("sample_noncomputational: annotation '" +
                                        std::string(gate_name(node.gate)) +
                                        "' repeats target qubit " + std::to_string(target.value()) +
                                        " at op " + std::to_string(op_index));
        }
    }
}

// Validate the circuit features that require model-wide knowledge. If the
// model can ever leak or lose a qubit, parity measurements are unsupported and
// ordinary measurements require a classifier.
void validate_model_contract(const Circuit& annotated, const NonComputationalModel& model) {
    // Interactions only spread existing leakage, so they cannot make an
    // otherwise computational request noncomputational on their own.
    bool noncomp_capable = false;
    for (const Level l : kAllLevels) {
        if (!is_computational(l) && model.initial_probability(l) > 0.0) {
            noncomp_capable = true;
            break;
        }
    }
    if (!noncomp_capable) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(annotated.nodes.size()); ++i) {
            const AstNode& node = annotated.nodes[i];
            if (is_inline_noncomputational_annotation(node.gate) && !node.args.empty() &&
                node.args[0] > 0.0) {
                noncomp_capable = true;
                break;
            }
            if (node.gate == GateType::LEVEL_TRANSITION) {
                const TransitionInstrument* instr = model.transition_named(node.tag);
                assert(instr != nullptr && "annotation validation resolves transition names");
                for (const Level src : kAllLevels) {
                    if (!is_computational(src)) {
                        continue;
                    }
                    for (const Level dst : kAllLevels) {
                        if (!is_computational(dst) && instr->prob(dst, src) > 0.0) {
                            noncomp_capable = true;
                            break;
                        }
                    }
                    if (noncomp_capable) {
                        break;
                    }
                }
            }
            if (noncomp_capable) {
                break;
            }
        }
    }

    if (!noncomp_capable) {
        return;
    }

    // A parity measurement has no defined result once an operand leaves the
    // computational subspace.
    for (uint32_t i = 0; i < static_cast<uint32_t>(annotated.nodes.size()); ++i) {
        const AstNode& node = annotated.nodes[i];
        // MPP has MULTI arity; MXX/MYY/MZZ desugar to MPP at parse time.
        if (node.gate == GateType::MPP) {
            throw std::invalid_argument(
                "sample_noncomputational: parity measurement 'MPP' at op " + std::to_string(i) +
                " is not supported under a model that can leak or lose qubits;"
                " expand the parity readout into an explicit ancilla circuit");
        }
    }

    // A physical-qubit measurement needs a classifier for any leaked or lost
    // operand. MPAD and status probes only append classical record bits.
    if (model.classifier() == nullptr) {
        for (const AstNode& node : annotated.nodes) {
            if (is_physical_measurement(node.gate)) {
                throw std::invalid_argument(
                    "sample_noncomputational: this model can leak or lose qubits and the circuit"
                    " measures; a classifier is required to define what a measurement of a leaked"
                    " or lost qubit reads");
            }
        }
    }
}

Circuit prepare_circuit(const Circuit& circuit, const NonComputationalModel& model) {
    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }
    Circuit annotated = expand_transition_hooks(circuit, model);
    for (uint32_t op_index = 0; op_index < static_cast<uint32_t>(annotated.nodes.size());
         ++op_index) {
        const AstNode& node = annotated.nodes[op_index];
        if (is_noncomputational_annotation(node.gate)) {
            validate_annotation(node, model, op_index, annotated.num_qubits);
        }
        if (is_measurement(node.gate) && node.gate != GateType::MPP && node.targets.size() != 1) {
            throw std::invalid_argument(
                "sample_noncomputational: measurement '" + std::string(gate_name(node.gate)) +
                "' at op " + std::to_string(op_index) + " carries " +
                std::to_string(node.targets.size()) + " targets; the parser" +
                " emits one measurement node per target and the rewrite's" +
                " record accounting relies on that shape; split the node" +
                " into single-target measurements");
        }
    }
    validate_model_contract(annotated, model);
    return annotated;
}

}  // namespace

PreparedNonComputationalRequest::PreparedNonComputationalRequest(
    const Circuit& circuit, const NonComputationalModel& model,
    NonComputationalRequirements requirements)
    : model_(model),
      annotated_circuit_(prepare_circuit(circuit, model_)),
      requirements_(requirements) {}

NonComputationalSample PreparedNonComputationalRequest::sample(uint32_t shots,
                                                               std::optional<uint64_t> seed,
                                                               uint32_t threads) const {
    return run_trajectory_driver(*this, shots, make_seed_root(shots, seed), threads);
}

}  // namespace clifft
