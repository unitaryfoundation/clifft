#include "clifft/noncomp/preparation.h"

#include "clifft/noncomp/trajectory_driver.h"
#include "clifft/noncomp/transition_hooks.h"
#include "clifft/util/shot_seed.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

// Validate the circuit features that require model-wide knowledge. If the
// model can ever leak or lose a qubit, parity measurements are unsupported and
// ordinary measurements require a classifier.
void validate_model_contract(const ResolvedNonComputationalCircuit& resolved,
                             const NonComputationalModel& model) {
    const Circuit& annotated = resolved.circuit();
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
            const auto* channel =
                std::get_if<ResolvedTransition>(&resolved.operation(i).annotation);
            if (channel == nullptr) {
                continue;
            }
            if (const auto* leakage = std::get_if<LeakageTransition>(channel)) {
                noncomp_capable = noncomp_capable || leakage->probability > 0.0;
            } else if (const auto* loss = std::get_if<LossTransition>(channel)) {
                noncomp_capable = noncomp_capable || loss->probability > 0.0;
            } else if (const auto* id = std::get_if<TransitionId>(channel)) {
                const TransitionInstrument& instr = resolved.instrument(*id);
                for (const Level src : kAllLevels) {
                    if (!is_computational(src)) {
                        continue;
                    }
                    for (const Level dst : kAllLevels) {
                        if (!is_computational(dst) && instr.prob(dst, src) > 0.0) {
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

ResolvedNonComputationalCircuit prepare_circuit(const Circuit& circuit,
                                                const NonComputationalModel& model) {
    if (circuit.num_exp_vals != 0) {
        throw std::invalid_argument(
            "sample_noncomputational: EXP_VAL probes are not supported in noncomputational "
            "sampling");
    }
    std::vector<std::optional<uint32_t>> generated_by;
    Circuit annotated = expand_transition_hooks(circuit, model, &generated_by);
    ResolvedNonComputationalCircuit resolved(std::move(annotated), model, "sample_noncomputational",
                                             std::move(generated_by));
    validate_model_contract(resolved, model);
    return resolved;
}

}  // namespace

PreparedNonComputationalRequest::PreparedNonComputationalRequest(
    const Circuit& circuit, const NonComputationalModel& model,
    NonComputationalRequirements requirements)
    : model_(model),
      resolved_circuit_(prepare_circuit(circuit, model_)),
      requirements_(requirements) {}

NonComputationalSample PreparedNonComputationalRequest::sample(uint32_t shots,
                                                               std::optional<uint64_t> seed,
                                                               uint32_t threads) const {
    return run_trajectory_driver(*this, shots, make_seed_root(shots, seed), threads);
}

}  // namespace clifft
