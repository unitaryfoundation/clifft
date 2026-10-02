#include "clifft/noncomp/interaction.h"

#include "clifft/util/numeric.h"

#include <cassert>
#include <stdexcept>
#include <string>

namespace clifft {

InteractionSource parse_interaction_source(std::string_view name) {
    if (name == "leaked") {
        return InteractionSource::Leaked;
    }
    if (name == "lost") {
        return InteractionSource::Lost;
    }
    throw std::invalid_argument("interaction source status '" + std::string(name) +
                                "' must be 'leaked' or 'lost'");
}

bool supports_partner_effect(GateType gate) {
    return is_unitary(gate) && gate_arity(gate) == GateArity::PAIR && !is_identity_noop(gate) &&
           !is_parser_desugared(gate);
}

void PartnerEffect::validate(bool allow_spreading, std::string_view context) const {
    for (double p : pauli) {
        if (!is_probability(p)) {
            throw std::invalid_argument(std::string(context) +
                                        ": Pauli probabilities must be finite and in [0, 1]");
        }
    }
    if (pauli[0] + pauli[1] + pauli[2] > 1.0) {
        throw std::invalid_argument(std::string(context) +
                                    ": Pauli probabilities must sum to at most 1");
    }
    if (!is_probability(spread_probability)) {
        throw std::invalid_argument(std::string(context) +
                                    ": spread_probability must be finite and in [0, 1]");
    }
    if (!allow_spreading && spread_probability != 0.0) {
        throw std::invalid_argument(std::string(context) +
                                    ": spreading is only supported for a leaked source");
    }
}

bool PartnerEffect::has_pauli() const {
    return pauli[0] != 0.0 || pauli[1] != 0.0 || pauli[2] != 0.0;
}

bool PartnerEffect::empty() const {
    return !has_pauli() && spread_probability == 0.0;
}

PartnerEffect interaction_arguments(GateType gate, std::span<const double> args) {
    const bool leakage = gate == GateType::LEAKAGE_INTERACTION;
    if (!is_noncomputational_interaction(gate)) {
        throw std::invalid_argument("expected a leakage or loss interaction annotation");
    }
    const std::string name(gate_name(gate));
    if (args.size() != (leakage ? 4 : 3)) {
        throw std::invalid_argument(name +
                                    (leakage ? " requires 4 arguments" : " requires 3 arguments"));
    }
    PartnerEffect effect{{args[0], args[1], args[2]}, leakage ? args[3] : 0.0};
    effect.validate(leakage, name);
    return effect;
}

PartnerEffect validate_interaction(const AstNode& node, uint32_t op_index, size_t num_qubits,
                                   std::string_view caller) {
    const std::string context = std::string(caller) + ": " + std::string(gate_name(node.gate)) +
                                " at op " + std::to_string(op_index);
    try {
        const PartnerEffect effect = interaction_arguments(node.gate, node.args);
        if (!node.tag.empty() || node.targets.size() != 2) {
            throw std::invalid_argument("requires an untagged pair of source and partner targets");
        }
        for (const auto& target : node.targets) {
            if (target.is_rec() || target.has_pauli() || target.is_inverted() ||
                target.value() >= num_qubits) {
                throw std::invalid_argument("requires plain qubit targets within the circuit");
            }
        }
        if (node.targets[0].value() == node.targets[1].value()) {
            throw std::invalid_argument("requires distinct source and partner targets");
        }
        return effect;
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(context + ": " + e.what());
    }
}

bool interaction_applies(const AstNode& node, std::span<const QubitStatus> status) {
    assert(node.targets.size() == 2);
    const auto source = node.targets[0].value();
    const auto partner = node.targets[1].value();
    assert(source < status.size() && partner < status.size());
    const bool matches = node.gate == GateType::LEAKAGE_INTERACTION ? is_leaked(status[source])
                                                                    : is_lost(status[source]);
    return matches && is_computational(status[partner]);
}

}  // namespace clifft
