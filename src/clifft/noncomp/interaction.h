#pragma once

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/level.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace clifft {

struct PartnerEffect {
    std::array<double, 3> pauli{};
    double spread_probability = 0.0;

    void validate(bool allow_spreading, std::string_view context) const;
    bool has_pauli() const;
    bool empty() const;
};

enum class InteractionSource { Leaked, Lost };

struct GatePartnerEffects {
    std::optional<PartnerEffect> leaked;
    std::optional<PartnerEffect> lost;
};

struct InteractionRule {
    std::string gate;
    uint32_t source_operand;
    InteractionSource source_status;
    PartnerEffect effect;
};

InteractionSource parse_interaction_source(std::string_view name);
bool supports_partner_effect(GateType gate);

PartnerEffect interaction_arguments(GateType gate, std::span<const double> args);

// Validate direct AST inputs as well as parsed instructions before consulting
// their ordered source and partner operands.
PartnerEffect validate_interaction(const AstNode& node, uint32_t op_index, size_t num_qubits,
                                   std::string_view caller);
bool interaction_applies(const AstNode& node, std::span<const QubitStatus> status);

}  // namespace clifft
