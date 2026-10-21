#pragma once

// Resolves leakage and loss annotations once during preparation so trajectory
// sampling and continuation rewriting share their interpretation without
// reparsing tags or arguments. An owned expanded circuit and instrument table
// keep operation identities stable across rewrites and copies.

#include "clifft/circuit/circuit.h"
#include "clifft/noncomp/model.h"

#include <cassert>
#include <compare>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace clifft {

struct TransitionId {
    uint32_t value = 0;
    constexpr auto operator<=>(const TransitionId&) const = default;
};

struct LeakageTransition {
    double probability;
};

struct LossTransition {
    double probability;
};

using ResolvedTransition = std::variant<TransitionId, LeakageTransition, LossTransition>;

struct ResolvedInteraction {
    uint32_t source;
    uint32_t partner;
    InteractionSource source_status;
    PartnerEffect effect;

    bool applies(std::span<const QubitStatus> status) const;
};

struct ResolvedStatusProbe {
    uint32_t qubit;
    InteractionSource source_status;
};

struct ResolvedNonComputationalOperation {
    std::variant<std::monostate, ResolvedTransition, ResolvedInteraction, ResolvedStatusProbe>
        annotation;

    // Expanded physical-gate index for a generated interaction or transition
    // hook. Explicit annotations have no parent and act at their own position.
    std::optional<uint32_t> generated_by;
};

// Owns the expanded AST and decoded annotations. Operation indices remain
// aligned with the AST and trajectory events. Instrument IDs refer to owned
// values, so copying or moving this object never leaves borrowed model pointers.
class ResolvedNonComputationalCircuit {
  public:
    ResolvedNonComputationalCircuit(Circuit annotated, const NonComputationalModel& model,
                                    std::string_view caller,
                                    std::vector<std::optional<uint32_t>> generated_by = {});

    const Circuit& circuit() const { return circuit_; }
    const ResolvedNonComputationalOperation& operation(uint32_t op_index) const {
        assert(op_index < operations_.size());
        return operations_[op_index];
    }
    const TransitionInstrument& instrument(TransitionId id) const {
        assert(id.value < instruments_.size());
        return instruments_[id.value];
    }

    // A spreading interaction uses its original operation index as a leakage
    // channel, preserving the Pauli draw before a trapped spreading event.
    ResolvedTransition transition(uint32_t op_index) const;

  private:
    Circuit circuit_;
    std::vector<ResolvedNonComputationalOperation> operations_;
    std::vector<TransitionInstrument> instruments_;
};

}  // namespace clifft
