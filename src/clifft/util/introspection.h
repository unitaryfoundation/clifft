#pragma once

// Shared string-formatting utilities for VM bytecode introspection. This
// header also re-exports the HIR-only helpers used by the bindings.

#include "clifft/backend/backend.h"
#include "clifft/util/hir_introspection.h"

#include <string>

namespace clifft {

std::string opcode_to_str(Opcode op);

// Opcode classification helpers.
[[nodiscard]] constexpr bool is_two_axis_opcode(Opcode op) noexcept {
    return op == Opcode::OP_FRAME_CNOT || op == Opcode::OP_FRAME_CZ ||
           op == Opcode::OP_FRAME_SWAP || op == Opcode::OP_ARRAY_CNOT ||
           op == Opcode::OP_ARRAY_CZ || op == Opcode::OP_ARRAY_SWAP ||
           op == Opcode::OP_SWAP_MEAS_INTERFERE || op == Opcode::OP_SWAP_MEAS_INTERFERE_FORCED ||
           op == Opcode::OP_ARRAY_U4;
}

[[nodiscard]] constexpr bool is_one_axis_opcode(Opcode op) noexcept {
    return op == Opcode::OP_FRAME_H || op == Opcode::OP_FRAME_S || op == Opcode::OP_FRAME_S_DAG ||
           op == Opcode::OP_ARRAY_H || op == Opcode::OP_ARRAY_S || op == Opcode::OP_ARRAY_S_DAG ||
           op == Opcode::OP_EXPAND || op == Opcode::OP_ARRAY_T || op == Opcode::OP_ARRAY_T_DAG ||
           op == Opcode::OP_EXPAND_T || op == Opcode::OP_EXPAND_T_DAG ||
           op == Opcode::OP_ARRAY_ROT || op == Opcode::OP_EXPAND_ROT || op == Opcode::OP_ARRAY_U2;
}

[[nodiscard]] constexpr bool is_meas_opcode(Opcode op) noexcept {
    return op == Opcode::OP_MEAS_DORMANT_STATIC || op == Opcode::OP_MEAS_DORMANT_RANDOM ||
           op == Opcode::OP_MEAS_ACTIVE_DIAGONAL || op == Opcode::OP_MEAS_ACTIVE_INTERFERE ||
           op == Opcode::OP_SWAP_MEAS_INTERFERE || op == Opcode::OP_MEAS_DORMANT_STATIC_FORCED ||
           op == Opcode::OP_MEAS_DORMANT_RANDOM_FORCED ||
           op == Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED ||
           op == Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED ||
           op == Opcode::OP_SWAP_MEAS_INTERFERE_FORCED;
}

std::string format_instruction(const Instruction& inst);

}  // namespace clifft
