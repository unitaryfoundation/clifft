#pragma once

// Shared string-formatting utilities for HIR and VM bytecode introspection.
// Used by both Python (nanobind) and Wasm (Embind) bindings.

#include "ucc/backend/backend.h"
#include "ucc/frontend/hir.h"

#include <string>

namespace ucc {

// Format a HeisenbergOp's Pauli mask as a human-readable string.
// Example: "+X0*Z3" for destab bit 0 and stab bit 3.
std::string format_pauli_mask(const HeisenbergOp& op);

// Convert an OpType enum to its string name.
std::string op_type_to_str(OpType type);

// Format a complete HIR operation as a human-readable string.
std::string format_hir_op(const HeisenbergOp& op);

// Convert an Opcode enum to its string name.
std::string opcode_to_str(Opcode op);

// Opcode classification helpers.
bool is_two_axis_opcode(Opcode op);
bool is_one_axis_opcode(Opcode op);
bool is_meas_opcode(Opcode op);

// Format a complete VM instruction as a human-readable string.
std::string format_instruction(const Instruction& inst);

}  // namespace ucc
