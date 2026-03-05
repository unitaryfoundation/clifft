#pragma once

// Instruction factory helpers for SVM/RISC-level tests.
// These construct individual Instruction objects and minimal CompiledModule
// programs for testing the bytecode interpreter in isolation.

#include "ucc/backend/backend.h"

#include <cstdint>
#include <vector>

namespace ucc {
namespace test {

// ---------------------------------------------------------------------------
// Frame opcode factories
// ---------------------------------------------------------------------------

inline Instruction make_frame_cnot(uint16_t c, uint16_t t) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_CNOT;
    i.axis_1 = c;
    i.axis_2 = t;
    return i;
}

inline Instruction make_frame_cz(uint16_t c, uint16_t t) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_CZ;
    i.axis_1 = c;
    i.axis_2 = t;
    return i;
}

inline Instruction make_frame_h(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_H;
    i.axis_1 = v;
    return i;
}

inline Instruction make_frame_s(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_S;
    i.axis_1 = v;
    return i;
}

inline Instruction make_frame_s_dag(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_S_DAG;
    i.axis_1 = v;
    return i;
}

inline Instruction make_frame_swap(uint16_t a, uint16_t b) {
    Instruction i{};
    i.opcode = Opcode::OP_FRAME_SWAP;
    i.axis_1 = a;
    i.axis_2 = b;
    return i;
}

// ---------------------------------------------------------------------------
// Array opcode factories
// ---------------------------------------------------------------------------

inline Instruction make_array_cnot(uint16_t c, uint16_t t) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_CNOT;
    i.axis_1 = c;
    i.axis_2 = t;
    return i;
}

inline Instruction make_array_cz(uint16_t c, uint16_t t) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_CZ;
    i.axis_1 = c;
    i.axis_2 = t;
    return i;
}

inline Instruction make_array_swap(uint16_t a, uint16_t b) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_SWAP;
    i.axis_1 = a;
    i.axis_2 = b;
    return i;
}

inline Instruction make_array_s(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_S;
    i.axis_1 = axis;
    return i;
}

inline Instruction make_array_s_dag(uint16_t axis) {
    Instruction i{};
    i.opcode = Opcode::OP_ARRAY_S_DAG;
    i.axis_1 = axis;
    return i;
}

// ---------------------------------------------------------------------------
// Expand / phase / measurement factories
// ---------------------------------------------------------------------------

inline Instruction make_expand(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_EXPAND;
    i.axis_1 = v;
    return i;
}

inline Instruction make_phase_t(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_PHASE_T;
    i.axis_1 = v;
    return i;
}

inline Instruction make_phase_t_dag(uint16_t v) {
    Instruction i{};
    i.opcode = Opcode::OP_PHASE_T_DAG;
    i.axis_1 = v;
    return i;
}

inline Instruction make_meas_dormant_static(uint16_t v, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_MEAS_DORMANT_STATIC;
    i.axis_1 = v;
    i.classical.classical_idx = classical_idx;
    return i;
}

inline Instruction make_meas_dormant_random(uint16_t v, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_MEAS_DORMANT_RANDOM;
    i.axis_1 = v;
    i.classical.classical_idx = classical_idx;
    return i;
}

inline Instruction make_meas_active_diagonal(uint16_t v, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_MEAS_ACTIVE_DIAGONAL;
    i.axis_1 = v;
    i.classical.classical_idx = classical_idx;
    return i;
}

inline Instruction make_meas_active_interfere(uint16_t v, uint32_t classical_idx) {
    Instruction i{};
    i.opcode = Opcode::OP_MEAS_ACTIVE_INTERFERE;
    i.axis_1 = v;
    i.classical.classical_idx = classical_idx;
    return i;
}

// ---------------------------------------------------------------------------
// Program builder
// ---------------------------------------------------------------------------

// Build a minimal CompiledModule from bytecode.
inline CompiledModule make_program(std::vector<Instruction> bytecode, uint32_t peak_rank,
                                   uint32_t num_meas = 0, uint32_t num_det = 0,
                                   uint32_t num_obs = 0) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = num_meas;
    mod.num_detectors = num_det;
    mod.num_observables = num_obs;
    return mod;
}

}  // namespace test
}  // namespace ucc
