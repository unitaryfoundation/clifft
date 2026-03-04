#pragma once

#include "ucc/frontend/hir.h"

#include "stim.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

namespace ucc {

// =============================================================================
// RISC Execution Bytecode Opcodes
// =============================================================================
//
// The VM uses a localized RISC instruction set. All multi-qubit global topology
// is compressed into local 1-qubit and 2-qubit virtual axis operations by the
// Back-End AOT compiler. The VM never evaluates basis spans or commutations.

enum class Opcode : uint8_t {
    // Frame Opcodes (Zero-cost dormant updates. Update p_x, p_z only)
    OP_FRAME_CNOT,
    OP_FRAME_CZ,
    OP_FRAME_H,
    OP_FRAME_S,
    OP_FRAME_SWAP,

    // Array Opcodes (Update p_x, p_z AND loop over v[] to swap/mix)
    OP_ARRAY_CNOT,
    OP_ARRAY_CZ,
    OP_ARRAY_SWAP,

    // Local Math & Expansion
    OP_EXPAND,       // Virtual H_v on dormant: k -> k+1, gamma /= sqrt(2)
    OP_PHASE_T,      // Active diagonal T phase
    OP_PHASE_T_DAG,  // Active diagonal T-dagger phase

    // Measurement
    OP_MEAS_DORMANT_STATIC,    // Deterministic outcome from p_x
    OP_MEAS_DORMANT_RANDOM,    // Random pivot, algebraic phase to gamma
    OP_MEAS_ACTIVE_DIAGONAL,   // Z-basis filter, halves array (k -> k-1)
    OP_MEAS_ACTIVE_INTERFERE,  // X-basis fold, halves array (k -> k-1)

    // Classical / Errors
    OP_APPLY_PAULI,  // XORs a full N-bit mask from ConstantPool into P
    OP_DETECTOR      // Parity check over measurement records
};

// =============================================================================
// 32-Byte RISC Instruction Bytecode
// =============================================================================
//
// Exactly 32 bytes ensures 2 instructions per 64-byte L1 cache line.
// Uses uint16_t for axis indices, enabling 512-qubit scaling without
// architectural changes.

struct alignas(32) Instruction {
    Opcode opcode;           // Offset 0
    uint8_t base_phase_idx;  // Offset 1
    uint8_t flags;           // Offset 2
    uint8_t _pad;            // Offset 3
    uint16_t axis_1;         // Offset 4 (Virtual axis target/control)
    uint16_t axis_2;         // Offset 6 (Virtual axis target 2)

    // 24 bytes remaining for payload (offsets 8..31)
    union {
        // Variant A: Local Math Payloads
        struct {
            double weight_re;   // Offset 8
            double weight_im;   // Offset 16
            uint8_t _pad_a[8];  // Offset 24
        } math;

        // Variant B: Classical targets (Measurements)
        struct {
            uint32_t classical_idx;  // Offset 8
            uint32_t expected_val;   // Offset 12
            uint8_t _pad_b[16];      // Offset 16
        } classical;

        // Variant C: Full Pauli injection (Errors/Conditionals)
        struct {
            uint32_t cp_mask_idx;    // Offset 8 (Index into ConstantPool)
            uint32_t condition_idx;  // Offset 12
            uint8_t _pad_c[16];      // Offset 16
        } pauli;

        uint8_t raw[24];  // Full payload access
    };
};

static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");

// =============================================================================
// Constant Pool
// =============================================================================
//
// Heavy data referenced by index from Instructions. Kept separate to maintain
// the 32-byte Instruction size constraint.

struct ConstantPool {
    // Forward Clifford tableau at circuit end (for statevector expansion).
    // Computed as U_C = U_phys * V_cum^dag at end of compilation.
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Global scalar gamma accumulated during compilation
    std::complex<double> global_weight = {1.0, 0.0};

    // Full N-bit Pauli masks for OP_APPLY_PAULI (indexed by cp_mask_idx)
    std::vector<stim::PauliString<kStimWidth>> pauli_masks;

    // Target lists for detector parity checks
    std::vector<std::vector<uint32_t>> detector_targets;
};

// =============================================================================
// Compiled Module
// =============================================================================
//
// Complete output of the Back-End: bytecode + constant pool + metadata.

struct CompiledModule {
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    uint32_t peak_rank = 0;         // Maximum active dimension k reached
    uint32_t num_measurements = 0;  // Total visible measurements
    uint32_t num_detectors = 0;     // Total detectors
    uint32_t num_observables = 0;   // Total observables
};

// =============================================================================
// Back-End API
// =============================================================================

/// Lower HIR to executable RISC bytecode.
/// Tracks virtual frame V_cum, compresses multi-qubit Paulis to local ops.
CompiledModule lower(const HirModule& hir);

}  // namespace ucc
