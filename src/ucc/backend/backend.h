#pragma once

#include "ucc/frontend/hir.h"

#include "stim.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

namespace ucc {

// =============================================================================
// Execution Bytecode Opcodes
// =============================================================================

enum class Opcode : uint8_t {
    // T-Gate Fast Paths (hardcoded tan(π/8) weight)
    OP_BRANCH,        // New dimension: array doubles, β ∉ span(V)
    OP_COLLIDE,       // Existing dimension: in-place butterfly, β ∈ span(V)
    OP_SCALAR_PHASE,  // β=0: diagonal phase application (no array change)

    // Generic LCU (arbitrary non-Clifford via constant pool) - future
    OP_BRANCH_LCU,
    OP_COLLIDE_LCU,
    OP_SCALAR_PHASE_LCU,

    // Measurement & State Collapse
    OP_MEASURE_MERGE,          // Anti-commuting: sample + shrink array
    OP_MEASURE_FILTER,         // Commuting (β ∈ span(V)): sample + zero half
    OP_MEASURE_DETERMINISTIC,  // Deterministic (β=0): sign-track only

    // Topology & Classical Logic
    OP_AG_PIVOT,     // Aaronson-Gottesman sign-tracker update
    OP_CONDITIONAL,  // Apply Pauli if measurement was 1
    OP_INDEX_CNOT,   // Basis relabeling within GF(2) space (future)
    OP_DETECTOR,     // Classical parity check (future)

    // Control Flow
    OP_POSTSELECT  // Mid-circuit early abort (future)
};

// =============================================================================
// 32-Byte Instruction Bytecode
// =============================================================================
//
// Exactly 32 bytes ensures 2 instructions per 64-byte L1 cache line.
// Uses a union for type-specific payloads.

struct alignas(32) Instruction {
    // --- Offset 0: Common Header (8 bytes) ---
    Opcode opcode;           // Offset 0 (1 byte) - discriminant
    uint8_t base_phase_idx;  // Offset 1 (1 byte) - maps to {1, i, -1, -i}
    union {
        bool is_dagger;      // Offset 2 (1 byte) - T-gate: true for T†
        bool reuse_outcome;  // Offset 2 (1 byte) - AG_PIVOT: true to reuse MERGE outcome
    };
    uint8_t ag_ref_outcome;     // Offset 3 (1 byte) - reference outcome for measurements
    uint32_t commutation_mask;  // Offset 4 (4 bytes) - pre-computed sign interference

    // --- Offset 8: Payload (24 bytes) ---
    union {
        // Variant A: T-Gate / Measurement (BRANCH, COLLIDE, SCALAR_PHASE, MEASURE_*)
        struct {
            uint64_t destab_mask;  // Offset 8 (8 bytes)
            uint64_t stab_mask;    // Offset 16 (8 bytes)
            uint32_t x_mask;       // Offset 24 (4 bytes) - GF(2) index routing
            uint32_t bit_index;    // Offset 28 (4 bytes) - dimension index
        } branch;

        // Variant B: AG Pivot / Conditional
        struct {
            uint32_t payload_idx;  // Offset 8 - index into ag_matrices
            union {
                uint32_t controlling_meas;  // Offset 12 - CONDITIONAL: index into meas_record
                uint32_t ag_stab_slot;      // Offset 12 - AG_PIVOT: stabilizer row for pivot
            };
            uint64_t destab_mask;  // Offset 16 - for CONDITIONAL Pauli
            uint64_t stab_mask;    // Offset 24 - for CONDITIONAL Pauli
        } meta;
    };
};

static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");

// =============================================================================
// GF(2) Basis Tracker
// =============================================================================
//
// Tracks the evolving GF(2) vector space during lowering. Each non-Clifford
// gate (T, T†) may add a new dimension. Measurements may remove dimensions.
//
// Maintains row-echelon form in auxiliary arrays indexed by leading bit,
// enabling O(log n) lookup for span membership rather than O(n) linear search.
//
// CRITICAL: x_mask is uint32_t, limiting rank to 32. This is intentional:
// rank 32 requires 2^32 amplitudes * 16 bytes = 68.7 GB RAM. Rank 33 would
// need 137 GB, exceeding single-node memory.

class GF2Basis {
  public:
    static constexpr uint32_t kMaxRank = 32;

    // Check if β is in the span of current basis.
    // If so, returns the x_mask (which basis vectors XOR to produce β).
    // If not, returns std::nullopt.
    std::optional<uint32_t> find_in_span(stim::bitword<kStimWidth> beta) const;

    // Add a new vector to the basis.
    // Returns the index of the new basis vector.
    // Throws if rank would exceed kMaxRank.
    uint32_t add(stim::bitword<kStimWidth> destab);

    // Remove a dimension after MEASURE_MERGE collapses it.
    // Mirrors the VM's array compaction: all indices above bit_index shift down.
    void remove(uint32_t bit_index);

    [[nodiscard]] uint32_t rank() const;

    [[nodiscard]] const std::vector<stim::bitword<kStimWidth>>& vectors() const;

  private:
    std::vector<stim::bitword<kStimWidth>> basis_;  // X-parts (destab)
    uint64_t echelon_basis_[kStimWidth] = {0};      // Row-echelon indexed by leading bit
    uint32_t echelon_x_mask_[kStimWidth] = {0};     // Corresponding x_mask for each row

    void add_to_echelon(stim::bitword<kStimWidth> beta, uint32_t initial_x_mask);
    void rebuild_echelon();
};

// =============================================================================
// Constant Pool
// =============================================================================
//
// Heavy data referenced by index from Instructions. Kept separate to maintain
// the 32-byte Instruction size constraint.
//
// NOTE: destab_mask/stab_mask use uint64_t, limiting circuits to 64 qubits.
// This matches kStimWidth and is sufficient for near-term hardware. Scaling
// beyond 64 qubits would require indirect mask references or larger instructions.

using AGMatrix = stim::Tableau<kStimWidth>;

struct ConstantPool {
    // AG pivot matrices for measurement collapse (indexed by OP_AG_PIVOT)
    std::vector<AGMatrix> ag_matrices;

    // GF(2) basis for statevector expansion:
    // Each T-gate's rewound Z observable has X-part (destab), Z-part (stab), and sign.
    // gf2_basis[i] = X-part (destab) of basis vector i
    // NOTE: Z-parts and signs are encoded in VM coefficients via base_phase_idx,
    // so only X-parts are needed for statevector expansion.
    std::vector<stim::bitword<64>> gf2_basis;

    // Forward Clifford tableau at circuit end (for statevector expansion)
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Global weight factor accumulated during compilation
    std::complex<double> global_weight = {1.0, 0.0};
};

// =============================================================================
// Compiled Module
// =============================================================================
//
// Complete output of the Back-End: bytecode + constant pool + metadata.
// Ownership is clear: bytecode and constant_pool must be used together.

struct CompiledModule {
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    uint32_t peak_rank;         // Maximum GF(2) dimension reached (determines array size)
    uint32_t num_measurements;  // Total measurements (for result record sizing)
};

// =============================================================================
// Back-End API
// =============================================================================

/// Lower HIR to executable bytecode.
/// Tracks GF(2) basis, computes x_mask/commutation_mask, emits opcodes.
CompiledModule lower(const HirModule& hir);

}  // namespace ucc
