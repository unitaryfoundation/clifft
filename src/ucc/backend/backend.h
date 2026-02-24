#pragma once

#include "ucc/frontend/hir.h"

#include "stim.h"

#include <bit>
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

    // Noise & QEC
    OP_READOUT_NOISE,  // Classical bit-flip on measurement result
    OP_DETECTOR,       // Parity check over measurement records
    OP_OBSERVABLE,     // Logical observable accumulator

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
    Opcode opcode;              // Offset 0 (1 byte) - discriminant
    uint8_t base_phase_idx;     // Offset 1 (1 byte) - maps to {1, i, -1, -i}
    uint8_t flags;              // Offset 2 (1 byte) - bitfield flags
    uint8_t ag_ref_outcome;     // Offset 3 (1 byte) - reference outcome for measurements
    uint32_t commutation_mask;  // Offset 4 (4 bytes) - pre-computed sign interference

    // Flag constants
    static constexpr uint8_t FLAG_IS_DAGGER = 1 << 0;      // T-gate: true for T†
    static constexpr uint8_t FLAG_REUSE_OUTCOME = 1 << 1;  // AG_PIVOT: reuse preceding outcome
    static constexpr uint8_t FLAG_HIDDEN = 1 << 2;  // Measurement: don't record to meas_record
    static constexpr uint8_t FLAG_USE_LAST_OUTCOME =
        1 << 3;  // CONDITIONAL: use last_outcome instead of meas_record

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

        // Variant C: Readout Noise (OP_READOUT_NOISE)
        // Inline storage avoids pointer chase in hot loop.
        struct {
            double prob;        // Offset 8 (8 bytes) - bit-flip probability
            uint32_t meas_idx;  // Offset 16 (4 bytes) - measurement to flip
            uint32_t _pad0;     // Offset 20 (4 bytes) - padding
            uint64_t _pad1;     // Offset 24 (8 bytes) - padding
        } readout;

        // Variant D: Detector (OP_DETECTOR)
        struct {
            uint32_t target_idx;  // Offset 8 - index into ConstantPool::detector_targets
            uint32_t _pad0;       // Offset 12 - padding
            uint64_t _pad1;       // Offset 16 - padding
            uint64_t _pad2;       // Offset 24 - padding
        } detector;

        // Variant E: Observable (OP_OBSERVABLE)
        struct {
            uint32_t target_idx;  // Offset 8 - index into ConstantPool::observable_targets
            uint32_t obs_idx;     // Offset 12 - which logical observable (0, 1, 2, ...)
            uint64_t _pad1;       // Offset 16 - padding
            uint64_t _pad2;       // Offset 24 - padding
        } observable;
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
// Noise Schedule Entry (Backend)
// =============================================================================
//
// A compiled noise site for the VM's gap sampling algorithm.
// Different from HIR's NoiseSite: adds pc (bytecode position) and precomputed
// total_probability for efficient runtime sampling.

struct NoiseScheduleEntry {
    uint32_t pc;                         // Bytecode index where this noise applies
    double total_probability;            // Sum of all channel probabilities
    std::vector<NoiseChannel> channels;  // Mutually exclusive Pauli channels
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

// =============================================================================
// AG Matrix (Sparse GF(2) Transform)
// =============================================================================
//
// Extracts the Boolean change-of-basis matrix from a Stim Tableau.
// Evaluates measurement pivots using sparse column extraction, completely
// avoiding the allocations and phase-tracking of stim::PauliString.
//
// The sign bit of the Pauli product is intentionally discarded: the error
// frame is a projective Pauli (global phase is unobservable), so only the
// X/Z components matter for error propagation.

class AGMatrix {
  public:
    AGMatrix() = default;

    /// Construct the interleaved boolean columns from a Stim Tableau.
    explicit AGMatrix(const stim::Tableau<kStimWidth>& tab);

    /// Apply the matrix to the SVM's sign trackers (modifies in-place).
    /// Inlined into op_ag_pivot to eliminate 5M cross-TU calls per shot batch.
    inline void apply(uint64_t& destab_signs, uint64_t& stab_signs) const {
        uint64_t new_destab = 0;
        uint64_t new_stab = 0;

        uint64_t d_signs = destab_signs;
        while (d_signs) {
            int i = std::countr_zero(d_signs);
            d_signs &= d_signs - 1;
            new_destab ^= destab_cols_[i].x;
            new_stab ^= destab_cols_[i].z;
        }

        uint64_t s_signs = stab_signs;
        while (s_signs) {
            int i = std::countr_zero(s_signs);
            s_signs &= s_signs - 1;
            new_destab ^= stab_cols_[i].x;
            new_stab ^= stab_cols_[i].z;
        }

        destab_signs = new_destab;
        stab_signs = new_stab;
    }

  private:
    // X and Z columns interleaved for L1 cache locality: both fetched
    // from the same 16-byte-aligned slot, enabling 128-bit XOR (vpxor).
    struct alignas(16) ColPair {
        uint64_t x;
        uint64_t z;
    };
    ColPair destab_cols_[64] = {};
    ColPair stab_cols_[64] = {};
};

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

    // Noise schedule for gap sampling (quantum Pauli noise).
    // Each entry maps to a bytecode pc where the noise applies.
    // The VM processes these before executing the instruction at that pc.
    std::vector<NoiseScheduleEntry> noise_schedule;

    // Cumulative log-survival hazards for O(log N) geometric gap sampling.
    // H[k] = sum_{i=0}^{k} -ln(1 - p_i), strictly monotonically increasing.
    // The SVM binary-searches this array to jump directly to the next error.
    std::vector<double> cumulative_hazards;

    // Target lists for detector and observable parity checks.
    // Each entry is a list of absolute measurement indices to XOR.
    std::vector<std::vector<uint32_t>> detector_targets;
    std::vector<std::vector<uint32_t>> observable_targets;
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
    uint32_t num_measurements;  // Total visible measurements (for result record sizing)
    uint32_t num_detectors;     // Total detectors (for detector record sizing)
    uint32_t num_observables;   // Total observables (for observable record sizing)
};

// =============================================================================
// Back-End API
// =============================================================================

/// Lower HIR to executable bytecode.
/// Tracks GF(2) basis, computes x_mask/commutation_mask, emits opcodes.
CompiledModule lower(const HirModule& hir);

}  // namespace ucc
