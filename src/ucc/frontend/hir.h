#pragma once

// Heisenberg IR (HIR) Data Structures
//
// The HIR is the output of the Front-End and input to the Back-End (optimizer skipped in MVP).
// It represents non-Clifford gates and measurements as abstract Pauli-string operations
// with explicit masks and weights. Clifford gates are absorbed into the tableau and do not
// appear in the HIR.
//
// Key design decisions:
// - Uses stim::bitword<64> for Pauli masks (zero-overhead wrapper around uint64_t)
// - Uses stim::Tableau<W> for AG pivot matrices (SIMD-aligned, templated for future expansion)
// - MVP uses W=64 (64 qubits); changing W scales to more qubits automatically
// - 32-byte HeisenbergOp for optimal cache alignment (2 ops per 64-byte L1 cache line)
// - No noise operations (deferred to Phase 2)
// - No generic LCU gates (only T fast-path)
// - AG matrices stored in side-table to avoid bloating HIR

#include "stim.h"

#include <cassert>
#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

namespace ucc {

// =============================================================================
// Strong Typedefs for Index Types
// =============================================================================
//
// Zero-overhead strong types using enum class over integer types.
// Prevents accidental argument swapping at compile time (e.g., swapping
// meas_idx and ag_idx in make_measure). Compiles to raw integers in registers.

/// Index into the measurement record (absolute position)
enum class MeasRecordIdx : uint32_t {};

/// Pre-increment for MeasRecordIdx (for iteration)
inline MeasRecordIdx& operator++(MeasRecordIdx& idx) {
    idx = MeasRecordIdx{static_cast<uint32_t>(idx) + 1};
    return idx;
}

/// Index into HirModule::ag_matrices side-table
enum class AgMatrixIdx : uint32_t { None = UINT32_MAX };

/// Index of the measurement that controls a CONDITIONAL_PAULI
enum class ControllingMeasIdx : uint32_t {};

// SIMD width for Stim types. Change this to scale beyond 64 qubits.
// Note: For >64 qubits, upgrade bitword<64> to bitword<256> or bitword<512>
// (for AVX/AVX-512). Do NOT use simd_bits<W> for fixed-size masks - that's
// for variable-length bit arrays and would add heap allocation overhead.
constexpr size_t kStimWidth = 64;

// Operation types in the HIR
enum class OpType : uint8_t {
    T_GATE,            // T or T† gate (π/8 phase) - is_dagger distinguishes
    MEASURE,           // Destructive measurement (Z, X, or multi-Pauli)
    CONDITIONAL_PAULI  // Classical feedback: apply Pauli if measurement was 1
};

// A single operation in the Heisenberg IR.
//
// Layout optimized for 32-byte cache alignment (2 ops per 64-byte L1 cache line):
// - Largest fields first (8-byte masks)
// - Union payload (12 bytes)
// - Small fields at end (type, sign, is_dagger + 1 byte padding)
//
// The destab_mask and stab_mask encode the Pauli string in the computational basis:
// - Bit i of destab_mask = 1 means X_i is present
// - Bit i of stab_mask = 1 means Z_i is present
// - Both bits set means Y_i is present (Y = iXZ)
//
// Uses stim::bitword<64> instead of raw uint64_t for cleaner bitwise operations
// (.popcount(), ^=, &) that align with Stim idioms. Zero memory overhead.
//
// All construction goes through static factory methods (make_tgate, make_measure,
// make_conditional) to ensure type-safe initialization of the union payload.
struct HeisenbergOp {
    // --- Accessors (common to all OpTypes) ---

    [[nodiscard]] OpType op_type() const { return type_; }
    [[nodiscard]] stim::bitword<64> destab_mask() const { return destab_mask_; }
    [[nodiscard]] stim::bitword<64> stab_mask() const { return stab_mask_; }
    [[nodiscard]] bool sign() const { return sign_; }

    // --- T_GATE accessor ---

    [[nodiscard]] bool is_dagger() const { return is_dagger_; }

    // --- MEASURE accessors (debug-asserted) ---

    [[nodiscard]] AgMatrixIdx ag_matrix_idx() const {
        assert(type_ == OpType::MEASURE && "ag_matrix_idx called on non-MEASURE op");
        return static_cast<AgMatrixIdx>(measure_.ag_matrix_idx);
    }
    [[nodiscard]] MeasRecordIdx meas_record_idx() const {
        assert(type_ == OpType::MEASURE && "meas_record_idx called on non-MEASURE op");
        return static_cast<MeasRecordIdx>(measure_.meas_record_idx);
    }
    [[nodiscard]] uint8_t ag_ref_outcome() const {
        assert(type_ == OpType::MEASURE && "ag_ref_outcome called on non-MEASURE op");
        return measure_.ag_ref_outcome;
    }

    // --- CONDITIONAL_PAULI accessor (debug-asserted) ---

    [[nodiscard]] ControllingMeasIdx controlling_meas() const {
        assert(type_ == OpType::CONDITIONAL_PAULI &&
               "controlling_meas called on non-CONDITIONAL op");
        return static_cast<ControllingMeasIdx>(conditional_.controlling_meas);
    }

    // --- Factory Methods ---

    // Factory for T/T† gates
    static HeisenbergOp make_tgate(stim::bitword<64> destab, stim::bitword<64> stab, bool s,
                                   bool dagger = false) {
        HeisenbergOp op(OpType::T_GATE, destab, stab, s, dagger);
        return op;
    }

    // Factory for MEASURE
    // Uses strong typedefs to prevent accidental argument swapping.
    static HeisenbergOp make_measure(stim::bitword<64> destab, stim::bitword<64> stab, bool s,
                                     MeasRecordIdx meas_idx, AgMatrixIdx ag_idx = AgMatrixIdx::None,
                                     uint8_t ag_ref = 0) {
        HeisenbergOp op(OpType::MEASURE, destab, stab, s);
        op.measure_.ag_matrix_idx = static_cast<uint32_t>(ag_idx);
        op.measure_.meas_record_idx = static_cast<uint32_t>(meas_idx);
        op.measure_.ag_ref_outcome = ag_ref;
        return op;
    }

    // Factory for CONDITIONAL_PAULI
    static HeisenbergOp make_conditional(stim::bitword<64> destab, stim::bitword<64> stab, bool s,
                                         ControllingMeasIdx controlling_meas) {
        HeisenbergOp op(OpType::CONDITIONAL_PAULI, destab, stab, s);
        op.conditional_.controlling_meas = static_cast<uint32_t>(controlling_meas);
        return op;
    }

  private:
    // Private constructor - use factory methods
    HeisenbergOp(OpType t, stim::bitword<64> destab, stim::bitword<64> stab, bool s,
                 bool dagger = false)
        : destab_mask_(destab), stab_mask_(stab), type_(t), sign_(s), is_dagger_(dagger) {
        // Zero-initialize the union
        measure_ = {UINT32_MAX, 0, 0};
    }

    // --- Data Members ---

    // The rewound Pauli string (topological geometry at t=0)
    stim::bitword<64> destab_mask_;  // X-bits (destabilizer component) - 8 bytes
    stim::bitword<64> stab_mask_;    // Z-bits (stabilizer component)   - 8 bytes

    // Payload (interpretation depends on OpType) - 12 bytes
    union {
        // T_GATE: no payload needed (weight is implicit: tan(π/8))
        // is_dagger_ determines ±i phase of spawned branch

        // MEASURE: measurement metadata
        struct {
            uint32_t ag_matrix_idx;    // Index into HirModule::ag_matrices (UINT32_MAX if none)
            uint32_t meas_record_idx;  // Index in measurement record (for rec[-k] resolution)
            uint8_t ag_ref_outcome;    // Reference outcome for AG pivot (0 or 1)
        } measure_;

        // CONDITIONAL_PAULI: classical feedback metadata
        struct {
            uint32_t controlling_meas;  // Absolute index of controlling measurement
        } conditional_;
    };

    OpType type_;     // 1 byte
    bool sign_;       // Phase sign: false = +, true = -  (1 byte)
    bool is_dagger_;  // For T_GATE: true = T†, false = T (1 byte)
    // 1 byte padding -> Total: 32 bytes
};

// Verify 32-byte layout for cache alignment
static_assert(sizeof(HeisenbergOp) == 32, "HeisenbergOp must be exactly 32 bytes");

// The complete HIR module - output of the Front-End.
//
// AG pivot matrices use stim::Tableau<kStimWidth> directly. This provides:
// - SIMD-aligned storage for efficient operations
// - Built-in composition via then() and inverse()
// - Template parameter for easy scaling beyond 64 qubits
//
// The Tableau represents the GF(2) change-of-basis transformation computed
// when a measurement anti-commutes with the current stabilizer state.
// Computed as: fwd_after.then(inv_before) where fwd_after is the tableau
// after measurement collapse and inv_before is the inverse of the tableau
// before collapse.
struct HirModule {
    // Operations in execution order
    std::vector<HeisenbergOp> ops;

    // Side-table for AG pivot matrices (indexed by HeisenbergOp::measure.ag_matrix_idx)
    // Uses Stim's Tableau type directly - AG pivots ARE Clifford transformations.
    std::vector<stim::Tableau<kStimWidth>> ag_matrices;

    // Circuit metadata
    uint32_t num_qubits = 0;
    uint32_t num_measurements = 0;

    // Global weight accumulator.
    // The Front-End accumulates the dominant terms factored out of each gate.
    // For T gates: exp(iπ/8) * cos(π/8)
    // The VM ignores this (deferred normalization), but it's needed for
    // correct amplitude tracking if exporting to physical routing tools.
    std::complex<double> global_weight = {1.0, 0.0};

    // --- Optional Debugging Artifacts ---
    // If statevector debugging is enabled, the Front-End saves the final
    // forward tableau here. This represents the geometric reference frame
    // at t=end, needed to map GF(2) indices back to physical qubit basis.
    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Convenience accessors
    [[nodiscard]] size_t num_ops() const { return ops.size(); }

    [[nodiscard]] size_t num_t_gates() const {
        size_t count = 0;
        for (const auto& op : ops) {
            if (op.op_type() == OpType::T_GATE) {
                ++count;
            }
        }
        return count;
    }
};

}  // namespace ucc
