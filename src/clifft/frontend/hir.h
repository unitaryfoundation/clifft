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
// - No generic LCU gates (only T fast-path)
// - AG matrices and noise sites stored in side-tables to avoid bloating HIR

#include "clifft/util/bitmask.h"
#include "clifft/util/config.h"

#include "stim.h"

#include <cassert>
#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

namespace clifft {

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

/// Index of the measurement that controls a CONDITIONAL_PAULI
enum class ControllingMeasIdx : uint32_t {};

/// Index into HirModule::noise_sites side-table
enum class NoiseSiteIdx : uint32_t {};

/// Index into HirModule::detector_targets side-table
enum class DetectorIdx : uint32_t {};

/// Index into HirModule::observable_targets side-table
enum class ObservableIdx : uint32_t {};

/// Index into the expectation value record (absolute position)
enum class ExpValIdx : uint32_t {};

// SIMD lane width for Stim types. Stim's TableauSimulator<W> handles
// arbitrary qubit counts via dynamic heap allocation at any W; this
// controls only the SIMD register width for internal bit operations.
constexpr size_t kStimWidth = 64;

// Inline Pauli mask type used throughout the HIR and SVM.
using PauliBitMask = BitMask<kMaxInlineQubits>;

// =============================================================================
// Noise Channel Structures
// =============================================================================
//
// Quantum noise is represented as a set of mutually exclusive Pauli channels.
// Each channel has a rewound Pauli mask (at t=0) and a probability.
// The Back-End will extract these into a noise schedule for gap sampling.

/// A single Pauli error channel with its rewound masks and probability.
struct NoiseChannel {
    PauliBitMask destab_mask;  // X-bits of the rewound Pauli
    PauliBitMask stab_mask;    // Z-bits of the rewound Pauli
    double prob;               // Probability of this channel firing
};

/// A noise site: a collection of mutually exclusive Pauli channels.
/// For X_ERROR(p): single channel with prob p.
/// For DEPOLARIZE1(p): 3 channels (X, Y, Z) each with prob p/3.
/// For DEPOLARIZE2(p): 15 channels (all non-II two-qubit Paulis) each with prob p/15.
struct NoiseSite {
    std::vector<NoiseChannel> channels;
};

/// Readout noise entry: classical bit-flip on a measurement result.
struct ReadoutNoiseEntry {
    uint32_t meas_idx;  // Absolute measurement index to potentially flip
    double prob;        // Flip probability
};

/// Index into HirModule::readout_noise side-table
enum class ReadoutNoiseIdx : uint32_t {};

// Operation types in the HIR
enum class OpType : uint8_t {
    T_GATE,             // T or T_dag gate (pi/8 phase) - FLAG_IS_DAGGER distinguishes
    MEASURE,            // Destructive measurement (Z, X, or multi-Pauli)
    CONDITIONAL_PAULI,  // Classical feedback: apply Pauli if measurement was 1
    NOISE,              // Stochastic Pauli channel (references NoiseSite side-table)
    READOUT_NOISE,      // Classical bit-flip on measurement result
    PHASE_ROTATION,     // Continuous Z-rotation by angle alpha (half-turns)
    DETECTOR,           // Parity check over measurement records
    OBSERVABLE,         // Logical observable accumulator
    EXP_VAL,            // Non-destructive expectation value probe
    NUM_OP_TYPES        // Sentinel: must remain last for binding completeness checks
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
    // Flag constants (matching Instruction flags)
    static constexpr uint8_t FLAG_IS_DAGGER = 1 << 0;
    static constexpr uint8_t FLAG_HIDDEN = 1 << 2;

    // --- Accessors (common to all OpTypes) ---

    [[nodiscard]] OpType op_type() const { return type_; }
    [[nodiscard]] const PauliBitMask& destab_mask() const { return destab_mask_; }
    [[nodiscard]] const PauliBitMask& stab_mask() const { return stab_mask_; }
    [[nodiscard]] bool sign() const { return sign_; }
    [[nodiscard]] uint8_t flags() const { return flags_; }

    // --- Flag accessors and setters ---

    [[nodiscard]] bool is_dagger() const { return (flags_ & FLAG_IS_DAGGER) != 0; }
    void set_dagger(bool v) {
        if (v)
            flags_ |= FLAG_IS_DAGGER;
        else
            flags_ &= ~FLAG_IS_DAGGER;
    }

    [[nodiscard]] bool is_hidden() const { return (flags_ & FLAG_HIDDEN) != 0; }
    void set_hidden(bool v) {
        if (v)
            flags_ |= FLAG_HIDDEN;
        else
            flags_ &= ~FLAG_HIDDEN;
    }

    // --- MEASURE accessor (debug-asserted) ---

    [[nodiscard]] MeasRecordIdx meas_record_idx() const {
        assert(type_ == OpType::MEASURE && "meas_record_idx called on non-MEASURE op");
        return static_cast<MeasRecordIdx>(measure_.meas_record_idx);
    }

    // --- CONDITIONAL_PAULI accessor (debug-asserted) ---

    [[nodiscard]] ControllingMeasIdx controlling_meas() const {
        assert(type_ == OpType::CONDITIONAL_PAULI &&
               "controlling_meas called on non-CONDITIONAL op");
        return static_cast<ControllingMeasIdx>(conditional_.controlling_meas);
    }

    // --- NOISE accessor (debug-asserted) ---

    [[nodiscard]] NoiseSiteIdx noise_site_idx() const {
        assert(type_ == OpType::NOISE && "noise_site_idx called on non-NOISE op");
        return static_cast<NoiseSiteIdx>(noise_.site_idx);
    }

    // --- READOUT_NOISE accessor (debug-asserted) ---

    [[nodiscard]] ReadoutNoiseIdx readout_noise_idx() const {
        assert(type_ == OpType::READOUT_NOISE &&
               "readout_noise_idx called on non-READOUT_NOISE op");
        return static_cast<ReadoutNoiseIdx>(readout_.entry_idx);
    }

    // --- DETECTOR accessor (debug-asserted) ---

    [[nodiscard]] DetectorIdx detector_idx() const {
        assert(type_ == OpType::DETECTOR && "detector_idx called on non-DETECTOR op");
        return static_cast<DetectorIdx>(detector_.target_list_idx);
    }

    // --- OBSERVABLE accessors (debug-asserted) ---

    [[nodiscard]] ObservableIdx observable_idx() const {
        assert(type_ == OpType::OBSERVABLE && "observable_idx called on non-OBSERVABLE op");
        return static_cast<ObservableIdx>(observable_.obs_idx);
    }

    [[nodiscard]] uint32_t observable_target_list_idx() const {
        assert(type_ == OpType::OBSERVABLE &&
               "observable_target_list_idx called on non-OBSERVABLE op");
        return observable_.target_list_idx;
    }

    // --- EXP_VAL accessor (debug-asserted) ---

    [[nodiscard]] ExpValIdx exp_val_idx() const {
        assert(type_ == OpType::EXP_VAL && "exp_val_idx called on non-EXP_VAL op");
        return static_cast<ExpValIdx>(exp_val_.exp_val_idx);
    }

    // --- PHASE_ROTATION accessor (debug-asserted) ---

    [[nodiscard]] double alpha() const {
        assert(type_ == OpType::PHASE_ROTATION && "alpha called on non-PHASE_ROTATION op");
        return phase_.alpha;
    }

    // --- Factory Methods ---

    // Factory for T/T_dag gates
    static HeisenbergOp make_tgate(PauliBitMask destab, PauliBitMask stab, bool s,
                                   bool dagger = false) {
        HeisenbergOp op(OpType::T_GATE, destab, stab, s);
        op.set_dagger(dagger);
        return op;
    }

    // Factory for MEASURE
    static HeisenbergOp make_measure(PauliBitMask destab, PauliBitMask stab, bool s,
                                     MeasRecordIdx meas_idx) {
        HeisenbergOp op(OpType::MEASURE, destab, stab, s);
        op.measure_.meas_record_idx = static_cast<uint32_t>(meas_idx);
        return op;
    }

    // Factory for CONDITIONAL_PAULI
    static HeisenbergOp make_conditional(PauliBitMask destab, PauliBitMask stab, bool s,
                                         ControllingMeasIdx controlling_meas) {
        HeisenbergOp op(OpType::CONDITIONAL_PAULI, destab, stab, s);
        op.conditional_.controlling_meas = static_cast<uint32_t>(controlling_meas);
        return op;
    }

    // Factory for NOISE (quantum Pauli channel)
    static HeisenbergOp make_noise(NoiseSiteIdx site_idx) {
        HeisenbergOp op(OpType::NOISE, 0, 0, false);
        op.noise_.site_idx = static_cast<uint32_t>(site_idx);
        return op;
    }

    // Factory for READOUT_NOISE (classical bit-flip)
    static HeisenbergOp make_readout_noise(ReadoutNoiseIdx entry_idx) {
        HeisenbergOp op(OpType::READOUT_NOISE, 0, 0, false);
        op.readout_.entry_idx = static_cast<uint32_t>(entry_idx);
        return op;
    }

    // Factory for DETECTOR
    static HeisenbergOp make_detector(DetectorIdx target_list_idx) {
        HeisenbergOp op(OpType::DETECTOR, 0, 0, false);
        op.detector_.target_list_idx = static_cast<uint32_t>(target_list_idx);
        return op;
    }

    // Factory for OBSERVABLE
    static HeisenbergOp make_observable(ObservableIdx obs_idx, uint32_t target_list_idx) {
        HeisenbergOp op(OpType::OBSERVABLE, 0, 0, false);
        op.observable_.obs_idx = static_cast<uint32_t>(obs_idx);
        op.observable_.target_list_idx = target_list_idx;
        return op;
    }

    // Factory for EXP_VAL (non-destructive expectation value probe)
    static HeisenbergOp make_exp_val(PauliBitMask destab, PauliBitMask stab, bool s,
                                     ExpValIdx idx) {
        HeisenbergOp op(OpType::EXP_VAL, destab, stab, s);
        op.exp_val_.exp_val_idx = static_cast<uint32_t>(idx);
        return op;
    }

    // Factory for PHASE_ROTATION (continuous Z-rotation)
    static HeisenbergOp make_phase_rotation(PauliBitMask destab, PauliBitMask stab, bool s,
                                            double alpha) {
        HeisenbergOp op(OpType::PHASE_ROTATION, destab, stab, s);
        op.phase_.alpha = alpha;
        return op;
    }

    // Replace the Pauli masks and sign in-place, preserving OpType and flags.
    void set_pauli(PauliBitMask destab, PauliBitMask stab, bool sign) {
        destab_mask_ = destab;
        stab_mask_ = stab;
        sign_ = sign;
    }

  private:
    // Private constructor - use factory methods
    HeisenbergOp(OpType t, PauliBitMask destab, PauliBitMask stab, bool s)
        : destab_mask_(destab), stab_mask_(stab), type_(t), sign_(s), flags_(0) {
        // Zero-initialize the union
        measure_ = {0};
    }

    // --- Data Members ---

    // The rewound Pauli string (topological geometry at t=0)
    PauliBitMask destab_mask_;  // X-bits (destabilizer component)
    PauliBitMask stab_mask_;    // Z-bits (stabilizer component)

    // Payload (interpretation depends on OpType) - 12 bytes
    union {
        // T_GATE: no payload needed (weight is implicit: tan(pi/8))
        // is_dagger_ determines +/-i phase of spawned branch

        // MEASURE: measurement metadata
        struct {
            uint32_t meas_record_idx;  // Index in measurement record (for rec[-k] resolution)
        } measure_;

        // CONDITIONAL_PAULI: classical feedback metadata
        struct {
            uint32_t controlling_meas;  // Absolute index of controlling measurement
        } conditional_;

        // NOISE: index into noise_sites side-table
        struct {
            uint32_t site_idx;
        } noise_;

        // READOUT_NOISE: index into readout_noise side-table
        struct {
            uint32_t entry_idx;
        } readout_;

        // DETECTOR: index into detector_targets side-table
        struct {
            uint32_t target_list_idx;
        } detector_;

        // OBSERVABLE: observable index and target list index
        struct {
            uint32_t obs_idx;          // Which logical observable (0, 1, 2, ...)
            uint32_t target_list_idx;  // Index into observable_targets side-table
        } observable_;

        // PHASE_ROTATION: continuous angle in half-turn units
        struct {
            double alpha;  // Rotation angle: R_Z(alpha) = exp(-i*alpha*pi/2 * Z)
        } phase_;

        // EXP_VAL: index into expectation value record
        struct {
            uint32_t exp_val_idx;
        } exp_val_;
    };

    OpType type_;    // 1 byte
    bool sign_;      // Phase sign: false = +, true = -  (1 byte)
    uint8_t flags_;  // Bitfield flags (1 byte)
    // 1 byte padding -> Total: 32 bytes
};

// At kMaxInlineQubits == 64 the HIR struct is exactly 32 bytes (2 per cache line).
// At larger widths the struct grows (offline AOT data, not hot-path).
#if CLIFFT_MAX_QUBITS == 64
static_assert(sizeof(HeisenbergOp) == 32,
              "HeisenbergOp must be exactly 32 bytes at 64-qubit width");
#endif

// The complete HIR module - output of the Front-End.
//
// =============================================================================
// HIR Module
// =============================================================================
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

    // Side-table for noise sites (indexed by HeisenbergOp::noise_.site_idx)
    // Each NoiseSite contains the rewound Pauli channels for a quantum noise operation.
    std::vector<NoiseSite> noise_sites;

    // Side-table for readout noise entries (indexed by HeisenbergOp::readout_.entry_idx)
    std::vector<ReadoutNoiseEntry> readout_noise;

    // Side-tables for detector and observable measurement targets.
    // Each entry is a list of absolute measurement indices to XOR together.
    std::vector<std::vector<uint32_t>> detector_targets;
    std::vector<std::vector<uint32_t>> observable_targets;

    // Circuit metadata
    uint32_t num_qubits = 0;
    uint32_t num_measurements = 0;         // Visible measurements only (M, MX, MY, MPP, MR, MRX)
    uint32_t num_hidden_measurements = 0;  // Hidden measurements (from reset decomposition)
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;

    // Global weight accumulator.
    // The Front-End accumulates the dominant terms factored out of each gate.
    // For T gates: exp(ipi/8) * cos(pi/8)
    // The VM ignores this (deferred normalization), but it's needed for
    // correct amplitude tracking if exporting to physical routing tools.
    std::complex<double> global_weight = {1.0, 0.0};

    // --- Source Mapping (Explorer) ---
    // Parallel to ops: source_map[i] lists the original source line(s)
    // that produced ops[i]. Inner vector has multiple entries when an
    // optimizer fuses operations (e.g. T+T -> S carries both lines).
    // Empty vector means the entire map was invalidated by an optimization
    // pass that could not maintain it.
    std::vector<std::vector<uint32_t>> source_map;

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

}  // namespace clifft
