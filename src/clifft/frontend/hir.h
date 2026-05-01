#pragma once

// Heisenberg IR (HIR) Data Structures
//
// The HIR is the output of the Front-End and input to the optimizer and Back-End.
// It represents non-Clifford gates and measurements as abstract Pauli-string operations
// with explicit masks and weights. Clifford gates are absorbed into the tableau and do not
// appear in the HIR.
//
// Pauli masks are stored in a HirModule-owned arena and referenced from each
// HeisenbergOp by an opaque PauliMaskHandle. Variable-sized payloads (noise
// channels, detector/observable target lists) live in side-tables on
// HirModule.

#include "clifft/util/bitmask.h"
#include "clifft/util/config.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_arena.h"

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

// Inline Pauli mask type used in legacy paths that have not yet migrated
// to runtime-width arena storage (e.g. NoiseChannel).
using PauliBitMask = BitMask<kMaxInlineQubits>;

// Copy a Stim PauliString row (xs or zs) into our fixed-width PauliBitMask.
inline PauliBitMask stim_to_bitmask(const stim::simd_bits_range_ref<kStimWidth>& bits, uint32_t n) {
    PauliBitMask m;
    uint32_t words = (n + 63) / 64;
    for (uint32_t w = 0; w < words && w < kMaxInlineWords; ++w) {
        m.w[w] = bits.u64[w];
    }
    return m;
}

// Copy a Stim PauliString row into the prefix of an existing arena slot.
inline void stim_to_mask_view(const stim::simd_bits_range_ref<kStimWidth>& bits, uint32_t n,
                              MutableMaskView dst) {
    dst.zero_out();
    uint32_t words = (n + 63) / 64;
    for (uint32_t w = 0; w < words && w < dst.num_words(); ++w) {
        dst.words[w] = bits.u64[w];
    }
}

// =============================================================================
// Noise Channel Structures
// =============================================================================
//
// Quantum noise is represented as a set of mutually exclusive Pauli channels.
// Each channel has a rewound Pauli mask (at t=0) and a probability.
// The Back-End will extract these into a noise schedule for gap sampling.

/// A single Pauli error channel: a handle into the surrounding arena
/// (HirModule::noise_channel_masks or ConstantPool::noise_channel_masks)
/// and a firing probability. The channel sign byte is unused.
struct NoiseChannel {
    PauliMaskHandle mask;
    double prob;
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

// Sentinel handle value indicating that an op carries no Pauli mask.
inline constexpr PauliMaskHandle kNoMask = static_cast<PauliMaskHandle>(~uint32_t{0});

// A single operation in the Heisenberg IR.
//
// Layout: 16 bytes, aligned to 8 (4 ops per 64-byte L1 cache line).
//   offset 0: PauliMaskHandle mask_handle_  (4 bytes; kNoMask for non-mask ops)
//   offset 4: OpType type_                  (1 byte)
//   offset 5: uint8_t flags_                (1 byte)
//   offset 6: padding                       (2 bytes)
//   offset 8: union payload                 (8 bytes; double-aligned)
//
// For mask-carrying ops, mask_handle_ indexes into HirModule::pauli_masks.
// The (X, Z, sign) triple is stored in the arena slot. The Pauli string is
// encoded in the computational basis: X_i set means bit i of x; Z_i set
// means bit i of z; both set means Y_i.
//
// Construct via HirModule::append_* builders, which allocate the mask slot
// and populate the op atomically.
struct HeisenbergOp {
    // Flag constants (matching Instruction flags)
    static constexpr uint8_t FLAG_IS_DAGGER = 1 << 0;
    static constexpr uint8_t FLAG_HIDDEN = 1 << 2;

    // --- Accessors (common to all OpTypes) ---

    [[nodiscard]] OpType op_type() const { return type_; }
    [[nodiscard]] PauliMaskHandle mask_handle() const { return mask_handle_; }
    [[nodiscard]] bool has_mask() const { return mask_handle_ != kNoMask; }
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

  private:
    friend struct HirModule;

    // Internal factories: callers go through HirModule's append_* builders.
    static HeisenbergOp make_tgate(PauliMaskHandle handle, bool dagger) {
        HeisenbergOp op(OpType::T_GATE, handle);
        op.set_dagger(dagger);
        return op;
    }
    static HeisenbergOp make_measure(PauliMaskHandle handle, MeasRecordIdx meas_idx) {
        HeisenbergOp op(OpType::MEASURE, handle);
        op.measure_.meas_record_idx = static_cast<uint32_t>(meas_idx);
        return op;
    }
    static HeisenbergOp make_conditional(PauliMaskHandle handle,
                                         ControllingMeasIdx controlling_meas) {
        HeisenbergOp op(OpType::CONDITIONAL_PAULI, handle);
        op.conditional_.controlling_meas = static_cast<uint32_t>(controlling_meas);
        return op;
    }
    static HeisenbergOp make_noise(NoiseSiteIdx site_idx) {
        HeisenbergOp op(OpType::NOISE, kNoMask);
        op.noise_.site_idx = static_cast<uint32_t>(site_idx);
        return op;
    }
    static HeisenbergOp make_readout_noise(ReadoutNoiseIdx entry_idx) {
        HeisenbergOp op(OpType::READOUT_NOISE, kNoMask);
        op.readout_.entry_idx = static_cast<uint32_t>(entry_idx);
        return op;
    }
    static HeisenbergOp make_detector(DetectorIdx target_list_idx) {
        HeisenbergOp op(OpType::DETECTOR, kNoMask);
        op.detector_.target_list_idx = static_cast<uint32_t>(target_list_idx);
        return op;
    }
    static HeisenbergOp make_observable(ObservableIdx obs_idx, uint32_t target_list_idx) {
        HeisenbergOp op(OpType::OBSERVABLE, kNoMask);
        op.observable_.obs_idx = static_cast<uint32_t>(obs_idx);
        op.observable_.target_list_idx = target_list_idx;
        return op;
    }
    static HeisenbergOp make_exp_val(PauliMaskHandle handle, ExpValIdx idx) {
        HeisenbergOp op(OpType::EXP_VAL, handle);
        op.exp_val_.exp_val_idx = static_cast<uint32_t>(idx);
        return op;
    }
    static HeisenbergOp make_phase_rotation(PauliMaskHandle handle, double alpha) {
        HeisenbergOp op(OpType::PHASE_ROTATION, handle);
        op.phase_.alpha = alpha;
        return op;
    }

    HeisenbergOp(OpType t, PauliMaskHandle h) : mask_handle_(h), type_(t), flags_(0), pad_{0, 0} {
        measure_ = {0};
    }

    // --- Data Members ---

    PauliMaskHandle mask_handle_;  // 4 bytes (kNoMask for ops with no Pauli)

    OpType type_;     // 1 byte
    uint8_t flags_;   // 1 byte
    uint8_t pad_[2];  // 2 bytes

    // Payload (interpretation depends on OpType) - 8 bytes
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
};

static_assert(sizeof(HeisenbergOp) == 16, "HeisenbergOp must be exactly 16 bytes");

// The complete HIR module - output of the Front-End.
//
// Holds the linear ops vector plus side-tables for variable-sized payloads
// (noise channels, readout-noise entries, detector and observable target lists)
// and circuit metadata.
//
// Pauli masks for mask-carrying ops live in `pauli_masks`, sized at
// construction. Use the append_* builders to allocate a slot and append an
// op atomically; use destab_mask(op) / stab_mask(op) / sign(op) (or
// mask_at(op) for mutation) to read or modify a slot via the op's handle.
//
// final_tableau is an optional debugging artifact used by the statevector
// path to map GF(2) indices back to the physical basis.
struct HirModule {
    HirModule() = default;

    HirModule(uint32_t n_qubits, size_t num_pauli_masks, size_t num_noise_channels = 0)
        : pauli_masks(n_qubits, num_pauli_masks),
          noise_channel_masks(n_qubits, num_noise_channels) {
        num_qubits = n_qubits;
    }

    // Pauli mask arena for HIR ops (T_GATE, MEASURE, CONDITIONAL_PAULI,
    // PHASE_ROTATION, EXP_VAL). Slots claimed by append_* builders in
    // call order.
    PauliMaskArena pauli_masks;

    // Pauli mask arena for noise channels. Each NoiseChannel inside a
    // NoiseSite carries a handle into this arena. Slots are claimed via
    // claim_noise_channel_mask() when noise sites are appended.
    PauliMaskArena noise_channel_masks;

    // Operations in execution order
    std::vector<HeisenbergOp> ops;

    // Side-table for noise sites (indexed by HeisenbergOp::noise_.site_idx)
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

    // --- Source Mapping (Playground) ---
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

    // --- Mask accessors (module-bound, by op) ---

    [[nodiscard]] MaskView destab_mask(const HeisenbergOp& op) const {
        assert(op.has_mask());
        return pauli_masks.at(op.mask_handle()).x();
    }
    [[nodiscard]] MaskView stab_mask(const HeisenbergOp& op) const {
        assert(op.has_mask());
        return pauli_masks.at(op.mask_handle()).z();
    }
    [[nodiscard]] bool sign(const HeisenbergOp& op) const {
        assert(op.has_mask());
        return pauli_masks.at(op.mask_handle()).sign();
    }
    [[nodiscard]] PauliMaskView mask_view(const HeisenbergOp& op) const {
        assert(op.has_mask());
        return pauli_masks.at(op.mask_handle());
    }
    [[nodiscard]] MutablePauliMaskView mask_at(const HeisenbergOp& op) {
        assert(op.has_mask());
        return pauli_masks.mut_at(op.mask_handle());
    }

    // --- Builders (allocate a mask slot, populate, append op) ---

    HeisenbergOp& append_tgate(MaskView destab, MaskView stab, bool s, bool dagger = false) {
        auto h = claim_pauli_mask(destab, stab, s);
        ops.push_back(HeisenbergOp::make_tgate(h, dagger));
        return ops.back();
    }
    HeisenbergOp& append_measure(MaskView destab, MaskView stab, bool s, MeasRecordIdx idx) {
        auto h = claim_pauli_mask(destab, stab, s);
        ops.push_back(HeisenbergOp::make_measure(h, idx));
        return ops.back();
    }
    HeisenbergOp& append_conditional(MaskView destab, MaskView stab, bool s,
                                     ControllingMeasIdx idx) {
        auto h = claim_pauli_mask(destab, stab, s);
        ops.push_back(HeisenbergOp::make_conditional(h, idx));
        return ops.back();
    }
    HeisenbergOp& append_phase_rotation(MaskView destab, MaskView stab, bool s, double alpha) {
        auto h = claim_pauli_mask(destab, stab, s);
        ops.push_back(HeisenbergOp::make_phase_rotation(h, alpha));
        return ops.back();
    }
    HeisenbergOp& append_exp_val(MaskView destab, MaskView stab, bool s, ExpValIdx idx) {
        auto h = claim_pauli_mask(destab, stab, s);
        ops.push_back(HeisenbergOp::make_exp_val(h, idx));
        return ops.back();
    }
    HeisenbergOp& append_noise(NoiseSiteIdx idx) {
        ops.push_back(HeisenbergOp::make_noise(idx));
        return ops.back();
    }
    HeisenbergOp& append_readout_noise(ReadoutNoiseIdx idx) {
        ops.push_back(HeisenbergOp::make_readout_noise(idx));
        return ops.back();
    }
    HeisenbergOp& append_detector(DetectorIdx idx) {
        ops.push_back(HeisenbergOp::make_detector(idx));
        return ops.back();
    }
    HeisenbergOp& append_observable(ObservableIdx obs_idx, uint32_t target_list_idx) {
        ops.push_back(HeisenbergOp::make_observable(obs_idx, target_list_idx));
        return ops.back();
    }

    // --- In-place mask mutation ---

    /// Replace the (X, Z, sign) of an existing op's mask slot.
    void set_pauli(const HeisenbergOp& op, MaskView destab, MaskView stab, bool s) {
        auto m = mask_at(op);
        m.x().copy_from(destab);
        m.z().copy_from(stab);
        m.set_sign(s);
    }

    /// Replace just the sign on an existing op's mask slot.
    void set_sign(const HeisenbergOp& op, bool s) { mask_at(op).set_sign(s); }

    /// Convert an existing mask-carrying op to a T_GATE while preserving
    /// its mask handle. Resets the arena slot's sign to false to match
    /// the peephole pass's normalization convention.
    void demote_to_tgate(HeisenbergOp& op, bool dagger) {
        assert(op.has_mask());
        mask_at(op).set_sign(false);
        op.type_ = OpType::T_GATE;
        op.flags_ = 0;
        op.set_dagger(dagger);
        op.measure_ = {0};
    }

    /// Convert an existing mask-carrying op to a PHASE_ROTATION while
    /// preserving its mask handle. Resets the arena slot's sign to false.
    void demote_to_phase_rotation(HeisenbergOp& op, double alpha) {
        assert(op.has_mask());
        mask_at(op).set_sign(false);
        op.type_ = OpType::PHASE_ROTATION;
        op.flags_ = 0;
        op.phase_.alpha = alpha;
    }

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

  private:
    /// Claim the next pauli_masks slot, populate it, return the handle.
    PauliMaskHandle claim_pauli_mask(MaskView destab, MaskView stab, bool s) {
        assert(next_pauli_mask_ < pauli_masks.size() && "HirModule pauli_masks arena exhausted");
        auto h = static_cast<PauliMaskHandle>(next_pauli_mask_++);
        auto slot = pauli_masks.mut_at(h);
        slot.x().copy_from(destab);
        slot.z().copy_from(stab);
        slot.set_sign(s);
        return h;
    }

    size_t next_pauli_mask_ = 0;

  public:
    /// Claim the next noise_channel_masks slot, populate (X, Z) bits, and
    /// return the handle. Sign is unused for noise channels.
    PauliMaskHandle claim_noise_channel_mask(MaskView destab, MaskView stab) {
        assert(next_noise_channel_mask_ < noise_channel_masks.size() &&
               "HirModule noise_channel_masks arena exhausted");
        auto h = static_cast<PauliMaskHandle>(next_noise_channel_mask_++);
        auto slot = noise_channel_masks.mut_at(h);
        slot.x().copy_from(destab);
        slot.z().copy_from(stab);
        return h;
    }

  private:
    size_t next_noise_channel_mask_ = 0;
};

}  // namespace clifft
