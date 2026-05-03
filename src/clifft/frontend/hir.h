#pragma once

// Heisenberg IR (HIR) Data Structures
//
// The HIR is the output of the Front-End and input to the optimizer and Back-End.
// It represents non-Clifford gates and measurements as abstract Pauli-string operations
// with explicit masks and weights. Clifford gates are absorbed into the tableau and do not
// appear in the HIR.
//
// Pauli masks live in HirModule-owned arenas and are referenced from each
// HeisenbergOp by an opaque PauliMaskHandle. Variable-sized payloads (noise
// channels, detector/observable target lists) live in side-tables on HirModule.

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

// Inline Pauli mask retained for legacy debug paths and tests that build
// expected fixed-width references. New code should use MaskView / arena.
using PauliBitMask = BitMask<kMaxInlineQubits>;

/// Copy the first `n` bits of a Stim PauliString row into a destination
/// MutableMaskView. Trailing destination words are zeroed. The destination
/// must be at least `(n + 63) / 64` words wide.
inline void stim_to_mask_view(const stim::simd_bits_range_ref<kStimWidth>& bits, uint32_t n,
                              MutableMaskView dst) {
    const uint32_t words = (n + 63) / 64;
    assert(words <= dst.num_words() && "stim_to_mask_view: destination too narrow");
    for (uint32_t w = 0; w < words; ++w)
        dst.words[w] = bits.u64[w];
    for (uint32_t w = words; w < dst.num_words(); ++w)
        dst.words[w] = 0;
}

/// Truncating copy of a Stim PauliString row into a fixed-width
/// PauliBitMask. Bits beyond kMaxInlineQubits are dropped. Provided for
/// tests that build reference expected masks alongside the runtime arena.
inline PauliBitMask stim_to_bitmask(const stim::simd_bits_range_ref<kStimWidth>& bits, uint32_t n) {
    PauliBitMask m;
    uint32_t words = (n + 63) / 64;
    for (uint32_t w = 0; w < words && w < kMaxInlineWords; ++w) {
        m.w[w] = bits.u64[w];
    }
    return m;
}

// =============================================================================
// Noise Channel Structures
// =============================================================================
//
// A NoiseSite is a list of NoiseChannels; each channel has a Pauli mask
// (handle into the surrounding arena -- HirModule::noise_channel_masks for
// HIR-side sites, ConstantPool::noise_channel_masks for compiled sites)
// and a firing probability. Channel signs are unused (E and -E act
// identically as stochastic Pauli errors).

struct NoiseChannel {
    PauliMaskHandle mask;
    double prob;
};

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

/// Sentinel handle value indicating that an op carries no Pauli mask.
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
// Construct via HirModule::append_* builders (or append_*_empty for the
// frontend, which fills the slot directly from a stim row to avoid
// width-truncating intermediates).
struct HeisenbergOp {
    static constexpr uint8_t FLAG_IS_DAGGER = 1 << 0;
    static constexpr uint8_t FLAG_HIDDEN = 1 << 2;

    [[nodiscard]] OpType op_type() const { return type_; }
    [[nodiscard]] PauliMaskHandle mask_handle() const { return mask_handle_; }
    [[nodiscard]] bool has_mask() const { return mask_handle_ != kNoMask; }
    [[nodiscard]] uint8_t flags() const { return flags_; }

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

    [[nodiscard]] MeasRecordIdx meas_record_idx() const {
        assert(type_ == OpType::MEASURE && "meas_record_idx called on non-MEASURE op");
        return static_cast<MeasRecordIdx>(measure_.meas_record_idx);
    }

    [[nodiscard]] ControllingMeasIdx controlling_meas() const {
        assert(type_ == OpType::CONDITIONAL_PAULI &&
               "controlling_meas called on non-CONDITIONAL op");
        return static_cast<ControllingMeasIdx>(conditional_.controlling_meas);
    }

    [[nodiscard]] NoiseSiteIdx noise_site_idx() const {
        assert(type_ == OpType::NOISE && "noise_site_idx called on non-NOISE op");
        return static_cast<NoiseSiteIdx>(noise_.site_idx);
    }

    [[nodiscard]] ReadoutNoiseIdx readout_noise_idx() const {
        assert(type_ == OpType::READOUT_NOISE &&
               "readout_noise_idx called on non-READOUT_NOISE op");
        return static_cast<ReadoutNoiseIdx>(readout_.entry_idx);
    }

    [[nodiscard]] DetectorIdx detector_idx() const {
        assert(type_ == OpType::DETECTOR && "detector_idx called on non-DETECTOR op");
        return static_cast<DetectorIdx>(detector_.target_list_idx);
    }

    [[nodiscard]] ObservableIdx observable_idx() const {
        assert(type_ == OpType::OBSERVABLE && "observable_idx called on non-OBSERVABLE op");
        return static_cast<ObservableIdx>(observable_.obs_idx);
    }

    [[nodiscard]] uint32_t observable_target_list_idx() const {
        assert(type_ == OpType::OBSERVABLE &&
               "observable_target_list_idx called on non-OBSERVABLE op");
        return observable_.target_list_idx;
    }

    [[nodiscard]] ExpValIdx exp_val_idx() const {
        assert(type_ == OpType::EXP_VAL && "exp_val_idx called on non-EXP_VAL op");
        return static_cast<ExpValIdx>(exp_val_.exp_val_idx);
    }

    [[nodiscard]] double alpha() const {
        assert(type_ == OpType::PHASE_ROTATION && "alpha called on non-PHASE_ROTATION op");
        return phase_.alpha;
    }

  private:
    friend struct HirModule;

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

    PauliMaskHandle mask_handle_;  // 4 bytes (kNoMask for ops with no Pauli)

    OpType type_;     // 1 byte
    uint8_t flags_;   // 1 byte
    uint8_t pad_[2];  // 2 bytes

    // Per-OpType payload variants. Named to keep clang's
    // -Wgnu-anonymous-struct-in-union extension warning quiet.
    struct MeasurePayload {
        uint32_t meas_record_idx;
    };
    struct ConditionalPayload {
        uint32_t controlling_meas;
    };
    struct NoisePayload {
        uint32_t site_idx;
    };
    struct ReadoutPayload {
        uint32_t entry_idx;
    };
    struct DetectorPayload {
        uint32_t target_list_idx;
    };
    struct ObservablePayload {
        uint32_t obs_idx;
        uint32_t target_list_idx;
    };
    struct PhasePayload {
        double alpha;
    };
    struct ExpValPayload {
        uint32_t exp_val_idx;
    };

    union {
        MeasurePayload measure_;
        ConditionalPayload conditional_;
        NoisePayload noise_;
        ReadoutPayload readout_;
        DetectorPayload detector_;
        ObservablePayload observable_;
        PhasePayload phase_;
        ExpValPayload exp_val_;
    };
};

static_assert(sizeof(HeisenbergOp) == 16, "HeisenbergOp must be exactly 16 bytes");

/// HIR module: parsed, traced output of the Front-End. Owns Pauli mask
/// arenas, op vector, side-tables, and circuit metadata. Construct with
/// (num_qubits, num_pauli_masks[, num_noise_channels]) so the arenas are
/// pre-sized; default construction yields empty arenas.
struct HirModule {
    HirModule() = default;

    HirModule(uint32_t n_qubits, size_t num_pauli_masks, size_t num_noise_channels = 0)
        : pauli_masks(n_qubits, num_pauli_masks),
          noise_channel_masks(n_qubits, num_noise_channels) {
        num_qubits = n_qubits;
    }

    PauliMaskArena pauli_masks;
    PauliMaskArena noise_channel_masks;

    std::vector<HeisenbergOp> ops;
    std::vector<NoiseSite> noise_sites;
    std::vector<ReadoutNoiseEntry> readout_noise;
    std::vector<std::vector<uint32_t>> detector_targets;
    std::vector<std::vector<uint32_t>> observable_targets;

    uint32_t num_qubits = 0;
    uint32_t num_measurements = 0;
    uint32_t num_hidden_measurements = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;

    std::complex<double> global_weight = {1.0, 0.0};

    /// Parallel to ops: source_map[i] lists the source line(s) that
    /// produced ops[i]. Empty inner vector means an optimizer pass
    /// invalidated the map for that op.
    std::vector<std::vector<uint32_t>> source_map;

    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // --- Mask accessors ---

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

    // --- Builders that take pre-built MaskViews ---
    //
    // Convenient for tests and any caller that already has the mask data
    // in MaskView form. Fails (via copy_from's contract) if the source
    // mask has set bits beyond the arena's width.

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

    // --- Builders that claim an empty slot, leave caller to fill ---
    //
    // Used by the Front-End to write rewound-Pauli data straight from a
    // stim::PauliString into the arena slot via mask_at(op), avoiding any
    // fixed-width PauliBitMask intermediate that would silently truncate
    // qubits beyond kMaxInlineQubits. Pattern:
    //
    //     auto& op = hir.append_tgate_empty(dagger);
    //     auto slot = hir.mask_at(op);
    //     stim_to_mask_view(rewound.xs, n, slot.x());
    //     stim_to_mask_view(rewound.zs, n, slot.z());
    //     slot.set_sign(rewound.sign);

    HeisenbergOp& append_tgate_empty(bool dagger = false) {
        auto h = claim_empty_pauli_mask();
        ops.push_back(HeisenbergOp::make_tgate(h, dagger));
        return ops.back();
    }
    HeisenbergOp& append_measure_empty(MeasRecordIdx idx) {
        auto h = claim_empty_pauli_mask();
        ops.push_back(HeisenbergOp::make_measure(h, idx));
        return ops.back();
    }
    HeisenbergOp& append_conditional_empty(ControllingMeasIdx idx) {
        auto h = claim_empty_pauli_mask();
        ops.push_back(HeisenbergOp::make_conditional(h, idx));
        return ops.back();
    }
    HeisenbergOp& append_phase_rotation_empty(double alpha) {
        auto h = claim_empty_pauli_mask();
        ops.push_back(HeisenbergOp::make_phase_rotation(h, alpha));
        return ops.back();
    }
    HeisenbergOp& append_exp_val_empty(ExpValIdx idx) {
        auto h = claim_empty_pauli_mask();
        ops.push_back(HeisenbergOp::make_exp_val(h, idx));
        return ops.back();
    }

    // --- Builders for ops that don't carry a Pauli mask ---

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

    // --- Noise channel mask claims (analogous to pauli_masks) ---

    /// Claim the next noise_channel_masks slot, populate (X, Z), return the handle.
    PauliMaskHandle claim_noise_channel_mask(MaskView destab, MaskView stab) {
        assert(next_noise_channel_mask_ < noise_channel_masks.size() &&
               "noise_channel_masks arena exhausted");
        auto h = static_cast<PauliMaskHandle>(next_noise_channel_mask_++);
        auto slot = noise_channel_masks.mut_at(h);
        slot.x().copy_from(destab);
        slot.z().copy_from(stab);
        return h;
    }

    /// Claim the next noise_channel_masks slot zero-initialized; caller
    /// fills via noise_channel_masks.mut_at(h).
    PauliMaskHandle claim_empty_noise_channel_mask() {
        assert(next_noise_channel_mask_ < noise_channel_masks.size() &&
               "noise_channel_masks arena exhausted");
        return static_cast<PauliMaskHandle>(next_noise_channel_mask_++);
    }

    // --- In-place mutation of existing op slots ---

    void set_pauli(const HeisenbergOp& op, MaskView destab, MaskView stab, bool s) {
        auto m = mask_at(op);
        m.x().copy_from(destab);
        m.z().copy_from(stab);
        m.set_sign(s);
    }

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

    [[nodiscard]] size_t num_ops() const { return ops.size(); }

    [[nodiscard]] size_t num_t_gates() const {
        size_t count = 0;
        for (const auto& op : ops) {
            if (op.op_type() == OpType::T_GATE)
                ++count;
        }
        return count;
    }

  private:
    PauliMaskHandle claim_pauli_mask(MaskView destab, MaskView stab, bool s) {
        auto h = claim_empty_pauli_mask();
        auto slot = pauli_masks.mut_at(h);
        slot.x().copy_from(destab);
        slot.z().copy_from(stab);
        slot.set_sign(s);
        return h;
    }

    PauliMaskHandle claim_empty_pauli_mask() {
        assert(next_pauli_mask_ < pauli_masks.size() && "pauli_masks arena exhausted");
        return static_cast<PauliMaskHandle>(next_pauli_mask_++);
    }

    size_t next_pauli_mask_ = 0;
    size_t next_noise_channel_mask_ = 0;
};

}  // namespace clifft
