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

#include "clifft/util/config.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_arena.h"
#include "clifft/util/stim_mask.h"

#include "stim.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
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

/// Readout noise entry: classical bit-flip on a measurement result. The
/// flip probability conditions on the recorded bit's value, so asymmetric
/// readout confusion (0->1 vs 1->0 at different rates) is expressible;
/// symmetric noise sets both probabilities equal.
struct ReadoutNoiseEntry {
    uint32_t meas_idx;        // Absolute measurement index to potentially flip
    double prob_zero_to_one;  // Flip probability when the recorded bit is 0
    double prob_one_to_zero;  // Flip probability when the recorded bit is 1

    [[nodiscard]] bool is_symmetric() const { return prob_zero_to_one == prob_one_to_zero; }
};

/// Index into HirModule::readout_noise side-table
enum class ReadoutNoiseIdx : uint32_t {};

/// Index into HirModule::instrument_sites side-table. The index doubles as
/// the site's stable id: instruments are optimization barriers, so no pass
/// reorders or renumbers them, and the compiled module's site -> bytecode
/// offset table is keyed by this index.
enum class InstrumentSiteIdx : uint32_t {};

/// Sentinel handle value indicating that an op carries no Pauli mask.
inline constexpr PauliMaskHandle kNoMask = static_cast<PauliMaskHandle>(~uint32_t{0});

// =============================================================================
// Instrument Sites
// =============================================================================
//
// A five-level transition matrix T[to][from] is compressed before it reaches
// the HIR because only the computational source levels remain quantum. For
// each source s in {G, E}, InstrumentProbabilities stores
//
//   p_fire[s]                  = sum_to T[to][s]
//   p_computational_dest[s][d] = T[d][s], d in {G, E}
//
// and therefore
//
//   p_noncomputational_dest[s]
//       = p_fire[s]
//         - p_computational_dest[s][G]
//         - p_computational_dest[s][E].
//
// The last quantity aggregates destinations LeakG, LeakE, and Lost. A fire
// into that remainder traps to the exact-mode driver, which consults the
// original matrix to choose the specific noncomputational destination.
// Columns whose source is already noncomputational are handled classically
// by the driver and never enter this HIR representation. The no-fire
// probability is 1 - p_fire[s]; a diagonal matrix entry is still a fire
// that lands back on its source, not the no-fire branch.
//
// The probabilities are scalar weights, not quantum operators. An
// INSTRUMENT op's own Pauli mask is the source observable Z_q rewound through
// the trace tableau; its +/- eigenspaces supply the G/E source projectors.
// InstrumentSite::destination_flip_mask is the separately rewound X_q,
// applied when an in-line computational destination differs from the
// collapsed source. Source/destination index 0 is G = |0>, 1 is E = |1>.

struct InstrumentProbabilities {
    // Total fire probability for each computational source.
    double p_fire[2] = {0.0, 0.0};

    // Unconditional fire probability from source s to computational
    // destination d. Dividing by p_fire[s] gives the destination
    // probability conditioned on a fire from s.
    double p_computational_dest[2][2] = {{0.0, 0.0}, {0.0, 0.0}};

    [[nodiscard]] double p_noncomputational_dest(uint8_t source) const {
        assert(source < 2 && "instrument source must be G or E");
        const double remainder =
            p_fire[source] - p_computational_dest[source][0] - p_computational_dest[source][1];
        // Model validation permits derived column sums to drift within its
        // probability tolerance. Do not expose a tiny negative trap weight.
        return std::max(0.0, remainder);
    }
};

// One state-dependent jump site on one qubit. The probability payload is
// the quantum-source compression above; qubit and destination_flip_mask
// supply the site-specific carrier operation.

struct InstrumentSite {
    uint32_t qubit = 0;  // Physical qubit, for suffix rewrites + diagnostics

    // The rewound X observable of the qubit at the site (same tableau
    // conjugation as the op's Z-projector mask): the destination flip a
    // computational fire applies when its destination differs from its
    // source. Handle into HirModule::pauli_masks. Passes that conjugate
    // the op's mask must conjugate this slot identically.
    PauliMaskHandle destination_flip_mask = kNoMask;

    InstrumentProbabilities probabilities;
};

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
    INSTRUMENT,         // State-dependent jump site (references InstrumentSite side-table)
    NUM_OP_TYPES        // Sentinel: must remain last for binding completeness checks
};

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
// Construct via HirModule::append_* builders. For mask-carrying ops, the
// builders take a fill callable that writes (X, Z, sign) into a freshly
// claimed arena slot.
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

    [[nodiscard]] InstrumentSiteIdx instrument_site_idx() const {
        assert(type_ == OpType::INSTRUMENT && "instrument_site_idx called on non-INSTRUMENT op");
        return static_cast<InstrumentSiteIdx>(instrument_.site_idx);
    }

    [[nodiscard]] double alpha() const {
        assert(type_ == OpType::PHASE_ROTATION && "alpha called on non-PHASE_ROTATION op");
        return phase_.alpha;
    }

    // Static factories. Each takes the mask handle (or kNoMask for ops with
    // no Pauli) plus the per-op payload, and produces a fully-initialized
    // HeisenbergOp. Use HirModule::append_* or claim_*_pauli_mask helpers
    // to allocate the slot first; these factories do not validate that the
    // handle belongs to any specific arena.
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
    static HeisenbergOp make_instrument(PauliMaskHandle handle, InstrumentSiteIdx site_idx) {
        HeisenbergOp op(OpType::INSTRUMENT, handle);
        op.instrument_.site_idx = static_cast<uint32_t>(site_idx);
        return op;
    }

    /// In-place rewrite the op as a T_GATE while preserving its mask handle.
    /// Used by optimizer passes (HirModule::demote_to_tgate) that fold a
    /// PHASE_ROTATION at alpha = +/-1/4 into a T gate. Caller is responsible
    /// for resetting the mask slot's sign.
    void reset_to_tgate(bool dagger) {
        type_ = OpType::T_GATE;
        flags_ = 0;
        set_dagger(dagger);
        measure_ = {0};
    }

    /// In-place rewrite the op as a PHASE_ROTATION(alpha) while preserving
    /// its mask handle. Caller is responsible for resetting the mask slot's
    /// sign.
    void reset_to_phase_rotation(double alpha) {
        type_ = OpType::PHASE_ROTATION;
        flags_ = 0;
        phase_.alpha = alpha;
    }

  private:
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
    struct InstrumentPayload {
        uint32_t site_idx;
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
        InstrumentPayload instrument_;
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
    std::vector<InstrumentSite> instrument_sites;
    std::vector<std::vector<uint32_t>> detector_targets;
    std::vector<std::vector<uint32_t>> observable_targets;

    uint32_t num_qubits = 0;
    uint32_t num_measurements = 0;
    uint32_t num_hidden_measurements = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;

    // Under damping="neglect", dormant-random instrument sites skip the
    // expansion and apply no no-fire back-action. The noncomputational
    // policy is module-wide, so trace() records it once rather than on every
    // InstrumentSite.
    bool neglect_instrument_damping = false;

    std::complex<double> global_weight = {1.0, 0.0};

    /// Parallel to ops: source_map[i] lists the source line(s) that
    /// produced ops[i]. Empty inner vector means an optimizer pass
    /// invalidated the map for that op.
    std::vector<std::vector<uint32_t>> source_map;

    std::optional<stim::Tableau<kStimWidth>> final_tableau;

    // Hidden measurement slot trace() assigned to the requested node's
    // reset (set when InstrumentTraceOptions::forced_traceout_node names a
    // node index whose hidden-branch target trace() processes; nullopt
    // when no slot was requested or the node was not encountered).
    std::optional<size_t> forced_traceout_slot;

    /// True when the evolution is a fixed unitary: no measurements, noise,
    /// readout noise, or measurement-conditioned Paulis. Deterministic
    /// modules have a well-defined final statevector including its global
    /// phase; stochastic ones are only defined per shot.
    [[nodiscard]] bool is_deterministic() const {
        for (const auto& op : ops) {
            const OpType type = op.op_type();
            if (type == OpType::MEASURE || type == OpType::NOISE || type == OpType::READOUT_NOISE ||
                type == OpType::CONDITIONAL_PAULI || type == OpType::INSTRUMENT) {
                return false;
            }
        }
        return true;
    }

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

    // --- Lambda-fill builders ---
    //
    // The mask slot is claimed, the fill callable is invoked with a
    // MutablePauliMaskView pointing at the freshly claimed (zeroed) slot,
    // and the op is appended -- atomically at the call site. Example:
    //
    //     hir.append_tgate(/*dagger=*/false, [&](MutablePauliMaskView slot) {
    //         stim_to_mask_view(rewound.xs, n, slot.x());
    //         stim_to_mask_view(rewound.zs, n, slot.z());
    //         slot.set_sign(rewound.sign);
    //     });
    //
    // The fill callable's MutablePauliMaskView argument starts zeroed; the
    // callable need only write the bits it cares about.

    template <typename Fill>
    HeisenbergOp& append_tgate(bool dagger, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_tgate(h, dagger));
        return ops.back();
    }
    template <typename Fill>
    HeisenbergOp& append_measure(MeasRecordIdx idx, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_measure(h, idx));
        return ops.back();
    }
    template <typename Fill>
    HeisenbergOp& append_conditional(ControllingMeasIdx idx, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_conditional(h, idx));
        return ops.back();
    }
    template <typename Fill>
    HeisenbergOp& append_phase_rotation(double alpha, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_phase_rotation(h, alpha));
        return ops.back();
    }
    template <typename Fill>
    HeisenbergOp& append_exp_val(ExpValIdx idx, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_exp_val(h, idx));
        return ops.back();
    }
    template <typename Fill>
    HeisenbergOp& append_instrument(InstrumentSiteIdx idx, Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        ops.push_back(HeisenbergOp::make_instrument(h, idx));
        return ops.back();
    }

    /// Claim a pauli_masks slot referenced from a side-table entry rather
    /// than an op (e.g. an instrument's destination flip), filled via the
    /// callable like the append_* builders.
    template <typename Fill>
    PauliMaskHandle claim_side_mask(Fill&& fill) {
        auto h = claim_empty_pauli_mask();
        fill(pauli_masks.mut_at(h));
        return h;
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

    /// Claim the next noise_channel_masks slot zero-initialized; caller
    /// fills via noise_channel_masks.mut_at(h). Throws (rather than
    /// asserting) so a stale count_noise_channels() upper bound is
    /// caught in Release as well as Debug.
    PauliMaskHandle claim_empty_noise_channel_mask() {
        if (next_noise_channel_mask_ >= noise_channel_masks.size()) {
            throw std::runtime_error(
                "noise_channel_masks arena exhausted: pre-trace counter was too low");
        }
        return static_cast<PauliMaskHandle>(next_noise_channel_mask_++);
    }

    // --- In-place mutation of existing op slots ---

    void set_sign(const HeisenbergOp& op, bool s) { mask_at(op).set_sign(s); }

    /// Convert an existing mask-carrying op to a T_GATE while preserving
    /// its mask handle. Resets the arena slot's sign to false to match
    /// the peephole pass's normalization convention.
    void demote_to_tgate(HeisenbergOp& op, bool dagger) {
        assert(op.has_mask());
        mask_at(op).set_sign(false);
        op.reset_to_tgate(dagger);
    }

    /// Convert an existing mask-carrying op to a PHASE_ROTATION while
    /// preserving its mask handle. Resets the arena slot's sign to false.
    void demote_to_phase_rotation(HeisenbergOp& op, double alpha) {
        assert(op.has_mask());
        mask_at(op).set_sign(false);
        op.reset_to_phase_rotation(alpha);
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
    /// Throws (rather than asserting) so a stale count_pauli_masks() upper
    /// bound is caught in Release as well as Debug.
    PauliMaskHandle claim_empty_pauli_mask() {
        if (next_pauli_mask_ >= pauli_masks.size()) {
            throw std::runtime_error("pauli_masks arena exhausted: pre-trace counter was too low");
        }
        return static_cast<PauliMaskHandle>(next_pauli_mask_++);
    }

    size_t next_pauli_mask_ = 0;
    size_t next_noise_channel_mask_ = 0;
};

}  // namespace clifft
