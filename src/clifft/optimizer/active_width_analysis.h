#pragma once

// Structural active-width analysis for the Heisenberg IR.
//
// Active width is num_qubits - dim(S), where S is the unsigned subspace of
// dormant stabilizers. Tracking S in the initial HIR frame lets a scheduler
// score candidate orders without building a sampling plan or evolving amplitudes.

#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/mask_view.h"

#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

// Tracks the dormant stabilizer subspace S as a reduced GF(2) basis,
// initially the span of Z on every qubit. Rotations can remove a generator;
// measurements can add or replace one. Signs do not affect active width.
//
// Each row combines X bits followed by Z bits, with each half padded to
// words_per_row_ * 64 bits. Padding bits must remain zero. The independent
// rows span S; each pivot is its row's lowest set bit and is zero in every
// other row. Column bitsets transpose the active rows by qubit so sparse
// Pauli queries inspect only the columns in their support, and pivot_owner_
// maps each combined bit directly to its owning row.
class DormantSubspace {
  public:
    // Reusable mutation log for speculative search steps without cloning S.
    class UndoLog {
      public:
        void clear() {
            entries_.clear();
            words_.clear();
        }
        [[nodiscard]] bool empty() const { return entries_.empty(); }

      private:
        friend class DormantSubspace;

        enum class Kind : uint8_t { Intersect, Insert };

        struct Entry {
            Kind kind = Kind::Intersect;
            uint32_t row = 0;
            uint32_t saved_pivot = 0;
            uint32_t saved_begin = 0;
            uint32_t saved_end = 0;
            uint32_t active_row_words = 0;
            size_t word_offset = 0;
        };

        std::vector<Entry> entries_;
        std::vector<uint64_t> words_;
    };

    explicit DormantSubspace(uint32_t num_qubits);

    DormantSubspace(const DormantSubspace&) = default;
    DormantSubspace& operator=(const DormantSubspace&) = default;
    DormantSubspace(DormantSubspace&&) = default;
    DormantSubspace& operator=(DormantSubspace&&) = default;

    [[nodiscard]] uint32_t active_width() const { return num_qubits_ - dimension_; }

    // True when `p` commutes with every current generator of S.
    [[nodiscard]] bool commutes_with_all(const PauliString& p) const {
        return commutes_with_all(p.x(), p.z());
    }
    // A remembered row is only a hint: basis updates can replace or remove it.
    [[nodiscard]] bool commutes_with_all(MaskView x, MaskView z,
                                         uint32_t* anticommuting_row = nullptr) const;

    // True when the unsigned Pauli body belongs to S. For a Pauli commuting
    // with S, this distinguishes a stabilizer from an active operation.
    [[nodiscard]] bool contains(const PauliString& p) const { return contains(p.x(), p.z()); }
    [[nodiscard]] bool contains(MaskView x, MaskView z) const;

    // If axis anticommutes with S, restrict S to generators commuting with
    // axis and return true: dimension drops by one and active width grows by one.
    // Otherwise leave S unchanged and return false.
    bool apply_rotation(const PauliString& axis) { return apply_rotation(axis.x(), axis.z()); }
    bool apply_rotation(MaskView x, MaskView z, UndoLog* undo = nullptr);

    enum class MeasurementEffect : uint8_t { DormantRandom, Classical, Active };

    // A body anticommuting with S replaces a generator (DormantRandom,
    // width unchanged). A body in S is Classical and leaves S unchanged.
    // A commuting body outside S adds a generator (Active, width drops by one).
    MeasurementEffect apply_measurement(const PauliString& body) {
        return apply_measurement(body.x(), body.z());
    }
    MeasurementEffect apply_measurement(MaskView x, MaskView z, UndoLog* undo = nullptr);

    // Reverses all mutations recorded in `undo` in LIFO order and clears `undo`.
    void rollback(UndoLog& undo);

    // Returns unsigned generators. Bases need not match element-for-element;
    // compare subspaces by checking mutual containment of their generators.
    [[nodiscard]] std::vector<PauliString> generators() const;

    // Verifies internal row/column/pivot invariants (intended for tests).
    void verify_invariants() const;

  private:
    static constexpr uint32_t kNoOwner = ~uint32_t{0};

    [[nodiscard]] MaskView row_x(uint32_t index) const {
        return MaskView{std::span<const uint64_t>(
            rows_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }
    [[nodiscard]] MaskView row_z(uint32_t index) const {
        return MaskView{std::span<const uint64_t>(
            rows_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }

    // Populates `anti_out[0 .. active_row_words)` with the bitmask of rows
    // anticommuting with (x, z).
    void compute_anticommuting_rows(MaskView x, MaskView z, std::span<uint64_t> anti_out) const;

    void toggle_row_in_cols(uint32_t row_idx, const uint64_t* vx, const uint64_t* vz,
                            uint32_t begin_w, uint32_t end_w);

    void xor_rows(const uint64_t* vx, const uint64_t* vz, uint32_t v_begin, uint32_t v_end,
                  std::span<const uint64_t> target_rows);

    void write_row(uint32_t slot, uint32_t pivot, uint32_t begin_w, uint32_t end_w,
                   const uint64_t* src_x, const uint64_t* src_z);
    void clear_row(uint32_t slot);
    void move_row(uint32_t from, uint32_t to);
    void erase_row(uint32_t k);

    // Reduces (x, z) modulo S into scratch_[scratch_begin_ .. scratch_end_).
    // An empty range means membership in S; otherwise insert_reduced consumes it.
    void reduce_into_scratch(MaskView x, MaskView z) const;

    // Restricts S to the subspace commuting with (x, z). Returns whether a
    // generator was removed; otherwise leaves S unchanged.
    bool intersect(MaskView x, MaskView z, UndoLog* undo);

    // Inserts the nonzero reduced vector currently in scratch, clearing its new
    // pivot from existing rows to preserve the reduced basis.
    void insert_reduced(UndoLog* undo);

    uint32_t num_qubits_;
    uint32_t words_per_row_;
    uint32_t dimension_;
    std::vector<uint64_t> rows_x_;
    std::vector<uint64_t> rows_z_;
    std::vector<uint64_t> col_x_;
    std::vector<uint64_t> col_z_;
    std::vector<uint32_t> row_begin_;
    std::vector<uint32_t> row_end_;
    std::vector<uint32_t> pivot_;
    std::vector<uint32_t> pivot_owner_;

    std::vector<uint64_t> row_mask_scratch_;
    mutable std::vector<uint64_t> anti_scratch_;
    mutable std::vector<uint64_t> scratch_x_;
    mutable std::vector<uint64_t> scratch_z_;
    mutable uint32_t scratch_begin_ = 0;
    mutable uint32_t scratch_end_ = 0;
};

// Structural effects corresponding to the sampling planner's actions.
// None covers operations that leave the active state untouched.
enum class WidthEffect : uint8_t {
    None,
    RotationStabilizer,     // p in S: the planner emits no action.
    RotationNeutral,        // p commutes with S but p not in S: RotateActivePauli.
    RotationPromote,        // p anticommutes with S: PromoteDormantRotation.
    MeasureClassical,       // p in S: RecordClassical.
    MeasureDormantRandom,   // p anticommutes with S: MeasureDormantRandom.
    MeasureActive,          // p commutes with S but p not in S: MeasureActivePauli.
    InstrumentClassical,    // p in S: InstrumentMode::Classical.
    InstrumentActive,       // p commutes with S but p not in S: InstrumentMode::Active.
    InstrumentActivate,     // p anticommutes with S, damping applies: InstrumentMode::Activate.
    InstrumentDormantTrap,  // p anticommutes with S, damping neglected:
                            // InstrumentMode::DormantTrap.
};

// Schedulers branch on expansions; other effects can be applied eagerly.
[[nodiscard]] constexpr bool is_expanding_effect(WidthEffect effect) {
    return effect == WidthEffect::RotationPromote || effect == WidthEffect::InstrumentActivate;
}

struct WidthTransition {
    uint32_t before = 0;
    uint32_t after = 0;
    WidthEffect effect = WidthEffect::None;
};

struct ActiveWidthTrace {
    uint32_t initial_width = 0;
    uint32_t peak_width = 0;
    uint32_t final_width = 0;
    std::vector<WidthTransition> transitions;
};

// Shared transition logic for full traces and incremental scheduler scoring.
[[nodiscard]] WidthTransition classify_and_apply(const HirModule& hir, const HeisenbergOp& op,
                                                 DormantSubspace& subspace,
                                                 DormantSubspace::UndoLog* undo = nullptr);

namespace detail {

inline bool instrument_damping_neglected(const HirModule& hir, const HeisenbergOp& op) {
    const InstrumentSite& site =
        hir.instrument_sites.at(static_cast<uint32_t>(op.instrument_site_idx()));
    return hir.neglect_instrument_damping ||
           site.probabilities.p_fire[0] == site.probabilities.p_fire[1];
}

// Classify expansion without changing the subspace. The row is a rechecked hint.
inline bool is_expanding(const HirModule& hir, const HeisenbergOp& op,
                         const DormantSubspace& subspace, uint32_t& anticommuting_row) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            return !subspace.commutes_with_all(hir.destab_mask(op), hir.stab_mask(op),
                                               &anticommuting_row);
        case OpType::INSTRUMENT: {
            const MaskView x = hir.destab_mask(op);
            const MaskView z = hir.stab_mask(op);
            if (subspace.commutes_with_all(x, z, &anticommuting_row)) {
                return false;  // Classical or Active: non-expanding.
            }
            return !instrument_damping_neglected(hir, op);
        }
        default:
            return false;
    }
}

// Requires a non-expanding verdict for this op and the current subspace.
// For rotations, only the stabilizer-versus-active membership test remains.
inline WidthTransition apply_non_expanding(const HirModule& hir, const HeisenbergOp& op,
                                           DormantSubspace& subspace,
                                           DormantSubspace::UndoLog* undo = nullptr) {
    if (op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION) {
        const uint32_t width = subspace.active_width();
        const bool stabilizer = subspace.contains(hir.destab_mask(op), hir.stab_mask(op));
        return {width, width,
                stabilizer ? WidthEffect::RotationStabilizer : WidthEffect::RotationNeutral};
    }
    return classify_and_apply(hir, op, subspace, undo);
}

// Per-operation contribution to estimate_dense_work, for scoring a
// candidate schedule incrementally.
[[nodiscard]] double dense_work_contribution(WidthEffect effect, uint32_t before, uint32_t after);

}  // namespace detail

struct ActiveWidthCost {
    uint32_t peak_width = 0;
    double dense_work = 0.0;
};

// Returns peak active width and estimated dense work without allocating a
// per-operation transition vector.
[[nodiscard]] ActiveWidthCost summarize_active_width(const HirModule& hir);
[[nodiscard]] ActiveWidthCost summarize_active_width(const HirModule& hir,
                                                     std::span<const uint32_t> order);

// Returns one structural transition per HIR op without modifying hir.
// Unlike plan_sampling, this does not enforce the dense active-width limit:
// predicting a width trace does not require allocating a dense state.
[[nodiscard]] ActiveWidthTrace analyze_active_width(const HirModule& hir);

// Estimates dense work by summing 2^w for actions that touch coefficients.
// Uses the width before a collapsing measurement and after other dense actions.
// This breaks ties between schedules with equal peak width; it is a heuristic
// that does not account for kernel fusion, batching, or ISA dispatch.
[[nodiscard]] double estimate_dense_work(const ActiveWidthTrace& trace);

}  // namespace clifft
