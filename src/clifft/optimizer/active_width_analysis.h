#pragma once

// Structural active-width analysis for the Heisenberg IR.
//
// The sampling planner's active width is a pure function of one GF(2)
// linear-algebra object: the unsigned isotropic subspace S of dormant
// stabilizer generators, initialized to the span of Z on every qubit. Every
// planner decision -- promote a coordinate, collapse one, or leave the
// active state untouched -- reduces to whether an operation's Pauli body
// anticommutes with a generator of S, or, when it commutes with all of
// them, whether it already lies in S. This module recomputes that same
// sequence of decisions directly from HIR-frame Pauli masks, using GF(2)
// elimination in place of the planner's coordinate-frame bookkeeping
// (promoted-qubit tracking, pivot selection, inverse tableau caching).
//
// The payoff is a width predictor cheap enough to evaluate repeatedly: code
// exploring alternative HIR operation orders can score a candidate order's
// peak active width without materializing a coordinate frame or an
// executable plan for it. This file never mutates the HIR and performs no
// execution; it only answers what the width trace of a fixed operation
// sequence would look like.

#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/mask_view.h"

#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

// Tracks the unsigned isotropic subspace S of dormant stabilizer generators
// in fixed HIR (initial-frame) coordinates. This is the same subspace the
// sampling planner evolves through its coordinate frame, but represented
// directly as a GF(2) basis rather than a physical-to-current basis change,
// so deciding whether a Pauli anticommutes with S or lies in it costs
// linear algebra instead of tableau composition.
//
// The basis is kept permanently in reduced row echelon form (RREF) and
// updated incrementally, so every query and update below costs
// O(dimension * words_per_row) with no separate cache to invalidate or
// rebuild. Combined-vector convention: each row (and each Pauli body passed
// in) is treated as one GF(2) vector of length 2 * domain, x occupying
// combined bits [0, domain) and z occupying [domain, 2 * domain), domain =
// words_per_row_ * 64. Bits at or beyond num_qubits within each half are
// always zero.
//
// The basis rows_x_[0, dimension_), rows_z_[0, dimension_) and their
// pivots pivot_[0, dimension_) maintain three invariants:
//
//   I1: the rows are linearly independent and span S.
//   I2: pivot_[i] is the lowest set combined bit of row i.
//   I3: pivots are pairwise distinct, and every row has a 0 at every OTHER
//       row's pivot. This is what "reduced" (as opposed to merely
//       "echelon") means: it lets a query vector be reduced against the
//       rows in any order and still land on the unique remainder, which is
//       what contains() relies on.
//
// The initial state (row q = Z_q, pivot domain + q) satisfies all three:
// those rows are the standard basis vectors of the Z half of the combined
// space, so they are independent (I1), each is its own lowest and only set
// bit (I2), and distinct qubits give distinct pivots with no cross terms
// (I3).
//
// dimension + active_width() == num_qubits always: S is a maximal isotropic
// (Lagrangian) subspace exactly when active_width() == 0, and every unit the
// active width grows shrinks S by exactly one dimension.
//
// Every query and update below has a MaskView-pair overload that does the
// real work and a PauliString overload that forwards to it via p.x()/p.z().
// HIR-frame Pauli bodies already live as two MaskViews per op
// (HirModule::destab_mask/stab_mask, backed by pauli_masks storage), so a
// caller that classifies ops one at a time -- every scheduler here -- passes
// those views straight through and never materializes an owned PauliString.
class DormantSubspace {
  public:
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
    [[nodiscard]] bool commutes_with_all(MaskView x, MaskView z) const;

    // True when `p` is a product of generators of S, equivalently an element
    // of S. S is abelian, so an anticommuting Pauli is never in S; callers
    // that already know commutes_with_all(p) is true use this to tell a
    // stabilizer Pauli from a width-neutral active one.
    [[nodiscard]] bool contains(const PauliString& p) const { return contains(p.x(), p.z()); }
    [[nodiscard]] bool contains(MaskView x, MaskView z) const;

    // Applies the planner's rotation bookkeeping for an operation with Pauli
    // body `axis`: if `axis` anticommutes with a generator of S, replaces S
    // with S intersect axis-perp (dimension drops by one, active width grows
    // by one) and returns true. Otherwise leaves S unchanged and returns
    // false.
    bool apply_rotation(const PauliString& axis) { return apply_rotation(axis.x(), axis.z()); }
    bool apply_rotation(MaskView x, MaskView z);

    enum class MeasurementEffect : uint8_t { DormantRandom, Classical, Active };

    // Applies the planner's measurement-collapse bookkeeping to `body` and
    // reports which branch it took. DormantRandom replaces one generator
    // with `body` (active width unchanged); Active adds `body` as a new,
    // independent generator (active width drops by one); Classical leaves S
    // unchanged.
    MeasurementEffect apply_measurement(const PauliString& body) {
        return apply_measurement(body.x(), body.z());
    }
    MeasurementEffect apply_measurement(MaskView x, MaskView z);

    // Returns the current generators of S as unsigned Pauli bodies (S is an
    // unsigned isotropic subspace; no sign is tracked). apply_rotation and
    // apply_measurement can leave S with different generators depending on
    // pivot choice, so two DormantSubspace instances describe the same
    // subspace exactly when each one's generators are all contained in the
    // other, not when this list matches element-for-element.
    [[nodiscard]] std::vector<PauliString> generators() const;

  private:
    // Defined here rather than in the .cc so every per-row iteration in
    // commutes_with_all, intersect, reduce_into_scratch, and
    // insert_reduced can inline the accessor instead of paying a call per
    // row: perf attributes 6-14% of the pass to these four functions when
    // they are out of line.
    [[nodiscard]] MaskView row_x(uint32_t index) const {
        return MaskView{std::span<const uint64_t>(
            rows_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }
    [[nodiscard]] MaskView row_z(uint32_t index) const {
        return MaskView{std::span<const uint64_t>(
            rows_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }
    [[nodiscard]] MutableMaskView row_x(uint32_t index) {
        return MutableMaskView{std::span<uint64_t>(
            rows_x_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }
    [[nodiscard]] MutableMaskView row_z(uint32_t index) {
        return MutableMaskView{std::span<uint64_t>(
            rows_z_.data() + static_cast<size_t>(index) * words_per_row_, words_per_row_)};
    }

    // Copies (x, z) into scratch_x_/scratch_z_ and reduces the copy against
    // every row in place: for each row i, if the scratch vector has a set
    // bit at pivot_[i], XOR row i into it. I3 makes this correct in any row
    // order, so rows are visited by index without regard to pivot value.
    // contains() reads whether the remainder left in scratch is zero;
    // apply_measurement's two branches read the (possibly nonzero)
    // remainder as an already-reduced vector to hand to insert_reduced.
    void reduce_into_scratch(MaskView x, MaskView z) const;

    // Replaces S with S intersect p-perp, where p = (x, z), given that at
    // least one row anticommutes with p. Scans every row once, recording
    // into anticommute_flags_ which rows anticommute with p and which of
    // those has the largest pivot value (k); returns false and leaves S
    // untouched when no row anticommutes (this is how callers discover
    // whether the precondition holds, in the same pass that would otherwise
    // have to find k separately). Otherwise XORs row k into every other
    // row i with anticommute_flags_[i] set, deletes row k by moving the
    // last row (and its pivot) into slot k, decrements dimension_, and
    // returns true.
    //
    // RREF survives: for i != k with anticommute_flags_[i] set, row i has
    // zeros below pivot_[i] and a zero at pivot_[k] (I3), row k has zeros
    // below pivot_[k], and pivot_[k] > pivot_[i] by the choice of k, so row
    // i XOR row k still has lowest bit pivot_[i]; both rows already have
    // zeros at every other remaining pivot, so the sum does too. Bit
    // pivot_[k] may now appear in other rows, which is fine because row k
    // is gone and pivot_[k] is no longer a pivot.
    //
    // The result spans S intersect p-perp: every modified row now commutes
    // with p (its anticommute flag with p and row k's both were set, so
    // xoring cancels the anticommutation), every untouched row already
    // commuted with p, and the dimension dropped by exactly one (row k was
    // removed and nothing else changed the row count).
    bool intersect(MaskView x, MaskView z);

    // Inserts (r_x, r_z) as a new generator, given that it is nonzero and
    // already reduced against every current row (zero at each row's own
    // pivot_[i]), e.g. the output of reduce_into_scratch. Lets b be the
    // lowest set combined bit of r; XORs r into every existing row that has
    // bit b set, then appends r as a new row with pivot b.
    //
    // RREF survives: r has zeros at every existing pivot (reduced) and
    // zeros below b (b is its own lowest set bit), so a row i with bit b
    // set has b strictly above pivot_[i] (b is set in row i but is not row
    // i's pivot, and pivot_[i] is row i's lowest set bit). XORing r into
    // row i therefore leaves every bit of row i below b unchanged, hence
    // leaves pivot_[i] unchanged, and leaves row i's zeros at every other
    // pivot unchanged (r is zero there). After the loop, bit b is cleared
    // from every row that had it, so no other row has bit b when r is
    // appended with pivot b.
    void insert_reduced(MaskView r_x, MaskView r_z);

    uint32_t num_qubits_;
    uint32_t words_per_row_;
    uint32_t dimension_;
    std::vector<uint64_t> rows_x_;
    std::vector<uint64_t> rows_z_;
    std::vector<uint32_t> pivot_;

    // Per-row anticommute flags computed by intersect(), reused across
    // calls (indices [0, dimension_) are live) so finding and eliminating
    // a pivot never allocates.
    std::vector<uint8_t> anticommute_flags_;

    mutable std::vector<uint64_t> scratch_x_;
    mutable std::vector<uint64_t> scratch_z_;
};

// Per-operation classification of how an HIR op moves the active width.
// Values correspond 1:1 with the sampling planner's action taxonomy (see
// sampling/plan.h): the Rotation* values classify a T_GATE or
// PHASE_ROTATION, Measure* a MEASURE, and Instrument* an INSTRUMENT. None
// covers every op that leaves the active state untouched (NOISE,
// READOUT_NOISE, CONDITIONAL_PAULI, DETECTOR, OBSERVABLE, EXP_VAL).
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

// True for the two WidthEffect values that raise active width. A scheduler
// branches only on these; every other effect is safe to fold into a closure
// sweep (see docs/theory/active-width.md for the closure argument this
// distinction supports).
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

namespace detail {

// Per-transition contribution to estimate_dense_work's planning proxy: 2^w
// for the width w the dense state has when the transition's action runs, or
// 0 for an action that never touches the dense state. See
// estimate_dense_work's comment for the exact per-effect table. Exposed
// separately so a caller that builds a schedule incrementally (the
// scheduling pass) can accumulate this per op instead of replaying a whole
// trace after every candidate move.
[[nodiscard]] double dense_work_contribution(WidthEffect effect, uint32_t before, uint32_t after);

}  // namespace detail

// Classifies `op` against `subspace` exactly as analyze_active_width does,
// applies its effect, and returns the transition (before/after width and
// effect). This is the per-op primitive analyze_active_width loops over;
// the scheduling pass reuses it too, so both agree on the same
// classification by construction rather than by two hand-synchronized
// copies.
[[nodiscard]] WidthTransition classify_and_apply(const HirModule& hir, const HeisenbergOp& op,
                                                 DormantSubspace& subspace);

// Recomputes the sampling planner's active-width trace directly from HIR,
// without building a coordinate frame or symbolic Pauli frame and without
// mutating `hir`. Produces exactly one transition per HIR op, in op order.
//
// Unlike the planner, this analysis never throws on width overflow: it
// predicts the structural trace rather than committing to build a dense
// coefficient state, so it has no reason to enforce
// sampling::kDenseActiveWidthLimit. A caller that needs the planner's
// executable output still goes through sampling::plan_sampling, which
// enforces that limit itself.
[[nodiscard]] ActiveWidthTrace analyze_active_width(const HirModule& hir);

// Planning proxy for the dense work a trace performs: the sum, over
// transitions, of 2^w for each action that touches the dense state, where w
// is the width at the moment that action runs -- the width just before a
// collapsing measurement (MeasureActive), or just after an expansion
// (RotationPromote, InstrumentActivate) or a width-neutral dense action
// (RotationNeutral, InstrumentActive). Every other effect contributes
// nothing: it never reads or writes the dense coefficient state.
//
// This is a planning proxy, not a runtime cost model: kernel fusion,
// batching, and ISA dispatch all change the real per-action cost. It exists
// only so a scheduler can rank candidate traces of otherwise-equal peak
// width by how much dense work they do along the way.
[[nodiscard]] double estimate_dense_work(const ActiveWidthTrace& trace);

}  // namespace clifft
