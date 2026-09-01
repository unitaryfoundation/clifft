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
#include <optional>
#include <vector>

namespace clifft {

// Tracks the unsigned isotropic subspace S of dormant stabilizer generators
// in fixed HIR (initial-frame) coordinates. This is the same subspace the
// sampling planner evolves through its coordinate frame, but represented
// directly as a GF(2) generator list rather than a physical-to-current basis
// change, so deciding whether a Pauli anticommutes with S or lies in it
// costs linear algebra instead of tableau composition.
//
// dimension + active_width() == num_qubits always: S is a maximal isotropic
// (Lagrangian) subspace exactly when active_width() == 0, and every unit the
// active width grows shrinks S by exactly one dimension.
class DormantSubspace {
  public:
    explicit DormantSubspace(uint32_t num_qubits);

    DormantSubspace(const DormantSubspace&) = default;
    DormantSubspace& operator=(const DormantSubspace&) = default;
    DormantSubspace(DormantSubspace&&) = default;
    DormantSubspace& operator=(DormantSubspace&&) = default;

    [[nodiscard]] uint32_t active_width() const { return num_qubits_ - dimension_; }

    // True when `p` commutes with every current generator of S.
    [[nodiscard]] bool commutes_with_all(const PauliString& p) const;

    // True when `p` is a product of generators of S, equivalently an element
    // of S. S is abelian, so an anticommuting Pauli is never in S; callers
    // that already know commutes_with_all(p) is true use this to tell a
    // stabilizer Pauli from a width-neutral active one.
    [[nodiscard]] bool contains(const PauliString& p) const;

    // Applies the planner's rotation bookkeeping for an operation with Pauli
    // body `axis`: if `axis` anticommutes with a generator of S, replaces S
    // with S intersect axis-perp (dimension drops by one, active width grows
    // by one) and returns true. Otherwise leaves S unchanged and returns
    // false.
    bool apply_rotation(const PauliString& axis);

    enum class MeasurementEffect : uint8_t { DormantRandom, Classical, Active };

    // Applies the planner's measurement-collapse bookkeeping to `body` and
    // reports which branch it took. DormantRandom replaces one generator
    // with `body` (active width unchanged); Active adds `body` as a new,
    // independent generator (active width drops by one); Classical leaves S
    // unchanged.
    MeasurementEffect apply_measurement(const PauliString& body);

    // Returns the current generators of S as unsigned Pauli bodies (S is an
    // unsigned isotropic subspace; no sign is tracked). apply_rotation and
    // apply_measurement can leave S with different generators depending on
    // pivot choice, so two DormantSubspace instances describe the same
    // subspace exactly when each one's generators are all contained in the
    // other, not when this list matches element-for-element.
    [[nodiscard]] std::vector<PauliString> generators() const;

  private:
    [[nodiscard]] MaskView row_x(uint32_t index) const;
    [[nodiscard]] MaskView row_z(uint32_t index) const;
    [[nodiscard]] MutableMaskView row_x(uint32_t index);
    [[nodiscard]] MutableMaskView row_z(uint32_t index);
    [[nodiscard]] MutableMaskView echelon_row_x(uint32_t index) const;
    [[nodiscard]] MutableMaskView echelon_row_z(uint32_t index) const;

    [[nodiscard]] std::optional<uint32_t> find_anticommuting_generator(const PauliString& p) const;

    // Replaces S with S intersect p-perp given that generator `pivot`
    // anticommutes with p: XORs every other anticommuting generator with the
    // pivot row, then drops the pivot row. Invalidates the membership cache.
    void intersect_with_pivot(const PauliString& p, uint32_t pivot);

    // Appends `p` as a new, independent generator. Only valid when p is not
    // already in S, which callers establish before invoking this.
    void append_generator(const PauliString& p);

    // Reduces the (work_x, work_z) vector against the cached echelon basis
    // in place and returns the pivot bit of the nonzero remainder, or
    // nullopt when it reduces to zero (the vector was in the cached span).
    [[nodiscard]] std::optional<uint32_t> reduce_against_membership_cache(
        MutableMaskView work_x, MutableMaskView work_z) const;

    // Rebuilds the GF(2) echelon basis from the current generator list when
    // a change has invalidated it. apply_rotation's promoting branch and
    // apply_measurement's DormantRandom branch never query membership, so a
    // run of promotions between two contains() calls rebuilds only once.
    void rebuild_membership_cache_if_dirty() const;

    uint32_t num_qubits_;
    uint32_t words_per_row_;
    uint32_t dimension_;
    std::vector<uint64_t> gen_x_;
    std::vector<uint64_t> gen_z_;

    mutable std::vector<uint64_t> echelon_x_;
    mutable std::vector<uint64_t> echelon_z_;
    mutable std::vector<uint32_t> echelon_pivot_;
    mutable uint32_t echelon_dimension_ = 0;
    mutable bool echelon_dirty_ = true;
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

}  // namespace clifft
