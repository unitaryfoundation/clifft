#pragma once

// Instrument kernels: the array-level pieces of an exact state-dependent
// jump site.
//
// A transition instrument attached to one qubit carries per-source jump
// probabilities (p_g, p_e). The exact channel is the Kraus set
//
//   K_jump,s = sqrt(p_s) |dest_s><s|              (one per source s)
//   K_stay   = r_g P_g + r_e P_e,  r_s = sqrt(1 - p_s)
//
// so the fire probability p_g<P_g> + p_e<P_e> is runtime state and the
// no-fire branch applies the weak damping filter diag(r_g, r_e). These
// kernels realize that channel per site classification -- the same
// active/dormant, diagonal-basis classification that selects measurement
// opcodes. A site whose localized basis is not Z-like on its axis is the
// lowering's problem (basis-change sandwich), not a kernel variant.
//
// None of these kernels rolls the PRNG: instrument_fire_branch() turns one
// caller-supplied uniform variate into the branch decision, so every
// function here is deterministic and directly oracle-testable. The
// destination fixup for a jump whose destination differs from its source
// is a Pauli and is likewise the caller's, through the existing frame
// machinery.
//
// Conventions shared with the measurement kernels: coefficients and levels
// are physical -- the active-axis kernels read p_x[v] to map them onto
// array halves, and p_z[v] commutes with every real diagonal here. A
// compile-time localization sign is folded in by the caller before the
// call. Renormalization goes through SchrodingerState::scale_magnitude.
//
// The kernels are declared in this internal header so both the dispatcher
// and the test suite can call them. Implementations live in
// svm_instrument_kernels.cc.

#include "clifft/svm/svm.h"

#include <cstdint>

namespace clifft {

// Populations of the two physical levels on one active-axis qubit,
// accumulated before any damping. pop_g + pop_e is the array norm-squared
// at site entry: the renormalization target for whichever branch follows.
struct InstrumentPopulations {
    double pop_g = 0.0;
    double pop_e = 0.0;
};

// Branch decision for one instrument site.
struct InstrumentBranch {
    bool fired = false;
    uint8_t source = 0;  // Physical source level; meaningful only when fired.
};

// Fused damp + evaluate on active axis v: one pass that accumulates the
// pre-damp physical populations and multiplies the array halves by the
// damp coefficients (r_g, r_e). No renormalization and no draw happen
// here -- the caller draws the branch from the returned populations, then
// either renormalizes the no-fire state by
// scale_magnitude(sqrt((pop_g + pop_e) / (r_g^2 pop_g + r_e^2 pop_e)))
// or collapses the fire state. Applying the damp before the draw is
// exact: on fire, the pre-applied r_source is a scalar on the surviving
// half and cancels in the collapse renormalization, while the fire
// probability uses the pre-damp populations accumulated here.
//
// Precondition: r_g, r_e in (0, 1]. A p = 1 source (r = 0) would zero the
// half a subsequent fire must renormalize, so an exact p = 1 site lowers
// as the eval-only call (r_g = r_e = 1, array untouched) followed by a
// collapse on every branch -- including no-fire, whose posterior excludes
// the certain-fire source -- instead of the fused form.
[[nodiscard]] InstrumentPopulations exec_instrument_damp_eval(SchrodingerState& state, uint16_t v,
                                                              double r_g, double r_e);

// Forced projection of active axis v onto physical level `source`, in
// place: the discarded half is zeroed while the array layout, active_k,
// and the Pauli frame all stay put, because downstream bytecode was
// compiled for this layout. gamma absorbs sqrt(target_norm2 / kept),
// preserving the physical norm across the site the same way the
// measurement kernels do. `target_norm2` is the raw array norm-squared
// at site entry: the populations total of the paired damp_eval call. A
// standalone collapse obtains it from an eval-only call (it needs the
// populations for its draw anyway); a stale total double-counts the
// rescale. No measurement record is written: this is instrument
// back-action, not a Born measurement.
void exec_instrument_collapse_active(SchrodingerState& state, uint16_t v, uint8_t source,
                                     double target_norm2);

// Fused subspace expansion + damp for a dormant-random qubit: promotes
// axis v (which must be the next dormant axis, v == active_k) exactly
// like OP_EXPAND -- array doubles, gamma /= sqrt(2) -- while multiplying
// the two halves by the damp coefficients (r_g, r_e) in the same pass.
// With r_g = r_e = 1 this is exactly the plain expansion. No evaluation
// is needed at such a site: a dormant-random qubit's level populations
// are exactly half-half, so the caller draws from
// pop_g = pop_e = (array norm-squared) / 2 and renormalizes the no-fire
// branch by scale_magnitude(sqrt(2 / (r_g^2 + r_e^2))). The same
// (0, 1] precondition and p = 1 lowering recipe as damp_eval apply.
void exec_instrument_expand_damp(SchrodingerState& state, uint16_t v, double r_g, double r_e);

// Frame-level forced collapse of a dormant-random qubit onto level
// `level`, mirroring the state math of the dormant-random measurement
// kernel with the outcome forced and no record written: extracts the
// phase (-1)^(p_x[v] * level) into gamma and re-anchors the frame
// (p_x[v] = level, p_z[v] = 0). Both levels of a dormant-random qubit
// carry probability exactly one half, so no renormalization is needed.
// Unlike the active-axis kernels, `level` is the localized (abstract)
// outcome the frame anchors to, exactly as in the measurement kernel.
void exec_instrument_collapse_dormant(SchrodingerState& state, uint16_t v, uint8_t level);

// Turn one uniform variate u in [0, 1) into the branch decision for an
// instrument site: fire-with-source-g occupies [0, p_g pop_g / N),
// fire-with-source-e the next p_e pop_e / N, and no-fire the remainder,
// where N = pop_g + pop_e. A population at or below the measurement
// kernels' dust threshold (kDustEpsilon * N) is clamped to zero first, so
// a source whose population is floating-point dust is never selected --
// the collapse it would trigger has no ray left to renormalize.
[[nodiscard]] InstrumentBranch instrument_fire_branch(InstrumentPopulations pops, double p_g,
                                                      double p_e, double u);

}  // namespace clifft
