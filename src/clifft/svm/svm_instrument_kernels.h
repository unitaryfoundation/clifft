#pragma once

// Helpers for evaluating and applying one-qubit transition instruments.
//
// For source level s, let p_s be the total transition probability and
// r_s = sqrt(1 - p_s). The channel is
//
//   K_jump[d,s] = sqrt(p[d,s]) |d><s|
//   K_stay     = r_g P_g + r_e P_e
//
// where p_s is the sum of p[d,s] over destination levels d. The kernels use
// only p_g, p_e, r_g, and r_e; the caller separately draws the destination.
//
// A no-jump result scales the g and e amplitudes by r_g and r_e. When a
// transition fires, an active qubit is first projected onto its source level;
// the caller then applies any computational destination change or reports a
// trap. If a dormant qubit's source is not definite, it has no active array
// axis on which to perform the projection, so a continuation performs it
// after the VM traps.
//
// These functions do not draw random numbers. instrument_fire_branch() uses a
// uniform value supplied by the caller. On an active axis, p_x[v] determines
// which array half represents physical g and e; the kernels handle that
// mapping. The caller handles the instruction's sign flag and any destination
// Pauli update.
//
// This internal header is shared by the dispatcher and kernel tests.

#include "clifft/svm/svm.h"

#include <cstdint>

namespace clifft {

// Physical g and e populations on an active axis before damping.
// pop_g + pop_e is the array norm squared at function entry.
struct InstrumentPopulations {
    double pop_g = 0.0;
    double pop_e = 0.0;
};

// Branch decision for one instrument site.
struct InstrumentBranch {
    bool fired = false;
    uint8_t source = 0;  // Physical source level; meaningful only when fired.
};

// In one pass, return the physical g and e populations before damping and
// multiply the corresponding array halves by r_g and r_e. This function does
// not draw a branch or renormalize the result.
//
// If the transition does not fire, the caller renormalizes the damped array.
// If it fires, collapsing to the selected source removes that source's
// previously applied factor. r_g and r_e must be in (0, 1]. When either value
// would be zero, the caller evaluates with r_g = r_e = 1 and performs an
// explicit collapse for both fire and no-fire outcomes.
[[nodiscard]] InstrumentPopulations exec_instrument_damp_eval(SchrodingerState& state, uint16_t v,
                                                              double r_g, double r_e);

// Project active axis v onto physical level `source`. Zero the other array
// half, keep the existing layout and Pauli frame, and rescale the result to
// `target_norm2`. target_norm2 must be the array norm squared before the paired
// damping/evaluation call. No measurement record is written because this is
// transition back-action, not a circuit measurement.
void exec_instrument_collapse_active(SchrodingerState& state, uint16_t v, uint8_t source,
                                     double target_norm2);

// Expand the array for the next dormant axis and apply r_g and r_e to the new
// halves in the same pass. This increments active_k and divides gamma by
// sqrt(2), matching OP_EXPAND. With r_g = r_e = 1 it is a plain expansion.
// Before expansion, a dormant random qubit has equal g and e populations, so
// the caller can draw the branch without evaluating the array first.
void exec_instrument_expand_damp(SchrodingerState& state, uint16_t v, double r_g, double r_e);

// Use u in [0, 1) to choose among a jump from g, a jump from e, and no jump.
// Their weights are p_g * pop_g, p_e * pop_e, and the remaining norm. A
// population at or below kDustEpsilon times the total is treated as zero so a
// numerically empty source is never selected for collapse.
[[nodiscard]] InstrumentBranch instrument_fire_branch(InstrumentPopulations pops, double p_g,
                                                      double p_e, double u);

}  // namespace clifft
