#pragma once

// Forced-outcome measurement kernels.
//
// Each kernel mirrors its sampling counterpart in svm_kernels.inl, but:
//   - Reads the desired outcome from state.forced_record[classical_idx]
//     instead of sampling from the PRNG.
//   - Accumulates the log-probability of the forced outcome into
//     state.forced_log_probability under the same dust-clamping policy
//     sample_branch() uses. Branches with prob <= kDustEpsilon * total
//     are treated as exactly zero, so:
//       * Forcing the dust outcome sets state.forced_reachable = false
//         (sample() can never emit that record).
//       * Forcing the surviving outcome contributes 0 to the log sum
//         when the other branch is dust (deterministic outcome), not
//         log(prob_b / total).
//       * Non-dust branches contribute log(prob_b / total).
//   - Returns false when the forced outcome is unreachable so the caller
//     can short-circuit the rest of the bytecode.
//   - Keeps post-measurement renormalization (same as the sampling
//     variants) so the state norm does not underflow on deep trajectories.
//
// The kernels are declared in this internal header so both the dispatcher
// and the test suite can call them. Implementations live in
// svm_forced_kernels.cc.

#include "clifft/svm/svm.h"

#include <cstdint>

namespace clifft {

[[nodiscard]] bool exec_meas_dormant_static_forced(SchrodingerState& state, uint16_t v,
                                                   uint32_t classical_idx, bool sign);

[[nodiscard]] bool exec_meas_dormant_random_forced(SchrodingerState& state, uint16_t v,
                                                   uint32_t classical_idx, bool sign);

[[nodiscard]] bool exec_meas_active_diagonal_forced(SchrodingerState& state, uint16_t v,
                                                    uint32_t classical_idx, bool sign);

[[nodiscard]] bool exec_meas_active_interfere_forced(SchrodingerState& state, uint16_t v,
                                                     uint32_t classical_idx, bool sign);

[[nodiscard]] bool exec_swap_meas_interfere_forced(SchrodingerState& state, uint16_t f, uint16_t t,
                                                   uint32_t classical_idx, bool sign);

}  // namespace clifft
