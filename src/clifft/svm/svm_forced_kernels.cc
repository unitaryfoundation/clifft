// Forced-outcome measurement kernel implementations. See
// svm_forced_kernels.h for the API contract.

#include "clifft/svm/svm_forced_kernels.h"

#include "clifft/svm/svm_internal.h"
#include "clifft/svm/svm_math.h"
#include "clifft/util/constants.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>

namespace clifft {

namespace {

// Helper: short-circuit a forced kernel on an unreachable outcome.
// Sets state.forced_reachable = false and returns false so the caller
// can early-out of the dispatcher loop.
[[nodiscard]] bool mark_unreachable(SchrodingerState& state) {
    state.forced_reachable = false;
    return false;
}

// Mirror sample_branch()'s dust-clamping policy when accounting for the
// log-probability of a forced outcome b in {0, 1}. The sampling path
// treats any branch with prob <= kDustEpsilon * total as exactly zero,
// so records containing such an outcome are never emitted by sample().
// probability_of() must agree:
//   - The dust branch is unreachable.
//   - When the other branch is dust, the surviving branch is
//     deterministic and its log-increment is 0 (not log(prob/total)).
//   - Otherwise, log_increment is log(prob_b / total).
//
// Returns (reachable, log_increment). State mutations (renormalization,
// frame updates) still use the actual prob_b -- this helper only affects
// reachability and the log accumulator. Bumps state.dust_clamps on a
// genuine dust event, matching sample_branch()'s telemetry.
[[nodiscard]] std::pair<bool, double> forced_log_increment(SchrodingerState& state, double prob_0,
                                                           double prob_1, double total, uint8_t b) {
    const double eps = kDustEpsilon * total;
    if (prob_1 <= eps) {
        if (prob_1 > 0.0) {
            state.dust_clamps++;
        }
        // Sample-equivalent behavior: outcome 0 is deterministic.
        return (b == 0) ? std::pair{true, 0.0} : std::pair{false, 0.0};
    }
    if (prob_0 <= eps) {
        if (prob_0 > 0.0) {
            state.dust_clamps++;
        }
        return (b == 1) ? std::pair{true, 0.0} : std::pair{false, 0.0};
    }
    const double prob_b = (b == 0) ? prob_0 : prob_1;
    return {true, std::log(prob_b / total)};
}

}  // namespace

bool exec_meas_dormant_static_identity_forced(SchrodingerState& state, uint32_t classical_idx,
                                              bool sign) {
    // Sampling counterpart in the dispatcher writes meas_record[idx] = sign.
    // Forced variant checks the record matches the deterministic sign bit.
    uint8_t expected = sign ? 1U : 0U;
    if (state.forced_record[classical_idx] != expected) {
        return mark_unreachable(state);
    }
    state.meas_record[classical_idx] = expected;
    return true;
}

bool exec_meas_dormant_static_forced(SchrodingerState& state, uint16_t v, uint32_t classical_idx,
                                     bool sign) {
    // Sampling counterpart: writes the deterministic outcome to meas_record.
    // In forced mode we must check the record's requested outcome matches.
    uint8_t deterministic = (bit_get(state.p_x, v) ? 1U : 0U) ^ static_cast<uint8_t>(sign);
    uint8_t requested = state.forced_record[classical_idx];
    if (requested != deterministic) {
        return mark_unreachable(state);
    }
    // prob = 1; log(1) = 0; no accumulator update needed.
    state.meas_record[classical_idx] = deterministic;
    return true;
}

bool exec_meas_dormant_random_forced(SchrodingerState& state, uint16_t v, uint32_t classical_idx,
                                     bool sign) {
    // Sampling counterpart picks m_abs in {0, 1} with prob 1/2 each.
    // The forced variant takes m_abs from the record (after undoing sign).
    uint8_t requested = state.forced_record[classical_idx];
    uint8_t m_abs = requested ^ static_cast<uint8_t>(sign);

    // Both branches have prob 1/2; accumulate log(1/2).
    state.forced_log_probability += std::log(0.5);

    // Phase extraction: (-1)^(p_x[v] * m_abs)
    if (bit_get(state.p_x, v) && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = requested;
    return true;
}

bool exec_meas_active_diagonal_forced(SchrodingerState& state, uint16_t v, uint32_t classical_idx,
                                      bool sign) {
    assert(v == state.active_k - 1 && "Active diagonal measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Same probability computation as the sampling kernel.
    double prob_b0 = 0.0;
    double prob_b1 = 0.0;
    parallel_reduce(static_cast<int64_t>(half), state.active_k, prob_b0, prob_b1,
                    [&](int64_t ii, double& p0, double& p1) {
                        uint64_t i = static_cast<uint64_t>(ii);
                        p0 += std::norm(arr[i]);
                        p1 += std::norm(arr[i + half]);
                    });
    double total = prob_b0 + prob_b1;
    assert(total > 0.0 && "Active diagonal measurement on zero-norm state");

    // Map the requested physical outcome back to the abstract branch index.
    // Sampling: m_phys = m_abs XOR sign, m_abs = b XOR px_v.
    // Forced:   b      = (m_phys XOR sign) XOR px_v.
    uint8_t requested = state.forced_record[classical_idx];
    uint8_t m_abs = requested ^ static_cast<uint8_t>(sign);
    uint8_t b = m_abs ^ static_cast<uint8_t>(px_v);

    // Mirror sample_branch()'s dust policy so records sample() can never
    // emit (due to dust clamping) are reported as unreachable here.
    auto [reachable, log_inc] = forced_log_increment(state, prob_b0, prob_b1, total, b);
    if (!reachable) {
        return mark_unreachable(state);
    }
    state.forced_log_probability += log_inc;
    double prob_b = (b == 0) ? prob_b0 : prob_b1;

    // Phase, fold, decrement active_k, renormalize, frame reset -- identical
    // to the sampling kernel.
    if (b == 1 && pz_v) {
        state.multiply_phase({-1.0, 0.0});
    }
    if (b == 1) {
        int64_t n = static_cast<int64_t>(half);
        parallel_for(n, state.active_k,
                     [&](int64_t ii) { arr[ii] = arr[static_cast<uint64_t>(ii) + half]; });
    }
    state.active_k--;
    state.scale_magnitude(std::sqrt(total / prob_b));
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = requested;
    return true;
}

bool exec_meas_active_interfere_forced(SchrodingerState& state, uint16_t v, uint32_t classical_idx,
                                       bool sign) {
    assert(v == state.active_k - 1 && "Active interfere measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Same |+>/|-> probability sums as the sampling kernel.
    double prob_plus = 0.0;
    double prob_minus = 0.0;
    parallel_reduce(static_cast<int64_t>(half), state.active_k, prob_plus, prob_minus,
                    [&](int64_t ii, double& pp, double& pm) {
                        uint64_t i = static_cast<uint64_t>(ii);
                        auto sum = arr[i] + arr[i + half];
                        auto diff = arr[i] - arr[i + half];
                        pp += std::norm(sum);
                        pm += std::norm(diff);
                    });
    double total = prob_plus + prob_minus;
    assert(total > 0.0 && "Active interfere measurement on zero-norm state");

    // Map requested physical outcome back to b_x:
    // Sampling: m_phys = m_abs XOR sign, m_abs = b_x XOR pz_v.
    // Forced:   b_x    = (m_phys XOR sign) XOR pz_v.
    uint8_t requested = state.forced_record[classical_idx];
    uint8_t m_abs = requested ^ static_cast<uint8_t>(sign);
    uint8_t b_x = m_abs ^ static_cast<uint8_t>(pz_v);

    auto [reachable, log_inc] = forced_log_increment(state, prob_plus, prob_minus, total, b_x);
    if (!reachable) {
        return mark_unreachable(state);
    }
    state.forced_log_probability += log_inc;
    double prob_bx = (b_x == 0) ? prob_plus : prob_minus;

    // Fold (add or subtract), phase, decrement, renormalize, frame reset --
    // identical to the sampling kernel.
    {
        int64_t n = static_cast<int64_t>(half);
        if (b_x == 0) {
            parallel_for(n, state.active_k, [&](int64_t ii) {
                uint64_t i = static_cast<uint64_t>(ii);
                arr[i] = (arr[i] + arr[i + half]) * kInvSqrt2;
            });
        } else {
            parallel_for(n, state.active_k, [&](int64_t ii) {
                uint64_t i = static_cast<uint64_t>(ii);
                arr[i] = (arr[i] - arr[i + half]) * kInvSqrt2;
            });
        }
    }
    if (px_v && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }
    state.active_k--;
    state.scale_magnitude(std::sqrt(total / prob_bx));
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = requested;
    return true;
}

bool exec_swap_meas_interfere_forced(SchrodingerState& state, uint16_t f, uint16_t t,
                                     uint32_t classical_idx, bool sign) {
    assert(t == state.active_k - 1 && "Swap target must be k-1");

    if (f == t) {
        return exec_meas_active_interfere_forced(state, t, classical_idx, sign);
    }

    // Frame update: equivalent to FRAME_SWAP(f, t) before measurement.
    bit_swap(state.p_x, f, state.p_x, t);
    bit_swap(state.p_z, f, state.p_z, t);
    bool px_v = bit_get(state.p_x, t);
    bool pz_v = bit_get(state.p_z, t);

    uint64_t half = 1ULL << t;
    auto* __restrict arr = state.v();
    uint64_t f_bit = 1ULL << f;

    // Probability pass with swapped index mapping (same shape as sampling).
    double prob_plus = 0.0;
    double prob_minus = 0.0;
    parallel_reduce(static_cast<int64_t>(half), state.active_k, prob_plus, prob_minus,
                    [&](int64_t ii, double& pp, double& pm) {
                        uint64_t idx = static_cast<uint64_t>(ii);
                        uint64_t b_f = (idx >> f) & 1;
                        uint64_t base = (idx & ~f_bit) | (b_f << t);

                        auto sum = arr[base] + arr[base | f_bit];
                        auto diff = arr[base] - arr[base | f_bit];
                        pp += std::norm(sum);
                        pm += std::norm(diff);
                    });
    double total = prob_plus + prob_minus;
    assert(total > 0.0 && "Active interfere measurement on zero-norm state");

    uint8_t requested = state.forced_record[classical_idx];
    uint8_t m_abs = requested ^ static_cast<uint8_t>(sign);
    uint8_t b_x = m_abs ^ static_cast<uint8_t>(pz_v);

    auto [reachable, log_inc] = forced_log_increment(state, prob_plus, prob_minus, total, b_x);
    if (!reachable) {
        return mark_unreachable(state);
    }
    state.forced_log_probability += log_inc;
    double prob_bx = (b_x == 0) ? prob_plus : prob_minus;

    // In-place fold with swapped index mapping.
    if (b_x == 0) {
        for (uint64_t idx = 0; idx < half; ++idx) {
            uint64_t b_f = (idx >> f) & 1;
            uint64_t base = (idx & ~f_bit) | (b_f << t);
            arr[idx] = (arr[base] + arr[base | f_bit]) * kInvSqrt2;
        }
    } else {
        for (uint64_t idx = 0; idx < half; ++idx) {
            uint64_t b_f = (idx >> f) & 1;
            uint64_t base = (idx & ~f_bit) | (b_f << t);
            arr[idx] = (arr[base] - arr[base | f_bit]) * kInvSqrt2;
        }
    }

    state.active_k--;
    state.scale_magnitude(std::sqrt(total / prob_bx));

    if (px_v && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }
    bit_set(state.p_x, t, m_abs);
    bit_set(state.p_z, t, false);

    state.meas_record[classical_idx] = requested;
    return true;
}

}  // namespace clifft
