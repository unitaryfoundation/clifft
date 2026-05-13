// Directed tests for the five forced-outcome measurement kernels.
//
// Each forced kernel mirrors its sampling counterpart but reads the
// desired outcome from state.forced_record[classical_idx] instead of
// sampling, accumulates the log-probability of that outcome into
// state.forced_log_probability under sample_branch()'s dust-clamping
// policy, and sets state.forced_reachable = false on unreachable
// outcomes so the caller can short-circuit. Post-measurement
// renormalization is preserved (matching the sampling variants) so
// the state norm does not underflow on deep trajectories. See
// svm_forced_kernels.h for the full contract.
//
// These tests exercise the kernels directly, without the dispatcher
// (which is wired up in a separate step).

#include "clifft/svm/svm.h"
#include "clifft/svm/svm_forced_kernels.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <vector>

using namespace clifft;
using Catch::Matchers::WithinAbs;

namespace {

// Configure a fresh SchrodingerState with a one-byte forced record.
// Caller fills the record byte before calling the kernel.
SchrodingerState make_state(uint32_t peak_rank, std::span<const uint8_t> record) {
    SchrodingerState state(peak_rank, /*num_measurements=*/1);
    state.forced_record = record;
    return state;
}

}  // namespace

// =============================================================================
// dormant_static: outcome is deterministic (= p_x[v] XOR sign).
// Force matching outcome: probability = 1, log_prob += 0, reachable=true.
// Force mismatched outcome: probability = 0, reachable=false.
// =============================================================================

TEST_CASE("forced dormant_static: matching outcome accumulates zero log-prob") {
    std::vector<uint8_t> record{0};
    auto state = make_state(/*peak_rank=*/2, record);
    // p_x[0] = 0 -> deterministic outcome 0
    bool ok = exec_meas_dormant_static_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == true);
    REQUIRE(state.forced_log_probability == 0.0);
    REQUIRE(state.forced_reachable == true);
    REQUIRE(state.meas_record[0] == 0);
}

TEST_CASE("forced dormant_static: mismatched outcome marks unreachable") {
    std::vector<uint8_t> record{1};
    auto state = make_state(/*peak_rank=*/2, record);
    // p_x[0] = 0 -> deterministic 0; record says 1 -> mismatch
    bool ok = exec_meas_dormant_static_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == false);
    REQUIRE(state.forced_reachable == false);
}

TEST_CASE("forced dormant_static: sign-inverted outcome is matched correctly") {
    std::vector<uint8_t> record{1};
    auto state = make_state(/*peak_rank=*/2, record);
    // p_x[0] = 0, sign = true -> deterministic 1; record says 1 -> match
    bool ok = exec_meas_dormant_static_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/true);
    REQUIRE(ok == true);
    REQUIRE(state.forced_log_probability == 0.0);
}

// =============================================================================
// dormant_random: outcome is uniform over {0, 1}.
// Force either outcome: probability = 1/2, log_prob += log(1/2).
// =============================================================================

TEST_CASE("forced dormant_random: any outcome accumulates log(1/2)") {
    for (uint8_t forced : {uint8_t{0}, uint8_t{1}}) {
        std::vector<uint8_t> record{forced};
        auto state = make_state(/*peak_rank=*/2, record);
        bool ok = exec_meas_dormant_random_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/false);
        REQUIRE(ok == true);
        REQUIRE_THAT(state.forced_log_probability, WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(state.forced_reachable == true);
        REQUIRE(state.meas_record[0] == forced);
        // Frame is anchored to the abstract eigenstate.
        REQUIRE((state.p_x[0] & 1) == forced);
        REQUIRE((state.p_z[0] & 1) == 0);
    }
}

// =============================================================================
// active_diagonal: probabilities come from squared norms of v[i] (low half)
// vs v[i+half] (high half). Forced outcome maps to one of these branches.
// =============================================================================

TEST_CASE("forced active_diagonal: deterministic |0> branch forced to 0 accumulates log(1)") {
    std::vector<uint8_t> record{0};
    auto state = make_state(/*peak_rank=*/1, record);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {0.0, 0.0};

    bool ok = exec_meas_active_diagonal_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == true);
    REQUIRE(state.forced_log_probability == 0.0);  // log(1)
    REQUIRE(state.forced_reachable == true);
    REQUIRE(state.active_k == 0);
    REQUIRE(state.v()[0] == std::complex<double>{1.0, 0.0});
}

TEST_CASE("forced active_diagonal: deterministic |0> branch forced to 1 is unreachable") {
    std::vector<uint8_t> record{1};
    auto state = make_state(/*peak_rank=*/1, record);
    state.active_k = 1;
    state.v()[0] = {1.0, 0.0};
    state.v()[1] = {0.0, 0.0};

    bool ok = exec_meas_active_diagonal_forced(state, /*v=*/0, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == false);
    REQUIRE(state.forced_reachable == false);
}

TEST_CASE("forced active_diagonal: non-trivial probabilities accumulate the right log") {
    // |psi> = sqrt(1/3)|0> + sqrt(2/3)|1>; prob(0)=1/3, prob(1)=2/3
    const double inv_sqrt_3 = 1.0 / std::sqrt(3.0);
    const double sqrt_2_3 = std::sqrt(2.0 / 3.0);

    {
        std::vector<uint8_t> record{0};
        auto state = make_state(/*peak_rank=*/1, record);
        state.active_k = 1;
        state.v()[0] = {inv_sqrt_3, 0.0};
        state.v()[1] = {sqrt_2_3, 0.0};
        REQUIRE(exec_meas_active_diagonal_forced(state, 0, 0, false));
        REQUIRE_THAT(state.forced_log_probability, WithinAbs(std::log(1.0 / 3.0), 1e-12));
        // Surviving branch normalized: v[0] should be magnitude 1 after renorm.
        REQUIRE_THAT(std::norm(state.v()[0] * state.gamma()), WithinAbs(1.0, 1e-12));
    }
    {
        std::vector<uint8_t> record{1};
        auto state = make_state(/*peak_rank=*/1, record);
        state.active_k = 1;
        state.v()[0] = {inv_sqrt_3, 0.0};
        state.v()[1] = {sqrt_2_3, 0.0};
        REQUIRE(exec_meas_active_diagonal_forced(state, 0, 0, false));
        REQUIRE_THAT(state.forced_log_probability, WithinAbs(std::log(2.0 / 3.0), 1e-12));
        REQUIRE_THAT(std::norm(state.v()[0] * state.gamma()), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("forced active_diagonal: dust-clamped branch is treated as unreachable") {
    // The sampling kernel routes branch selection through sample_branch(),
    // which clamps any prob <= kDustEpsilon * total to zero so that
    // sample() never emits the dust outcome. record_probabilities() must match:
    // forcing the dust outcome is unreachable, and forcing the surviving
    // outcome is deterministic (log_inc = 0, not log(prob/total)).
    const double dust_amp = 1e-11;  // |amp|^2 = 1e-22 < kDustEpsilon (1e-18)

    {
        std::vector<uint8_t> record{0};
        auto state = make_state(/*peak_rank=*/1, record);
        state.active_k = 1;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {dust_amp, 0.0};
        REQUIRE(exec_meas_active_diagonal_forced(state, 0, 0, false));
        // outcome 0 is the surviving branch -- deterministic from sample()'s
        // perspective, so log_inc must be 0, not log(prob_0 / total).
        REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
        REQUIRE(state.forced_reachable == true);
    }
    {
        std::vector<uint8_t> record{1};
        auto state = make_state(/*peak_rank=*/1, record);
        state.active_k = 1;
        state.v()[0] = {1.0, 0.0};
        state.v()[1] = {dust_amp, 0.0};
        bool ok = exec_meas_active_diagonal_forced(state, 0, 0, false);
        // outcome 1 has dust probability -- sample() never emits it, so
        // record_probabilities() must report it as unreachable.
        REQUIRE(ok == false);
        REQUIRE(state.forced_reachable == false);
    }
}

// =============================================================================
// active_interfere: probabilities come from the |+>/|-> branch sums:
//   prob_plus  = sum |v[i] + v[i+half]|^2
//   prob_minus = sum |v[i] - v[i+half]|^2
// =============================================================================

TEST_CASE("forced active_interfere: |+> branch forced gives log(1) on plus eigenstate") {
    // |+> = (|0> + |1>)/sqrt(2)
    // prob_plus = |1+1|^2/2... actually: prob_plus = |v[0]+v[1]|^2 = (sqrt(2))^2 = 2
    // prob_minus = |v[0]-v[1]|^2 = 0
    // total = 2; prob_plus/total = 1.
    std::vector<uint8_t> record{0};  // b_x=0 (plus)
    auto state = make_state(/*peak_rank=*/1, record);
    state.active_k = 1;
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    state.v()[0] = {inv_sqrt_2, 0.0};
    state.v()[1] = {inv_sqrt_2, 0.0};

    bool ok = exec_meas_active_interfere_forced(state, 0, 0, false);
    REQUIRE(ok == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
}

TEST_CASE("forced active_interfere: |+> state forced to minus is unreachable") {
    std::vector<uint8_t> record{1};  // m_phys=1 -> b_x=1 (minus)
    auto state = make_state(/*peak_rank=*/1, record);
    state.active_k = 1;
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    state.v()[0] = {inv_sqrt_2, 0.0};
    state.v()[1] = {inv_sqrt_2, 0.0};

    bool ok = exec_meas_active_interfere_forced(state, 0, 0, false);
    REQUIRE(ok == false);
    REQUIRE(state.forced_reachable == false);
}

// =============================================================================
// swap_meas_interfere: forwards to active_interfere when f==t; otherwise
// does a frame swap + index-remapped fold. The mathematics is identical
// to active_interfere with a different index mapping.
// =============================================================================

TEST_CASE("forced swap_meas_interfere with f==t forwards to active_interfere") {
    std::vector<uint8_t> record{0};
    auto state = make_state(/*peak_rank=*/1, record);
    state.active_k = 1;
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    state.v()[0] = {inv_sqrt_2, 0.0};
    state.v()[1] = {inv_sqrt_2, 0.0};

    bool ok = exec_swap_meas_interfere_forced(state, /*f=*/0, /*t=*/0, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
}

TEST_CASE("forced swap_meas_interfere with f!=t handles the |+>|0> -> |+> case") {
    // 2-qubit state where axis 0 is in |+> and axis 1 is in |0>. Storage
    // convention: bit 0 of index = axis 0, bit 1 = axis 1. So axis 0 = |+>
    // (v[0] and v[1] both 1/sqrt(2)), axis 1 = |0> (v[2..3] are zero).
    //
    // swap_meas_interfere(f=0, t=1) applies a frame swap of axes 0 and 1
    // and then runs the interfere measurement targeting axis 1. The
    // swapped state has axis 1 in |+>, so the plus branch has probability
    // 1 -- the measurement is deterministic and log(prob/total) = log(1).
    std::vector<uint8_t> record{0};
    auto state = make_state(/*peak_rank=*/2, record);
    state.active_k = 2;
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    state.v()[0] = {inv_sqrt_2, 0.0};
    state.v()[1] = {inv_sqrt_2, 0.0};
    state.v()[2] = {0.0, 0.0};
    state.v()[3] = {0.0, 0.0};

    bool ok = exec_swap_meas_interfere_forced(state, /*f=*/0, /*t=*/1, /*idx=*/0, /*sign=*/false);
    REQUIRE(ok == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
    REQUIRE(state.active_k == 1);
}
