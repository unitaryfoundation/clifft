// Directed and dense-oracle tests for the instrument kernels.
//
// The kernels are the array-level pieces of a state-dependent transition
// site: fused damp+evaluate on an active axis, in-place forced collapse of an
// active axis, fused expand+damp for a dormant-random qubit, frame-level forced
// collapse of a dormant-random qubit, and the pure fire-branch draw helper.
// None of them rolls the PRNG, so every test is deterministic: the oracle is a
// dense reference that applies the instrument's Kraus operators to a copy of
// the gamma-scaled amplitudes.
//
// These tests exercise the kernels directly. Dispatcher coverage lives in
// test_execute_instruments.cc.

#include "clifft/svm/svm.h"
#include "clifft/svm/svm_forced_kernels.h"
#include "clifft/svm/svm_instrument_kernels.h"
#include "clifft/svm/svm_internal.h"
#include "clifft/util/xoshiro.h"

#include <algorithm>
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

using Amps = std::vector<std::complex<double>>;

constexpr double kTol = 1e-12;

// Deterministic pseudo-random active state: k active axes, normalized
// amplitudes with nonzero real and imaginary parts, cleared frame.
SchrodingerState make_active_state(uint32_t peak_rank, uint32_t k, uint64_t seed) {
    SchrodingerState state(peak_rank, /*num_measurements=*/1);
    state.active_k = k;
    Xoshiro256PlusPlus rng(seed);
    double norm2 = 0.0;
    for (uint64_t i = 0; i < (1ULL << k); ++i) {
        const std::complex<double> a(rng.next_double() - 0.5, rng.next_double() - 0.5);
        state.v()[i] = a;
        norm2 += std::norm(a);
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (uint64_t i = 0; i < (1ULL << k); ++i) {
        state.v()[i] *= inv;
    }
    return state;
}

void set_frame_bit(std::vector<uint64_t>& words, uint16_t v, bool b) {
    if (b) {
        words[v >> 6] |= (1ULL << (v & 63));
    } else {
        words[v >> 6] &= ~(1ULL << (v & 63));
    }
}

bool get_frame_bit(const std::vector<uint64_t>& words, uint16_t v) {
    return (words[v >> 6] >> (v & 63)) & 1;
}

// Gamma-scaled amplitudes: the physically meaningful content of the array.
Amps physical(const SchrodingerState& state) {
    Amps out(state.v_size());
    for (uint64_t i = 0; i < out.size(); ++i) {
        out[i] = state.gamma() * state.v()[i];
    }
    return out;
}

double norm2(const Amps& a) {
    double n = 0.0;
    for (const auto& x : a) {
        n += std::norm(x);
    }
    return n;
}

// Physical level of array index i on axis v under frame bit px.
uint8_t level_of(uint64_t i, uint16_t v, bool px) {
    return static_cast<uint8_t>(((i >> v) & 1) ^ static_cast<uint64_t>(px));
}

// Dense reference: population of physical level `lvl` on axis v.
double ref_population(const Amps& a, uint16_t v, bool px, uint8_t lvl) {
    double p = 0.0;
    for (uint64_t i = 0; i < a.size(); ++i) {
        if (level_of(i, v, px) == lvl) {
            p += std::norm(a[i]);
        }
    }
    return p;
}

// Dense reference: the damping filter K_stay = r_g P_g + r_e P_e.
void ref_damp(Amps& a, uint16_t v, bool px, double r_g, double r_e) {
    for (uint64_t i = 0; i < a.size(); ++i) {
        a[i] *= (level_of(i, v, px) == 0) ? r_g : r_e;
    }
}

// Dense reference: project onto physical level `lvl` on axis v.
void ref_project(Amps& a, uint16_t v, bool px, uint8_t lvl) {
    for (uint64_t i = 0; i < a.size(); ++i) {
        if (level_of(i, v, px) != lvl) {
            a[i] = {0.0, 0.0};
        }
    }
}

// Rescale so norm2(a) == target.
void ref_rescale_to(Amps& a, double target_norm2) {
    const double s = std::sqrt(target_norm2 / norm2(a));
    for (auto& x : a) {
        x *= s;
    }
}

// Dense reference: Hadamard on axis v (frame-free; used with px = pz = 0).
void ref_hadamard(Amps& a, uint16_t v) {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const uint64_t v_bit = 1ULL << v;
    for (uint64_t i = 0; i < a.size(); ++i) {
        if ((i & v_bit) == 0) {
            const std::complex<double> lo = a[i];
            const std::complex<double> hi = a[i | v_bit];
            a[i] = (lo + hi) * inv_sqrt2;
            a[i | v_bit] = (lo - hi) * inv_sqrt2;
        }
    }
}

void require_amps_match(const Amps& got, const Amps& want) {
    REQUIRE(got.size() == want.size());
    for (uint64_t i = 0; i < got.size(); ++i) {
        REQUIRE_THAT(got[i].real(), WithinAbs(want[i].real(), kTol));
        REQUIRE_THAT(got[i].imag(), WithinAbs(want[i].imag(), kTol));
    }
}

}  // namespace

// damp_eval: fused damp + populations on an active axis

TEST_CASE("instrument damp_eval: populations and damped state match the dense reference") {
    const struct {
        double p_g, p_e;
    } rates[] = {{0.0, 0.0}, {0.36, 0.0}, {0.0, 0.75}, {0.36, 0.75}};

    uint64_t seed = 11;
    for (uint32_t k = 1; k <= 3; ++k) {
        for (uint16_t v = 0; v < k; ++v) {
            for (int px = 0; px <= 1; ++px) {
                for (int pz = 0; pz <= 1; ++pz) {
                    for (const auto& rate : rates) {
                        const double r_g = std::sqrt(1.0 - rate.p_g);
                        const double r_e = std::sqrt(1.0 - rate.p_e);

                        auto state = make_active_state(/*peak_rank=*/4, k, seed++);
                        set_frame_bit(state.p_x, v, px != 0);
                        set_frame_bit(state.p_z, v, pz != 0);
                        const Amps before = physical(state);

                        const InstrumentPopulations pops =
                            exec_instrument_damp_eval(state, v, r_g, r_e);

                        REQUIRE_THAT(pops.pop_g,
                                     WithinAbs(ref_population(before, v, px != 0, 0), kTol));
                        REQUIRE_THAT(pops.pop_e,
                                     WithinAbs(ref_population(before, v, px != 0, 1), kTol));
                        REQUIRE_THAT(pops.pop_g + pops.pop_e, WithinAbs(norm2(before), kTol));

                        Amps want = before;
                        ref_damp(want, v, px != 0, r_g, r_e);
                        require_amps_match(physical(state), want);
                    }
                }
            }
        }
    }
}

TEST_CASE("instrument damp_eval: unit coefficients evaluate without touching the array") {
    auto state = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/7);
    Amps before(state.v(), state.v() + state.v_size());
    const auto gamma_before = state.gamma();

    const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/1, 1.0, 1.0);

    REQUIRE_THAT(pops.pop_g + pops.pop_e, WithinAbs(1.0, kTol));
    REQUIRE(state.gamma() == gamma_before);
    // Multiplication by exactly 1.0 must be bit-preserving.
    for (uint64_t i = 0; i < state.v_size(); ++i) {
        REQUIRE(state.v()[i] == before[i]);
    }
}

TEST_CASE("instrument damp_eval + no-fire renormalization matches the normalized K_stay state") {
    const double p_g = 0.2, p_e = 0.5;
    const double r_g = std::sqrt(1.0 - p_g), r_e = std::sqrt(1.0 - p_e);

    auto state = make_active_state(/*peak_rank=*/4, /*k=*/3, /*seed=*/23);
    const Amps before = physical(state);

    const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/1, r_g, r_e);
    const double total = pops.pop_g + pops.pop_e;
    const double no_fire = r_g * r_g * pops.pop_g + r_e * r_e * pops.pop_e;
    state.scale_magnitude(std::sqrt(total / no_fire));

    Amps want = before;
    ref_damp(want, /*v=*/1, /*px=*/false, r_g, r_e);
    ref_rescale_to(want, total);
    require_amps_match(physical(state), want);
}

// collapse_active: in-place forced projection

TEST_CASE("instrument fire path: collapse after damp recovers the pre-damp projection") {
    // The fused pass damps before the branch is drawn. On fire the
    // pre-applied r_source is a scalar on the surviving half, so the
    // collapse renormalization must reproduce the projection of the
    // *pre-damp* state exactly.
    const double r_g = std::sqrt(1.0 - 0.3), r_e = std::sqrt(1.0 - 0.6);

    for (int px = 0; px <= 1; ++px) {
        for (uint8_t source = 0; source <= 1; ++source) {
            auto state = make_active_state(/*peak_rank=*/4, /*k=*/3, /*seed=*/31 + px);
            set_frame_bit(state.p_x, /*v=*/2, px != 0);
            const Amps before = physical(state);
            const uint32_t k_before = state.active_k;

            const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/2, r_g, r_e);
            const double total = pops.pop_g + pops.pop_e;
            exec_instrument_collapse_active(state, /*v=*/2, source, total);

            Amps want = before;
            ref_project(want, /*v=*/2, px != 0, source);
            ref_rescale_to(want, total);
            require_amps_match(physical(state), want);

            // Layout and frame are untouched; the discarded half is
            // exactly zero.
            REQUIRE(state.active_k == k_before);
            REQUIRE(get_frame_bit(state.p_x, 2) == (px != 0));
            for (uint64_t i = 0; i < state.v_size(); ++i) {
                if (level_of(i, /*v=*/2, px != 0) != source) {
                    REQUIRE(state.v()[i] == std::complex<double>{0.0, 0.0});
                }
            }
        }
    }
}

TEST_CASE("instrument collapse_active: collapsing an already-definite level is the identity") {
    // Each site draws from its own evaluation, so the second collapse's
    // target is the raw norm measured at the *second* site, not a stale
    // total from the first.
    auto state = make_active_state(/*peak_rank=*/4, /*k=*/2, /*seed=*/43);
    const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/0, 0.9, 0.8);

    exec_instrument_collapse_active(state, /*v=*/0, /*source=*/1, pops.pop_g + pops.pop_e);
    const Amps once = physical(state);

    const InstrumentPopulations again = exec_instrument_damp_eval(state, /*v=*/0, 1.0, 1.0);
    REQUIRE_THAT(again.pop_e, WithinAbs(again.pop_g + again.pop_e, kTol));  // definite level
    exec_instrument_collapse_active(state, /*v=*/0, /*source=*/1, again.pop_g + again.pop_e);
    require_amps_match(physical(state), once);
}

// fire_branch: the pure draw helper

TEST_CASE("instrument fire_branch: regions partition the variate by exact branch probability") {
    const InstrumentPopulations pops{0.3, 0.7};
    const double p_g = 0.5, p_e = 0.1;
    // Region boundaries: fire-g on [0, 0.15), fire-e on [0.15, 0.22).
    const double w_g = p_g * pops.pop_g, w_e = p_e * pops.pop_e;
    REQUIRE_THAT(w_g, WithinAbs(0.15, kTol));
    REQUIRE_THAT(w_g + w_e, WithinAbs(0.22, kTol));

    auto branch = instrument_fire_branch(pops, p_g, p_e, 0.0);
    REQUIRE((branch.fired && branch.source == 0));
    branch = instrument_fire_branch(pops, p_g, p_e, 0.149999);
    REQUIRE((branch.fired && branch.source == 0));
    branch = instrument_fire_branch(pops, p_g, p_e, 0.150001);
    REQUIRE((branch.fired && branch.source == 1));
    branch = instrument_fire_branch(pops, p_g, p_e, 0.219999);
    REQUIRE((branch.fired && branch.source == 1));
    branch = instrument_fire_branch(pops, p_g, p_e, 0.220001);
    REQUIRE(!branch.fired);
    branch = instrument_fire_branch(pops, p_g, p_e, 0.999999);
    REQUIRE(!branch.fired);
}

TEST_CASE("instrument fire_branch: a dust population is never selected as the source") {
    // pop_e is floating-point dust: even a variate inside what would be
    // its region must not select it (the collapse it would trigger has no
    // ray to renormalize).
    const InstrumentPopulations pops{1.0, 1e-25};
    auto branch = instrument_fire_branch(pops, /*p_g=*/0.0, /*p_e=*/1.0, /*u=*/1e-26);
    REQUIRE(!branch.fired);
}

// expand_damp: dormant-random site under damping="exact"

TEST_CASE("instrument expand_damp: matches plain expansion followed by the damp") {
    const double r_g = std::sqrt(1.0 - 0.4), r_e = std::sqrt(1.0 - 0.1);

    for (int px = 0; px <= 1; ++px) {
        auto state = make_active_state(/*peak_rank=*/3, /*k=*/1, /*seed=*/57);
        const uint16_t v = 1;  // next dormant axis
        set_frame_bit(state.p_x, v, px != 0);
        const Amps before = physical(state);

        exec_instrument_expand_damp(state, v, r_g, r_e);
        REQUIRE(state.active_k == 2);

        // Dense reference: |phi> (x) |+> on the new axis, then the damp.
        Amps want(before.size() * 2);
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        for (uint64_t i = 0; i < before.size(); ++i) {
            want[i] = before[i] * inv_sqrt2;
            want[i + before.size()] = before[i] * inv_sqrt2;
        }
        ref_damp(want, v, px != 0, r_g, r_e);
        require_amps_match(physical(state), want);
    }
}

TEST_CASE("instrument expand_damp: unit coefficients reproduce the plain expansion") {
    auto state = make_active_state(/*peak_rank=*/3, /*k=*/1, /*seed=*/61);
    const Amps before = physical(state);

    exec_instrument_expand_damp(state, /*v=*/1, 1.0, 1.0);

    REQUIRE(state.active_k == 2);
    const Amps after = physical(state);
    REQUIRE_THAT(norm2(after), WithinAbs(norm2(before), kTol));
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (uint64_t i = 0; i < before.size(); ++i) {
        REQUIRE_THAT(std::abs(after[i] - before[i] * inv_sqrt2), WithinAbs(0.0, kTol));
        REQUIRE_THAT(std::abs(after[i + before.size()] - before[i] * inv_sqrt2),
                     WithinAbs(0.0, kTol));
    }
}

// Cross-checks against production machinery and mid-shot state shapes

TEST_CASE(
    "instrument damp_eval: populations match the forced measurement kernel's "
    "branch probabilities") {
    // The dense reference in this file shares its author's reading of the
    // frame convention; the forced measurement kernel does not -- it is
    // the production machinery behind exact record probabilities. An
    // eval-only pass and a forced Z-measurement on the same axis must
    // assign the same probability to each level.
    for (int px = 0; px <= 1; ++px) {
        for (uint8_t outcome = 0; outcome <= 1; ++outcome) {
            auto state = make_active_state(/*peak_rank=*/3, /*k=*/3, /*seed=*/97 + px);
            const uint16_t v = 2;  // the forced measurement requires the top axis
            set_frame_bit(state.p_x, v, px != 0);

            const InstrumentPopulations pops = exec_instrument_damp_eval(state, v, 1.0, 1.0);
            const double total = pops.pop_g + pops.pop_e;
            const double want = (outcome == 0 ? pops.pop_g : pops.pop_e) / total;

            const std::vector<uint8_t> record{outcome};
            state.forced_record = record;
            REQUIRE(exec_meas_active_diagonal_forced(state, v, /*classical_idx=*/0,
                                                     /*sign=*/false));
            REQUIRE_THAT(std::exp(state.forced_log_probability), WithinAbs(want, kTol));
        }
    }
}

TEST_CASE("instrument kernels: fused pass and collapse above the OpenMP rank threshold") {
    // The fused pass mutates the array inside parallel_reduce -- a pattern
    // the pre-existing kernels do not use -- so exercise it and the
    // collapse at the rank where the threaded path activates, on an
    // interior (strided) axis. Comparisons aggregate to keep the
    // assertion count independent of the array size.
    const uint32_t k = kMinRankForThreads;
    const double r_g = std::sqrt(1.0 - 0.3), r_e = std::sqrt(1.0 - 0.05);
    const uint16_t v = 7;

    auto state = make_active_state(/*peak_rank=*/k, k, /*seed=*/113);
    set_frame_bit(state.p_x, v, true);
    const Amps before = physical(state);

    const InstrumentPopulations pops = exec_instrument_damp_eval(state, v, r_g, r_e);
    REQUIRE_THAT(pops.pop_g, WithinAbs(ref_population(before, v, true, 0), 1e-9));
    REQUIRE_THAT(pops.pop_e, WithinAbs(ref_population(before, v, true, 1), 1e-9));

    Amps want = before;
    ref_damp(want, v, true, r_g, r_e);
    Amps got = physical(state);
    double max_diff = 0.0;
    for (uint64_t i = 0; i < got.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(got[i] - want[i]));
    }
    REQUIRE_THAT(max_diff, WithinAbs(0.0, 1e-10));

    exec_instrument_collapse_active(state, v, /*source=*/1, pops.pop_g + pops.pop_e);
    want = before;
    ref_project(want, v, true, 1);
    ref_rescale_to(want, pops.pop_g + pops.pop_e);
    got = physical(state);
    max_diff = 0.0;
    for (uint64_t i = 0; i < got.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(got[i] - want[i]));
    }
    REQUIRE_THAT(max_diff, WithinAbs(0.0, 1e-10));
}

TEST_CASE("instrument kernels: unnormalized array and nonunit gamma") {
    // Mid-shot states carry deferred normalization: the raw array norm
    // and gamma both sit away from one. Populations are raw-array
    // quantities, and both branch post-states must come out right through
    // the gamma-carried rescales.
    const double r_g = std::sqrt(1.0 - 0.25), r_e = std::sqrt(1.0 - 0.6);
    const double array_scale = 0.35;
    const std::complex<double> gamma{1.6, -0.9};

    auto make_scaled = [&] {
        auto state = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/131);
        for (uint64_t i = 0; i < state.v_size(); ++i) {
            state.v()[i] *= array_scale;
        }
        state.set_gamma(gamma);
        return state;
    };

    {  // No-fire branch: normalized K_stay at unchanged physical norm.
        auto state = make_scaled();
        const Amps before = physical(state);

        const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/0, r_g, r_e);
        // Populations are raw array quantities, not gamma-scaled ones.
        REQUIRE_THAT(pops.pop_g + pops.pop_e, WithinAbs(array_scale * array_scale, kTol));

        const double total = pops.pop_g + pops.pop_e;
        const double no_fire = r_g * r_g * pops.pop_g + r_e * r_e * pops.pop_e;
        state.scale_magnitude(std::sqrt(total / no_fire));

        Amps want = before;
        ref_damp(want, /*v=*/0, false, r_g, r_e);
        ref_rescale_to(want, norm2(before));
        require_amps_match(physical(state), want);
    }

    {  // Fire branch: pre-damp projection at unchanged physical norm.
        auto state = make_scaled();
        const Amps before = physical(state);

        const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/0, r_g, r_e);
        exec_instrument_collapse_active(state, /*v=*/0, /*source=*/0, pops.pop_g + pops.pop_e);

        Amps want = before;
        ref_project(want, /*v=*/0, false, 0);
        ref_rescale_to(want, norm2(before));
        require_amps_match(physical(state), want);
    }
}

// Physics checks

TEST_CASE("instrument damp_eval: frame conjugation swaps the coefficients and populations") {
    const double r_g = 0.6, r_e = 0.9;

    auto with_px = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/71);
    auto without = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/71);
    set_frame_bit(with_px.p_x, /*v=*/0, true);

    const auto pops_px = exec_instrument_damp_eval(with_px, /*v=*/0, r_g, r_e);
    const auto pops_swapped = exec_instrument_damp_eval(without, /*v=*/0, r_e, r_g);

    // Same array action either way; the population labels swap.
    require_amps_match(physical(with_px), physical(without));
    REQUIRE_THAT(pops_px.pop_g, WithinAbs(pops_swapped.pop_e, kTol));
    REQUIRE_THAT(pops_px.pop_e, WithinAbs(pops_swapped.pop_g, kTol));
}

TEST_CASE("instrument no-fire back-action: the FAQ interference case has its closed form") {
    // H; site; H on |0> with (p_g = p, p_e = 0): the no-fire damp tilts
    // |+> and the second H converts the tilt into a |1> population. The
    // joint probability of (no fire, measure 1) is exactly (1 - r_g)^2 / 4
    // -- approximately p^2/16 at small p -- the leading-order interference
    // error a collapse-at-site approximation would inflate to O(1).
    const double p = 0.36;
    const double r_g = std::sqrt(1.0 - p);

    SchrodingerState state(/*peak_rank=*/2, /*num_measurements=*/1);
    state.active_k = 1;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state.v()[0] = {inv_sqrt2, 0.0};  // H|0> = |+>
    state.v()[1] = {inv_sqrt2, 0.0};

    const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/0, r_g, 1.0);
    REQUIRE_THAT(pops.pop_g, WithinAbs(0.5, kTol));
    REQUIRE_THAT(pops.pop_e, WithinAbs(0.5, kTol));

    const double total = pops.pop_g + pops.pop_e;
    const double p_fire = p * pops.pop_g / total;
    REQUIRE_THAT(p_fire, WithinAbs(p / 2.0, kTol));

    const double no_fire = r_g * r_g * pops.pop_g + 1.0 * pops.pop_e;
    state.scale_magnitude(std::sqrt(total / no_fire));

    Amps amps = physical(state);
    ref_hadamard(amps, /*v=*/0);
    const double p_one_given_no_fire = std::norm(amps[1]) / norm2(amps);
    const double joint = (1.0 - p_fire) * p_one_given_no_fire;
    REQUIRE_THAT(joint, WithinAbs((1.0 - r_g) * (1.0 - r_g) / 4.0, kTol));
}

TEST_CASE("instrument p = 1 lowering recipe: eval-only plus per-branch collapse is exact") {
    // A certain-fire source (p_g = 1, r_g = 0) cannot use the fused form:
    // the damp would destroy the ray a fire must renormalize. The recipe
    // is an eval-only pass, then a collapse on *every* branch -- on
    // no-fire the posterior excludes the certain-fire source.
    const double p_g = 1.0, p_e = 0.19;

    auto reference_state = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/83);
    const Amps before = physical(reference_state);
    const double pop_g = ref_population(before, /*v=*/1, false, 0);
    const double pop_e = ref_population(before, /*v=*/1, false, 1);
    const double total = pop_g + pop_e;

    struct Branch {
        double u;
        bool want_fired;
        uint8_t collapse_to;
    };
    const Branch branches[] = {
        {0.5 * pop_g / total, true, 0},                    // fire from g
        {(pop_g + 0.5 * p_e * pop_e) / total, true, 1},    // fire from e
        {(pop_g + p_e * pop_e) / total + 1e-6, false, 1},  // no fire: source must be e
    };

    for (const Branch& b : branches) {
        auto state = make_active_state(/*peak_rank=*/3, /*k=*/2, /*seed=*/83);
        const InstrumentPopulations pops = exec_instrument_damp_eval(state, /*v=*/1, 1.0, 1.0);
        const InstrumentBranch drawn = instrument_fire_branch(pops, p_g, p_e, b.u);
        REQUIRE(drawn.fired == b.want_fired);
        if (drawn.fired) {
            REQUIRE(drawn.source == b.collapse_to);
        }

        exec_instrument_collapse_active(state, /*v=*/1, b.collapse_to, pops.pop_g + pops.pop_e);
        Amps want = before;
        ref_project(want, /*v=*/1, false, b.collapse_to);
        ref_rescale_to(want, total);
        require_amps_match(physical(state), want);
    }
}
