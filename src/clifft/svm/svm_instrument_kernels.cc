// Instrument kernel implementations. See svm_instrument_kernels.h for the
// API contract.

#include "clifft/svm/svm_instrument_kernels.h"

#include "clifft/svm/svm_internal.h"
#include "clifft/svm/svm_math.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>

namespace clifft {

namespace {

// Index of the ii-th amplitude pair's bit_v = 0 element: the k-1 loop bits
// of ii spread around an inserted zero at position v (the portable twin of
// the per-ISA kernels' scatter_bits_1).
inline uint64_t pair_index(uint64_t ii, uint16_t v) {
    const uint64_t low_mask = (1ULL << v) - 1;
    return (ii & low_mask) | ((ii & ~low_mask) << 1);
}

}  // namespace

InstrumentPopulations exec_instrument_damp_eval(SchrodingerState& state, uint16_t v, double r_g,
                                                double r_e) {
    assert(v < state.active_k && v < 64 && "instrument damp_eval: axis must be active");
    assert(r_g > 0.0 && r_g <= 1.0 && r_e > 0.0 && r_e <= 1.0 &&
           "instrument damp_eval: coefficients must be in (0, 1]; a p = 1 site lowers as "
           "eval-only plus per-branch collapse");

    const bool px_v = bit_get(state.p_x, v);
    // Frame conjugation: X^px diag(r_g, r_e) X^px swaps the coefficients;
    // Z^pz commutes with a real diagonal.
    const double c0 = px_v ? r_e : r_g;
    const double c1 = px_v ? r_g : r_e;

    const uint64_t v_bit = 1ULL << v;
    const uint64_t pairs = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();

    double pop0 = 0.0;
    double pop1 = 0.0;
    parallel_reduce(static_cast<int64_t>(pairs), state.active_k, pop0, pop1,
                    [&](int64_t ii, double& acc0, double& acc1) {
                        const uint64_t i0 = pair_index(static_cast<uint64_t>(ii), v);
                        const uint64_t i1 = i0 | v_bit;
                        acc0 += std::norm(arr[i0]);
                        acc1 += std::norm(arr[i1]);
                        arr[i0] *= c0;
                        arr[i1] *= c1;
                    });

    return InstrumentPopulations{px_v ? pop1 : pop0, px_v ? pop0 : pop1};
}

void exec_instrument_collapse_active(SchrodingerState& state, uint16_t v, uint8_t source,
                                     double target_norm2) {
    assert(v < state.active_k && v < 64 && "instrument collapse: axis must be active");
    assert(target_norm2 > 0.0 && "instrument collapse: target norm must be positive");

    const bool px_v = bit_get(state.p_x, v);
    // Frame conjugation maps the physical projector |s><s| onto array half
    // s XOR p_x[v]; a projector is invariant under Z^pz, so unlike the
    // frame-re-anchoring measurement kernels there is no phase to extract.
    const uint8_t b = source ^ static_cast<uint8_t>(px_v);

    const uint64_t v_bit = 1ULL << v;
    const uint64_t pairs = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();

    double kept = 0.0;
    double discarded = 0.0;
    parallel_reduce(static_cast<int64_t>(pairs), state.active_k, kept, discarded,
                    [&](int64_t ii, double& k_acc, double& d_acc) {
                        const uint64_t i0 = pair_index(static_cast<uint64_t>(ii), v);
                        const uint64_t i1 = i0 | v_bit;
                        const uint64_t keep = (b == 0) ? i0 : i1;
                        const uint64_t drop = (b == 0) ? i1 : i0;
                        k_acc += std::norm(arr[keep]);
                        d_acc += std::norm(arr[drop]);
                        arr[drop] = {0.0, 0.0};
                    });
    (void)discarded;
    assert(kept > kDustEpsilon * (kept + discarded) &&
           "instrument collapse onto a zero-probability level");

    state.scale_magnitude(std::sqrt(target_norm2 / kept));
}

void exec_instrument_expand_damp(SchrodingerState& state, uint16_t v, double r_g, double r_e) {
    assert(v == state.active_k && "instrument expand_damp must target the next dormant axis");
    assert(state.v_size() <= state.array_size() / 2 &&
           "instrument expand_damp exceeded the compiled peak_rank allocation");
    assert(r_g > 0.0 && r_g <= 1.0 && r_e > 0.0 && r_e <= 1.0 &&
           "instrument expand_damp: coefficients must be in (0, 1]; a p = 1 site lowers as "
           "plain expansion plus per-branch collapse");

    const bool px_v = bit_get(state.p_x, v);
    const double c_lo = px_v ? r_e : r_g;
    const double c_hi = px_v ? r_g : r_e;

    const uint64_t half = 1ULL << state.active_k;
    auto* __restrict arr = state.v();

    parallel_for(static_cast<int64_t>(half), state.active_k, [&](int64_t ii) {
        const uint64_t i = static_cast<uint64_t>(ii);
        arr[i + half] = arr[i] * c_hi;
        arr[i] *= c_lo;
    });

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));
}

void exec_instrument_collapse_dormant(SchrodingerState& state, uint16_t v, uint8_t level) {
    // Phase extraction and frame anchor, mirroring the dormant-random
    // measurement kernel with the outcome forced and no record written.
    if (bit_get(state.p_x, v) && level) {
        state.multiply_phase({-1.0, 0.0});
    }
    bit_set(state.p_x, v, level != 0);
    bit_set(state.p_z, v, false);
}

InstrumentBranch instrument_fire_branch(InstrumentPopulations pops, double p_g, double p_e,
                                        double u) {
    assert(u >= 0.0 && u < 1.0 && "instrument fire draw: variate out of [0, 1)");
    assert(p_g >= 0.0 && p_g <= 1.0 && p_e >= 0.0 && p_e <= 1.0 &&
           "instrument fire draw: probabilities out of [0, 1]");
    const double total = pops.pop_g + pops.pop_e;
    assert(total > 0.0 && "instrument fire draw on zero-norm populations");

    const double eps = kDustEpsilon * total;
    const double w_g = (pops.pop_g <= eps) ? 0.0 : p_g * pops.pop_g;
    const double w_e = (pops.pop_e <= eps) ? 0.0 : p_e * pops.pop_e;

    InstrumentBranch branch;
    const double scaled = u * total;
    if (scaled < w_g) {
        branch.fired = true;
        branch.source = 0;
    } else if (scaled < w_g + w_e) {
        branch.fired = true;
        branch.source = 1;
    }
    return branch;
}

}  // namespace clifft
