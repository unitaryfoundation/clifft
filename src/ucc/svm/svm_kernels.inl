// svm_kernels.inl -- VM execution kernels (textually included)
//
// This file contains all exec_* opcode handlers and the execute_internal()
// dispatch loop. It is #include'd into per-ISA translation units
// (svm_scalar.cc, svm_avx2.cc) that define UCC_SIMD_NAMESPACE before
// including this file.
//
// Required before this file:
//   #define UCC_SIMD_NAMESPACE scalar  (or avx2, etc.)
//   #include "ucc/svm/svm.h"
//   #include "ucc/svm/svm_internal.h"
//   #include "ucc/svm/svm_math.h"
//   #include "ucc/util/constants.h"
//   #include <immintrin.h>  (on x86)

#include <bit>
#include <cassert>
#include <cmath>
#include <numbers>
#include <utility>

namespace ucc {
namespace UCC_SIMD_NAMESPACE {

namespace {

#if defined(__AVX2__)
// Multiplies a vector of two packed complex numbers (V) by two complex
// scalars whose real and imaginary parts are pre-broadcast:
//   V      = [Re0, Im0, Re1, Im1]
//   S_re   = [c0_re, c0_re, c1_re, c1_re]
//   S_im   = [c0_im, c0_im, c1_im, c1_im]
// Returns  [c0*z0, c1*z1] via the addsub trick:
//   _mm256_addsub_pd subtracts on even lanes (real) and adds on odd (imag).
static inline __m256d cmul_m256d(__m256d V, __m256d S_re, __m256d S_im) {
    __m256d V_swap = _mm256_permute_pd(V, 0x5);
#if defined(__FMA__)
    return _mm256_fmaddsub_pd(V, S_re, _mm256_mul_pd(V_swap, S_im));
#else
    return _mm256_addsub_pd(_mm256_mul_pd(V, S_re), _mm256_mul_pd(V_swap, S_im));
#endif
}
#endif

// =============================================================================
// Frame Opcode Handlers (Zero-cost: update P only, no array touch)
// =============================================================================

// CNOT on virtual axes (c=control, t=target): conjugates the Pauli frame.
// Heisenberg rules: X_c spreads to X_t (p_x[t] ^= p_x[c]),
// Z_t spreads back to Z_c (p_z[c] ^= p_z[t]). No phase.
static inline void exec_frame_cnot(SchrodingerState& state, uint16_t c, uint16_t t) {
    bool px_c = bit_get(state.p_x, c);
    bool pz_t = bit_get(state.p_z, t);
    bit_xor(state.p_x, t, px_c);
    bit_xor(state.p_z, c, pz_t);
}

// CZ on virtual axes (c, t): conjugates the Pauli frame.
// Heisenberg rules: X_c picks up Z_t (p_z[t] ^= p_x[c]) and vice versa.
// When both X bits are set, CZ anticommutes: gamma *= -1.
static inline void exec_frame_cz(SchrodingerState& state, uint16_t c, uint16_t t) {
    bool px_c = bit_get(state.p_x, c);
    bool px_t = bit_get(state.p_x, t);
    if (px_c && px_t) {
        state.multiply_phase({-1.0, 0.0});
    }
    bit_xor(state.p_z, t, px_c);
    bit_xor(state.p_z, c, px_t);
}

// Hadamard on virtual axis v: conjugates X <-> Z in the Pauli frame.
// Swaps p_x[v] and p_z[v]. If both bits were set (Y Pauli), H*Y*H = -Y,
// so gamma *= -1.
static inline void exec_frame_h(SchrodingerState& state, uint16_t v) {
    bool px = bit_get(state.p_x, v);
    bool pz = bit_get(state.p_z, v);
    if (px && pz) {
        state.multiply_phase({-1.0, 0.0});
    }
    bit_set(state.p_x, v, pz);
    bit_set(state.p_z, v, px);
}

// S gate on virtual axis v: conjugates X -> Y in the Pauli frame.
// S*X*S_dag = Y = iXZ, so when p_x[v] is set: gamma *= i, p_z[v] ^= 1.
// Z commutes with S, so no change when only p_z is set.
static inline void exec_frame_s(SchrodingerState& state, uint16_t v) {
    bool px = bit_get(state.p_x, v);
    if (px) {
        state.multiply_phase(kI);
    }
    bit_xor(state.p_z, v, px);
}

// SWAP on virtual axes (a, b): exchanges Pauli frame bits for both axes.
// Swaps p_x[a] <-> p_x[b] and p_z[a] <-> p_z[b]. No phase.
static inline void exec_frame_swap(SchrodingerState& state, uint16_t a, uint16_t b) {
    bit_swap(state.p_x, a, state.p_x, b);
    bit_swap(state.p_z, a, state.p_z, b);
}

// =============================================================================
// Array Opcode Handlers (update P AND loop over v[])
// =============================================================================

// CNOT on active axes (c=control, t=target): permutes the amplitude array
// using branchless bit-weaving to iterate only over the 2^{k-2} relevant pairs.
static inline void exec_array_cnot(SchrodingerState& state, uint16_t c, uint16_t t) {
    assert(c != t && "Control and Target axes must be distinct");
    assert(c < state.active_k && c < 64 && "ARRAY_CNOT: control axis out of range");
    assert(t < state.active_k && t < 64 && "ARRAY_CNOT: target axis out of range");

    uint64_t iters = 1ULL << (state.active_k - 2);
    uint64_t c_bit = 1ULL << c;
    uint64_t t_bit = 1ULL << t;
    uint64_t pdep_mask = ~(c_bit | t_bit);
    auto* __restrict v = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base0 = scatter_bits_2(i, pdep_mask, c, t) | c_bit;
        std::swap(v[base0], v[base0 | t_bit]);
    }

    exec_frame_cnot(state, c, t);
}

// CZ on active axes (c, t): applies diag(1,1,1,-1) in the computational basis.
// Uses branchless bit-weaving to iterate only over the 2^{k-2} indices
// where both bit c and bit t are set.
static inline void exec_array_cz(SchrodingerState& state, uint16_t c, uint16_t t) {
    assert(c != t && "Control and Target axes must be distinct");
    assert(c < state.active_k && c < 64 && "ARRAY_CZ: control axis out of range");
    assert(t < state.active_k && t < 64 && "ARRAY_CZ: target axis out of range");

    uint64_t iters = 1ULL << (state.active_k - 2);
    uint64_t both_bits = (1ULL << c) | (1ULL << t);
    uint64_t pdep_mask = ~both_bits;
    auto* __restrict v = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_2(i, pdep_mask, c, t) | both_bits;
        v[idx] = -v[idx];
    }

    exec_frame_cz(state, c, t);
}

// SWAP on active axes (a, b): permutes the amplitude array using branchless
// bit-weaving to iterate only over the 2^{k-2} pairs that need swapping.
static inline void exec_array_swap(SchrodingerState& state, uint16_t a, uint16_t b) {
    assert(a != b && "ARRAY_SWAP: axes a and b must be distinct");
    assert(a < state.active_k && a < 64 && "ARRAY_SWAP: axis a out of range");
    assert(b < state.active_k && b < 64 && "ARRAY_SWAP: axis b out of range");

    uint64_t iters = 1ULL << (state.active_k - 2);
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    uint64_t pdep_mask = ~(a_bit | b_bit);
    auto* __restrict v = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, pdep_mask, a, b);
        std::swap(v[base | a_bit], v[base | b_bit]);
    }

    exec_frame_swap(state, a, b);
}

// MULTI_CNOT: fused star-graph of CNOTs sharing a target axis.
// Equivalent to CNOT(c1,t), CNOT(c2,t), ..., CNOT(cW,t) but in one pass.
// The combined unitary flips the target bit when the parity of control bits
// is odd: |x> -> |x XOR (popcount(x & ctrl_mask) % 2) << target>.
static inline void exec_array_multi_cnot(SchrodingerState& state, uint16_t target,
                                         uint64_t ctrl_mask) {
    assert(target < state.active_k && target < 64);

    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (state.active_k - 1);
    uint64_t pdep_mask = ~t_bit;
    uint64_t cm = ctrl_mask;
    auto* __restrict v = state.v();

    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, pdep_mask, target);
        if (std::popcount(actual & cm) & 1) {
            std::swap(v[actual], v[actual | t_bit]);
        }
    }

    // Frame updates: each individual CNOT(c, t) spreads X_c -> X_t, Z_t -> Z_c.
    // These all commute since they share the target.
    for (uint16_t c = 0; c < state.active_k; ++c) {
        if (ctrl_mask & (1ULL << c)) {
            exec_frame_cnot(state, c, target);
        }
    }
}

// MULTI_CZ: fused star-graph of CZs sharing a control axis.
// Equivalent to CZ(c,t1), CZ(c,t2), ..., CZ(c,tW) but in one pass.
// The combined diagonal negates v[idx] when the control bit is set AND
// the parity of target bits is odd.
static inline void exec_array_multi_cz(SchrodingerState& state, uint16_t control,
                                       uint64_t target_mask) {
    assert(control < state.active_k && control < 64);

    uint64_t c_bit = 1ULL << control;
    uint64_t size = 1ULL << state.active_k;
    uint64_t tm = target_mask;
    auto* __restrict v = state.v();

    for (uint64_t idx = 0; idx < size; ++idx) {
        if ((idx & c_bit) && (std::popcount(idx & tm) & 1)) {
            v[idx] = -v[idx];
        }
    }

    // Frame updates: each CZ(c, t) spreads X_c -> Z_t, X_t -> Z_c.
    for (uint16_t t = 0; t < state.active_k; ++t) {
        if (target_mask & (1ULL << t)) {
            exec_frame_cz(state, control, t);
        }
    }
}

// ARRAY_H on active axis v: applies the Hadamard butterfly transform,
// then updates the Pauli frame.
static inline void exec_array_h(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "ARRAY_H: axis out of range");
    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, pdep_mask, v);
        uint64_t idx1 = idx0 | v_bit;
        auto a = arr[idx0];
        auto b = arr[idx1];
        arr[idx0] = (a + b) * kInvSqrt2;
        arr[idx1] = (a - b) * kInvSqrt2;
    }

    exec_frame_h(state, v);
}

// S_dag gate on virtual axis v: conjugates X -> -Y in the Pauli frame.
// S_dag*X*S = -Y = -iXZ, so when p_x[v] is set: gamma *= -i, p_z[v] ^= 1.
// Z commutes with S_dag, so no change when only p_z is set.
static inline void exec_frame_s_dag(SchrodingerState& state, uint16_t v) {
    bool px = bit_get(state.p_x, v);
    if (px) {
        state.multiply_phase(kMinusI);
    }
    bit_xor(state.p_z, v, px);
}

// ARRAY_S on active axis v: applies diag(1, i) to the amplitude array,
// then updates the Pauli frame identically to FRAME_S.
static inline void exec_array_s(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "ARRAY_S: axis out of range");
    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kI;
    }
    exec_frame_s(state, v);
}

// ARRAY_S_DAG on active axis v: applies diag(1, -i) to the amplitude array,
// then updates the Pauli frame identically to FRAME_S_DAG.
static inline void exec_array_s_dag(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "ARRAY_S_DAG: axis out of range");
    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    for (uint64_t i = 0; i < iters; ++i) {
        arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kMinusI;
    }
    exec_frame_s_dag(state, v);
}

// =============================================================================
// Expansion & Phase Opcodes
// =============================================================================

// EXPAND axis v: promotes a dormant qubit to active by doubling the array.
// v must equal active_k (the next available axis). Copies v[i] to v[i + 2^k]
// for all i < 2^k, producing |phi> tensor |+> on the new axis.
// gamma /= sqrt(2) to maintain normalization.
static inline void exec_expand(SchrodingerState& state, uint16_t v) {
    assert(v == state.active_k && "EXPAND must target the next dormant axis");
    assert(state.v_size() <= state.array_size() / 2 && "EXPAND exceeded AOT peak_rank allocation!");
    uint64_t old_size = 1ULL << state.active_k;
    auto* __restrict arr = state.v();

    // Duplicate: v[i | (1 << k)] = v[i] for all i < 2^k
    uint64_t half = old_size;
    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i];
    }

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));
}

// T gate (pi/4 Z-rotation) on active axis v: applies diag(1, e^{i*pi/4})
// to array indices where bit v is set. If p_x[v]=1 (T anticommutes with X),
// the array gets T_dag instead and gamma absorbs e^{i*pi/4} to preserve
// the factored state identity.
static inline void exec_phase_t(SchrodingerState& state, uint16_t v) {
    assert(v < 64 && "PHASE_T: axis out of range");
    bool px = bit_get(state.p_x, v);

    if (v >= state.active_k) {
        if (px)
            state.multiply_phase(kExpIPiOver4);
        return;
    }

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    if (px) {
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kExpMinusIPiOver4;
        }
        state.multiply_phase(kExpIPiOver4);
    } else {
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kExpIPiOver4;
        }
    }
}

// T_dag gate (-pi/4 Z-rotation) on active axis v: applies diag(1, e^{-i*pi/4}).
// Mirror of T: if p_x[v]=1, the array gets T instead and gamma absorbs
// e^{-i*pi/4}.
static inline void exec_phase_t_dag(SchrodingerState& state, uint16_t v) {
    assert(v < 64 && "PHASE_T_DAG: axis out of range");
    bool px = bit_get(state.p_x, v);

    if (v >= state.active_k) {
        if (px)
            state.multiply_phase(kExpMinusIPiOver4);
        return;
    }

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    if (px) {
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kExpIPiOver4;
        }
        state.multiply_phase(kExpMinusIPiOver4);
    } else {
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= kExpMinusIPiOver4;
        }
    }
}

// Fused EXPAND + PHASE_T: duplicates the array into the upper half while
// applying the T phase to the new axis in a single pass.  The expand
// operation sets v[i + half] = v[i] for all i < 2^k, then T multiplies
// v[idx] by exp(i*pi/4) for all idx with the new bit set.  By fusing,
// we write v[i + half] = v[i] * phase in one loop instead of two.
static inline void exec_expand_t(SchrodingerState& state, uint16_t v) {
    assert(v == state.active_k && "EXPAND_T must target the next dormant axis");
    assert(state.v_size() <= state.array_size() / 2);
    uint64_t half = 1ULL << state.active_k;
    auto* __restrict arr = state.v();
    bool px = bit_get(state.p_x, v);

    // The phase applied to the upper half (bit v set).
    // If p_x[v]=1, T anticommutes with X -> array gets T_dag, gamma absorbs T.
    std::complex<double> phase = px ? kExpMinusIPiOver4 : kExpIPiOver4;

    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i] * phase;
    }

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));

    if (px) {
        state.multiply_phase(kExpIPiOver4);
    }
}

// Fused EXPAND + PHASE_T_DAG: same fusion but with T-dagger phase.
static inline void exec_expand_t_dag(SchrodingerState& state, uint16_t v) {
    assert(v == state.active_k && "EXPAND_T_DAG must target the next dormant axis");
    assert(state.v_size() <= state.array_size() / 2);
    uint64_t half = 1ULL << state.active_k;
    auto* __restrict arr = state.v();
    bool px = bit_get(state.p_x, v);

    // If p_x[v]=1, T_dag anticommutes -> array gets T, gamma absorbs T_dag.
    std::complex<double> phase = px ? kExpIPiOver4 : kExpMinusIPiOver4;

    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i] * phase;
    }

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));

    if (px) {
        state.multiply_phase(kExpMinusIPiOver4);
    }
}

// =============================================================================
// Continuous Rotation Opcodes
// =============================================================================

// Continuous Z-rotation on active axis v: applies diag(1, z) where
// z = weight_re + i*weight_im = e^{i*alpha*pi}. If p_x[v]=1 (X error),
// the array gets z* instead and gamma absorbs z to preserve the factored
// state identity.
static inline void exec_phase_rot(SchrodingerState& state, uint16_t v, double z_re, double z_im) {
    assert(v < 64 && "PHASE_ROT: axis out of range");
    bool px = bit_get(state.p_x, v);
    std::complex<double> z(z_re, z_im);

    if (v >= state.active_k) {
        if (px)
            state.multiply_phase(z);
        return;
    }

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << v;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    if (px) {
        std::complex<double> z_conj(z_re, -z_im);
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= z_conj;
        }
        state.multiply_phase(z);
    } else {
        for (uint64_t i = 0; i < iters; ++i) {
            arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= z;
        }
    }
}

// Fused EXPAND + PHASE_ROT: duplicates the array into the upper half while
// applying the continuous phase to the new axis in a single pass.
static inline void exec_expand_rot(SchrodingerState& state, uint16_t v, double z_re, double z_im) {
    assert(v == state.active_k && "EXPAND_ROT must target the next dormant axis");
    assert(state.v_size() <= state.array_size() / 2);
    uint64_t half = 1ULL << state.active_k;
    auto* __restrict arr = state.v();
    bool px = bit_get(state.p_x, v);

    std::complex<double> z(z_re, z_im);
    std::complex<double> phase = px ? std::complex<double>(z_re, -z_im) : z;

    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i] * phase;
    }

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));

    if (px) {
        state.multiply_phase(z);
    }
}

// =============================================================================
// Fused Single-Axis U2 Opcode
// =============================================================================

// OP_ARRAY_U2: applies a pre-computed 2x2 unitary matrix from the ConstantPool.
// The 4-state FSM selects the correct matrix based on the incoming Pauli frame
// bits (p_x, p_z) on the target axis. This replaces multiple sequential
// single-axis array passes with one butterfly sweep.
static inline void exec_array_u2(SchrodingerState& state, const ConstantPool& pool, uint16_t axis,
                                 uint32_t cp_idx) {
    assert(axis < 64 && "ARRAY_U2: axis out of range");
    const auto& node = pool.fused_u2_nodes[cp_idx];

    uint8_t in_state = static_cast<uint8_t>((bit_get(state.p_z, axis) ? 2 : 0) |
                                            (bit_get(state.p_x, axis) ? 1 : 0));

    const auto* mat = node.matrices[in_state];
    state.multiply_phase(node.gamma_multipliers[in_state]);

    uint8_t out = node.out_states[in_state];
    bit_set(state.p_x, axis, (out & 1) != 0);
    bit_set(state.p_z, axis, (out & 2) != 0);

    // Dormant fast-path: axis not in the active array, only frame + gamma matter.
    if (axis >= state.active_k)
        return;

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t v_bit = 1ULL << axis;
    uint64_t pdep_mask = ~v_bit;
    auto* __restrict arr = state.v();

    const std::complex<double> m00 = mat[0];
    const std::complex<double> m01 = mat[1];
    const std::complex<double> m10 = mat[2];
    const std::complex<double> m11 = mat[3];

#if defined(__AVX2__)
    auto* __restrict arr_dbl = reinterpret_cast<double*>(arr);

    if (axis == 0) {
        // axis==0: old0 and old1 are physically adjacent (idx0=i*2, idx1=i*2+1).
        // Load both into one 256-bit register, broadcast each to both lanes,
        // multiply by the matrix columns, and store back.
        __m256d m_col0_re = _mm256_setr_pd(m00.real(), m00.real(), m10.real(), m10.real());
        __m256d m_col0_im = _mm256_setr_pd(m00.imag(), m00.imag(), m10.imag(), m10.imag());
        __m256d m_col1_re = _mm256_setr_pd(m01.real(), m01.real(), m11.real(), m11.real());
        __m256d m_col1_im = _mm256_setr_pd(m01.imag(), m01.imag(), m11.imag(), m11.imag());

        for (uint64_t i = 0; i < iters; ++i) {
            uint64_t idx0 = i << 1;
            double* base = arr_dbl + (idx0 << 1);

            __m256d v_both = _mm256_load_pd(base);
            __m256d v_old0 = _mm256_permute2f128_pd(v_both, v_both, 0x00);
            __m256d v_old1 = _mm256_permute2f128_pd(v_both, v_both, 0x11);

            __m256d term0 = cmul_m256d(v_old0, m_col0_re, m_col0_im);
            __m256d term1 = cmul_m256d(v_old1, m_col1_re, m_col1_im);

            _mm256_store_pd(base, _mm256_add_pd(term0, term1));
        }
        return;
    } else {
        // axis>0: consecutive loop indices i, i+1 produce consecutive memory
        // addresses for idx0 (and likewise for idx1). Process two butterflies
        // per iteration using 256-bit loads that span two complex values.
        __m256d m00_re = _mm256_set1_pd(m00.real());
        __m256d m00_im = _mm256_set1_pd(m00.imag());
        __m256d m01_re = _mm256_set1_pd(m01.real());
        __m256d m01_im = _mm256_set1_pd(m01.imag());
        __m256d m10_re = _mm256_set1_pd(m10.real());
        __m256d m10_im = _mm256_set1_pd(m10.imag());
        __m256d m11_re = _mm256_set1_pd(m11.real());
        __m256d m11_im = _mm256_set1_pd(m11.imag());

        for (uint64_t i = 0; i < iters; i += 2) {
            uint64_t idx0 = scatter_bits_1(i, pdep_mask, axis);
            uint64_t idx1 = idx0 | v_bit;

            __m256d v_old0 = _mm256_load_pd(arr_dbl + (idx0 << 1));
            __m256d v_old1 = _mm256_load_pd(arr_dbl + (idx1 << 1));

            __m256d new0 = _mm256_add_pd(cmul_m256d(v_old0, m00_re, m00_im),
                                         cmul_m256d(v_old1, m01_re, m01_im));
            __m256d new1 = _mm256_add_pd(cmul_m256d(v_old0, m10_re, m10_im),
                                         cmul_m256d(v_old1, m11_re, m11_im));

            _mm256_store_pd(arr_dbl + (idx0 << 1), new0);
            _mm256_store_pd(arr_dbl + (idx1 << 1), new1);
        }
        return;
    }
#endif

    // Scalar fallback for non-AVX2 targets (e.g. Wasm, older x86).
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, pdep_mask, axis);
        uint64_t idx1 = idx0 | v_bit;

        auto old0 = arr[idx0];
        auto old1 = arr[idx1];

        arr[idx0] = m00 * old0 + m01 * old1;
        arr[idx1] = m10 * old0 + m11 * old1;
    }
}

// =============================================================================
// Measurement Opcodes
// =============================================================================

// Dormant-static measurement on axis v: the qubit is not in the amplitude
// array, and its Z-basis eigenvalue is deterministic. The physical outcome
// is simply p_x[v] (the X bit of the Pauli frame).
static inline void exec_meas_dormant_static(SchrodingerState& state, uint16_t v,
                                            uint32_t classical_idx, bool sign) {
    uint8_t outcome = bit_get(state.p_x, v) ? 1 : 0;
    outcome ^= static_cast<uint8_t>(sign);
    state.meas_record[classical_idx] = outcome;
}

// Dormant-random measurement on axis v: the qubit is dormant but its
// outcome is uniformly random (e.g. X-basis eigenstate measured in Z).
// Samples m in {0,1}, extracts phase (-1)^(p_x[v]*m), and resets the
// frame to anchor the post-measurement computational state.
static inline void exec_meas_dormant_random(SchrodingerState& state, uint16_t v,
                                            uint32_t classical_idx, bool sign) {
    uint8_t m_abs = state.random_double() < 0.5 ? 0 : 1;

    // Phase extraction: (-1)^(p_x[v] * m_abs)
    if (bit_get(state.p_x, v) && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }

    // Frame reset: anchor to abstract eigenstate
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    // Physical outcome includes the compression sign
    state.meas_record[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
}

// Active-diagonal measurement on axis v (must be active_k-1): the qubit is
// in the amplitude array and the measurement basis is diagonal (Z-like).
// Samples branch b from probabilities of the upper/lower halves of v[],
// compacts the array by discarding the unchosen half, and extracts phase
// (-1)^(p_z[v]*b). Physical outcome m = b XOR p_x[v].
static inline void exec_meas_active_diagonal(SchrodingerState& state, uint16_t v,
                                             uint32_t classical_idx, bool sign) {
    assert(v == state.active_k - 1 && "Active diagonal measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Compute probability of array branch b=0 (bit v = 0) and b=1 (bit v = 1)
    double prob_b0 = 0.0;
    double prob_b1 = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        prob_b0 += std::norm(arr[i]);         // bit v = 0
        prob_b1 += std::norm(arr[i + half]);  // bit v = 1
    }
    double total = prob_b0 + prob_b1;
    assert(total > 0.0 && "Active diagonal measurement on zero-norm state");

    uint8_t b = sample_branch(state, prob_b0, prob_b1, total);

    // Abstract outcome (determines array branch + frame state)
    uint8_t m_abs = b ^ static_cast<uint8_t>(px_v);
    // Physical outcome (classical record includes compression sign)
    uint8_t m_phys = m_abs ^ static_cast<uint8_t>(sign);

    // Phase extraction: (-1)^(p_z[v] * b) when b=1
    if (b == 1 && pz_v) {
        state.multiply_phase({-1.0, 0.0});
    }

    // Compact array: keep chosen branch
    if (b == 1) {
        for (uint64_t i = 0; i < half; ++i) {
            arr[i] = arr[i + half];
        }
    }

    // Decrement active_k before renormalization so scale_magnitude only
    // touches the surviving half, saving 50% of FLOPs.
    state.active_k--;

    double prob_b = (b == 0) ? prob_b0 : prob_b1;
    if (prob_b > 0.0) {
        state.scale_magnitude(std::sqrt(total / prob_b));
    }

    // Frame reset: anchor to abstract eigenstate
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = m_phys;
}

// Active-interfere measurement on axis v (must be active_k-1): the qubit is
// in the amplitude array and the measurement basis is off-diagonal (X-like).
// Computes |+> and |-> branch probabilities by summing |v[i] +/- v[i+half]|^2,
// folds the array (add or subtract), and normalizes. Physical outcome
// m = b_x XOR p_z[v], with phase extraction (-1)^(p_x[v]*m).
static inline void exec_meas_active_interfere(SchrodingerState& state, uint16_t v,
                                              uint32_t classical_idx, bool sign) {
    assert(v == state.active_k - 1 && "Active interfere measurement must target axis k-1");

    uint64_t half = 1ULL << (state.active_k - 1);
    auto* __restrict arr = state.v();
    bool px_v = bit_get(state.p_x, v);
    bool pz_v = bit_get(state.p_z, v);

    // Compute X-basis probabilities:
    // b_x=0 (|+> branch): sum |v[i] + v[i+half]|^2
    // b_x=1 (|-> branch): sum |v[i] - v[i+half]|^2
    double prob_plus = 0.0;
    double prob_minus = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        auto sum = arr[i] + arr[i + half];
        auto diff = arr[i] - arr[i + half];
        prob_plus += std::norm(sum);
        prob_minus += std::norm(diff);
    }
    double total = prob_plus + prob_minus;
    assert(total > 0.0 && "Active interfere measurement on zero-norm state");

    uint8_t b_x = sample_branch(state, prob_plus, prob_minus, total);

    // Abstract outcome (determines array fold + frame state)
    uint8_t m_abs = b_x ^ static_cast<uint8_t>(pz_v);
    // Physical outcome (classical record includes compression sign)
    uint8_t m_phys = m_abs ^ static_cast<uint8_t>(sign);

    // Fold array: v'[i] = (v[i] +/- v[i+half]) / sqrt(2)
    // b_x=0 -> add, b_x=1 -> subtract
    // The 1/sqrt(2) factor keeps the fold unitary, preventing exponential
    // magnitude growth from repeated EXPAND + INTERFERE sequences.
    // Branch hoisted outside the loop so the compiler can auto-vectorize.
    if (b_x == 0) {
        for (uint64_t i = 0; i < half; ++i) {
            arr[i] = (arr[i] + arr[i + half]) * kInvSqrt2;
        }
    } else {
        for (uint64_t i = 0; i < half; ++i) {
            arr[i] = (arr[i] - arr[i + half]) * kInvSqrt2;
        }
    }

    // Phase extraction: (-1)^(p_x[v] * m_abs)
    if (px_v && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }

    // Decrement active_k before renormalization so scale_magnitude only
    // touches the surviving half, saving 50% of FLOPs.
    state.active_k--;

    // Deferred normalization: compensate for probability of chosen branch.
    // With the unitary 1/sqrt(2) fold above, the surviving branch has
    // squared norm = prob_bx / 2, matching the diagonal measurement formula.
    double prob_bx = (b_x == 0) ? prob_plus : prob_minus;
    if (prob_bx > 0.0) {
        state.scale_magnitude(std::sqrt(total / prob_bx));
    }

    // Frame reset: anchor to abstract eigenstate
    bit_set(state.p_x, v, m_abs);
    bit_set(state.p_z, v, false);

    state.meas_record[classical_idx] = m_phys;
}

// Fused SWAP + MEAS_ACTIVE_INTERFERE: performs the logical swap and X-basis
// fold in a single O(2^k) memory pass, eliminating the redundant O(2^k)
// array permutation that a separate ARRAY_SWAP would require.
//
// The key insight is that we can map every output index `idx` directly to
// its unswapped source indices without physically permuting the array.
// For each output index in [0, 2^(k-1)):
//   - Extract bit f from idx to determine which "half" of the swap we are in
//   - Reconstruct the pre-swap base index by moving that bit to position t
//   - Read the pair (base, base|f_bit) and fold them with +/- as usual
//
// Memory safety of the in-place fold:
//   When b_f=0: sources are arr[idx] and arr[idx|f_bit] (both >= idx).
//     The arr[idx|f_bit] value is consumed here before that higher index
//     is reached; when the loop gets there, b_f=1 redirects reads to the
//     upper half, so the overwritten lower-half value is never re-read.
//   When b_f=1: sources are in the upper half (>= 2^t > idx), which is
//     strictly read-only during the fold pass.
static inline void exec_swap_meas_interfere(SchrodingerState& state, uint16_t f, uint16_t t,
                                            uint32_t classical_idx, bool sign) {
    assert(t == state.active_k - 1 && "Swap target must be k-1");

    if (f == t) {
        exec_meas_active_interfere(state, t, classical_idx, sign);
        return;
    }

    // Frame update: equivalent to FRAME_SWAP(f, t) before measurement
    exec_frame_swap(state, f, t);
    bool px_v = bit_get(state.p_x, t);
    bool pz_v = bit_get(state.p_z, t);

    uint64_t half = 1ULL << t;
    auto* __restrict arr = state.v();
    uint64_t f_bit = 1ULL << f;

    // Pass 1: Compute X-basis probabilities with swapped index mapping
    double prob_plus = 0.0;
    double prob_minus = 0.0;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> f) & 1;
        uint64_t base = (idx & ~f_bit) | (b_f << t);

        auto sum = arr[base] + arr[base | f_bit];
        auto diff = arr[base] - arr[base | f_bit];
        prob_plus += std::norm(sum);
        prob_minus += std::norm(diff);
    }

    double total = prob_plus + prob_minus;
    assert(total > 0.0 && "Active interfere measurement on zero-norm state");

    uint8_t b_x = sample_branch(state, prob_plus, prob_minus, total);

    // Pass 2: In-place fold with swapped index mapping
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

    // Decrement active_k before renormalization so scale_magnitude only
    // touches the surviving half, saving 50% of FLOPs.
    state.active_k--;

    double prob_bx = (b_x == 0) ? prob_plus : prob_minus;
    if (prob_bx > 0.0) {
        state.scale_magnitude(std::sqrt(total / prob_bx));
    }

    // Abstract and physical outcomes
    uint8_t m_abs = b_x ^ static_cast<uint8_t>(pz_v);
    uint8_t m_phys = m_abs ^ static_cast<uint8_t>(sign);

    if (px_v && m_abs) {
        state.multiply_phase({-1.0, 0.0});
    }

    // Frame reset: anchor to abstract eigenstate
    bit_set(state.p_x, t, m_abs);
    bit_set(state.p_z, t, false);

    state.meas_record[classical_idx] = m_phys;
}

// =============================================================================
// Classical / Error Opcodes
// =============================================================================

// Apply a Pauli error to the Pauli frame (shared logic for APPLY_PAULI and NOISE).
static inline void apply_pauli_to_frame(SchrodingerState& state, const PauliBitMask& err_x,
                                        const PauliBitMask& err_z, bool sign) {
    // Phase: (-1)^popcount(err_z & current_x)
    // When composing E*P, we commute Z^{e_z} past X^{p_x}, picking up (-1)^{e_z . p_x}.
    if ((state.p_x & err_z).popcount() & 1) {
        state.multiply_phase({-1.0, 0.0});
    }

    state.p_x ^= err_x;
    state.p_z ^= err_z;

    if (sign) {
        state.multiply_phase({-1.0, 0.0});
    }
}

// APPLY_PAULI: conditionally composes a Pauli error into the Pauli frame.
// Only applied if the controlling measurement recorded outcome 1.
static inline void exec_apply_pauli(SchrodingerState& state, const ConstantPool& pool,
                                    uint32_t cp_mask_idx, uint32_t condition_idx) {
    assert(cp_mask_idx < pool.pauli_masks.size());
    assert(condition_idx < state.meas_record.size());

    if (state.meas_record[condition_idx] == 0) {
        return;
    }

    const auto& pm = pool.pauli_masks[cp_mask_idx];
    apply_pauli_to_frame(state, pm.x, pm.z, pm.sign);
}

// NOISE: stochastic Pauli channel with gap-based skip optimization.
// If this site's index doesn't match the next expected noise event, it's
// guaranteed silent (identity) by the exponential gap sampling and we skip
// the RNG roll entirely.
static inline void exec_noise(SchrodingerState& state, const ConstantPool& pool,
                              uint32_t site_idx) {
    assert(site_idx < pool.noise_sites.size());

    if (site_idx != state.next_noise_idx)
        return;

    const auto& site = pool.noise_sites[site_idx];
    double prob_sum = 0.0;
    for (const auto& ch : site.channels) {
        prob_sum += ch.prob;
    }

    double rand = state.random_double() * prob_sum;
    double cumulative = 0.0;
    for (const auto& ch : site.channels) {
        cumulative += ch.prob;
        if (rand < cumulative) {
            apply_pauli_to_frame(state, ch.destab_mask, ch.stab_mask, false);
            break;
        }
    }

    state.next_noise_idx++;
    state.draw_next_noise(pool.noise_hazards);
}

// NOISE_BLOCK: processes a contiguous range of noise sites [start, start+count)
// in a tight loop. The gap-sampler's next_noise_idx determines which (if any)
// sites within the block actually fire. Most shots skip the entire block when
// next_noise_idx falls outside [start, start+count).
static inline void exec_noise_block(SchrodingerState& state, const ConstantPool& pool,
                                    uint32_t start_site, uint32_t count) {
    uint32_t end_site = start_site + count;
    while (state.next_noise_idx >= start_site && state.next_noise_idx < end_site) {
        exec_noise(state, pool, state.next_noise_idx);
    }
}

// READOUT_NOISE: classical bit-flip on a measurement result.
static inline void exec_readout_noise(SchrodingerState& state, const ConstantPool& pool,
                                      uint32_t entry_idx) {
    assert(entry_idx < pool.readout_noise.size());
    const auto& entry = pool.readout_noise[entry_idx];

    if (state.random_double() < entry.prob) {
        assert(entry.meas_idx < state.meas_record.size());
        state.meas_record[entry.meas_idx] ^= 1;
    }
}

// DETECTOR: computes the XOR parity of a list of measurement record entries.
// When expected_one is true, the parity is initialized to 1 so that the
// noiseless reference outcome (which would also be 1) normalizes to 0.
static inline void exec_detector(SchrodingerState& state, const ConstantPool& pool,
                                 uint32_t det_list_idx, uint32_t classical_idx, bool expected_one) {
    assert(det_list_idx < pool.detector_targets.size());
    const auto& targets = pool.detector_targets[det_list_idx];

    uint8_t parity = static_cast<uint8_t>(expected_one);
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }

    assert(classical_idx < state.det_record.size());
    state.det_record[classical_idx] = parity;
}

// POSTSELECT: computes XOR parity like DETECTOR, writes 0 to det_record,
// and sets discarded = true if parity != 0 (shot failed post-selection).
// When expected_one is true, the parity is initialized to 1 so that the
// noiseless reference outcome normalizes to 0 (no false discards).
// Returns true if the shot should be aborted.
static inline bool exec_postselect(SchrodingerState& state, const ConstantPool& pool,
                                   uint32_t det_list_idx, uint32_t classical_idx,
                                   bool expected_one) {
    assert(det_list_idx < pool.detector_targets.size());
    const auto& targets = pool.detector_targets[det_list_idx];

    uint8_t parity = static_cast<uint8_t>(expected_one);
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }

    assert(classical_idx < state.det_record.size());
    state.det_record[classical_idx] = 0;

    if (parity != 0) {
        state.discarded = true;
        return true;
    }
    return false;
}

// OBSERVABLE: computes XOR parity like DETECTOR, but writes to obs_record.
static inline void exec_observable(SchrodingerState& state, const ConstantPool& pool,
                                   uint32_t target_list_idx, uint32_t obs_idx) {
    assert(target_list_idx < pool.observable_targets.size());
    const auto& targets = pool.observable_targets[target_list_idx];

    uint8_t parity = 0;
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }

    assert(obs_idx < state.obs_record.size());
    state.obs_record[obs_idx] ^= parity;
}

}  // namespace

// =============================================================================
// SVM Execution
// =============================================================================

void execute_internal(const CompiledModule& program, SchrodingerState& state);

void execute_internal(const CompiledModule& program, SchrodingerState& state) {
    assert(program.peak_rank < 64 && "peak_rank >= 64 would cause UB in bit shifts");

    if (program.bytecode.empty()) {
        return;
    }

#if defined(__GNUC__) || defined(__clang__)
    // Threaded dispatch table (computed gotos) gives each opcode its own
    // indirect-branch history entry, dramatically improving prediction.
    // Sized to 256; designated initializers map enums directly to labels.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    static const void* dispatch_table[256] = {
        [static_cast<uint8_t>(Opcode::OP_FRAME_CNOT)] = &&L_OP_FRAME_CNOT,
        [static_cast<uint8_t>(Opcode::OP_FRAME_CZ)] = &&L_OP_FRAME_CZ,
        [static_cast<uint8_t>(Opcode::OP_FRAME_H)] = &&L_OP_FRAME_H,
        [static_cast<uint8_t>(Opcode::OP_FRAME_S)] = &&L_OP_FRAME_S,
        [static_cast<uint8_t>(Opcode::OP_FRAME_S_DAG)] = &&L_OP_FRAME_S_DAG,
        [static_cast<uint8_t>(Opcode::OP_FRAME_SWAP)] = &&L_OP_FRAME_SWAP,

        [static_cast<uint8_t>(Opcode::OP_ARRAY_CNOT)] = &&L_OP_ARRAY_CNOT,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_CZ)] = &&L_OP_ARRAY_CZ,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_SWAP)] = &&L_OP_ARRAY_SWAP,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_MULTI_CNOT)] = &&L_OP_ARRAY_MULTI_CNOT,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_MULTI_CZ)] = &&L_OP_ARRAY_MULTI_CZ,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_H)] = &&L_OP_ARRAY_H,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_S)] = &&L_OP_ARRAY_S,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_S_DAG)] = &&L_OP_ARRAY_S_DAG,

        [static_cast<uint8_t>(Opcode::OP_EXPAND)] = &&L_OP_EXPAND,
        [static_cast<uint8_t>(Opcode::OP_PHASE_T)] = &&L_OP_PHASE_T,
        [static_cast<uint8_t>(Opcode::OP_PHASE_T_DAG)] = &&L_OP_PHASE_T_DAG,
        [static_cast<uint8_t>(Opcode::OP_EXPAND_T)] = &&L_OP_EXPAND_T,
        [static_cast<uint8_t>(Opcode::OP_EXPAND_T_DAG)] = &&L_OP_EXPAND_T_DAG,
        [static_cast<uint8_t>(Opcode::OP_PHASE_ROT)] = &&L_OP_PHASE_ROT,
        [static_cast<uint8_t>(Opcode::OP_EXPAND_ROT)] = &&L_OP_EXPAND_ROT,
        [static_cast<uint8_t>(Opcode::OP_ARRAY_U2)] = &&L_OP_ARRAY_U2,

        [static_cast<uint8_t>(Opcode::OP_MEAS_DORMANT_STATIC)] = &&L_OP_MEAS_DORMANT_STATIC,
        [static_cast<uint8_t>(Opcode::OP_MEAS_DORMANT_RANDOM)] = &&L_OP_MEAS_DORMANT_RANDOM,
        [static_cast<uint8_t>(Opcode::OP_MEAS_ACTIVE_DIAGONAL)] = &&L_OP_MEAS_ACTIVE_DIAGONAL,
        [static_cast<uint8_t>(Opcode::OP_MEAS_ACTIVE_INTERFERE)] = &&L_OP_MEAS_ACTIVE_INTERFERE,
        [static_cast<uint8_t>(Opcode::OP_SWAP_MEAS_INTERFERE)] = &&L_OP_SWAP_MEAS_INTERFERE,

        [static_cast<uint8_t>(Opcode::OP_APPLY_PAULI)] = &&L_OP_APPLY_PAULI,
        [static_cast<uint8_t>(Opcode::OP_NOISE)] = &&L_OP_NOISE,
        [static_cast<uint8_t>(Opcode::OP_NOISE_BLOCK)] = &&L_OP_NOISE_BLOCK,
        [static_cast<uint8_t>(Opcode::OP_READOUT_NOISE)] = &&L_OP_READOUT_NOISE,
        [static_cast<uint8_t>(Opcode::OP_DETECTOR)] = &&L_OP_DETECTOR,
        [static_cast<uint8_t>(Opcode::OP_POSTSELECT)] = &&L_OP_POSTSELECT,
        [static_cast<uint8_t>(Opcode::OP_OBSERVABLE)] = &&L_OP_OBSERVABLE,
    };

    const Instruction* pc = program.bytecode.data();
    const Instruction* end = pc + program.bytecode.size();

#define DISPATCH()                                              \
    do {                                                        \
        if (++pc == end)                                        \
            return;                                             \
        goto* dispatch_table[static_cast<uint8_t>(pc->opcode)]; \
    } while (0)

    goto* dispatch_table[static_cast<uint8_t>(pc->opcode)];

L_OP_FRAME_CNOT:
    exec_frame_cnot(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_FRAME_CZ:
    exec_frame_cz(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_FRAME_H:
    exec_frame_h(state, pc->axis_1);
    DISPATCH();

L_OP_FRAME_S:
    exec_frame_s(state, pc->axis_1);
    DISPATCH();

L_OP_FRAME_S_DAG:
    exec_frame_s_dag(state, pc->axis_1);
    DISPATCH();

L_OP_FRAME_SWAP:
    exec_frame_swap(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_ARRAY_CNOT:
    exec_array_cnot(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_ARRAY_CZ:
    exec_array_cz(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_ARRAY_SWAP:
    exec_array_swap(state, pc->axis_1, pc->axis_2);
    DISPATCH();

L_OP_ARRAY_MULTI_CNOT:
    exec_array_multi_cnot(state, pc->axis_1, pc->multi_gate.mask);
    DISPATCH();

L_OP_ARRAY_MULTI_CZ:
    exec_array_multi_cz(state, pc->axis_1, pc->multi_gate.mask);
    DISPATCH();

L_OP_ARRAY_H:
    exec_array_h(state, pc->axis_1);
    DISPATCH();

L_OP_ARRAY_S:
    exec_array_s(state, pc->axis_1);
    DISPATCH();

L_OP_ARRAY_S_DAG:
    exec_array_s_dag(state, pc->axis_1);
    DISPATCH();

L_OP_EXPAND:
    exec_expand(state, pc->axis_1);
    DISPATCH();

L_OP_PHASE_T:
    exec_phase_t(state, pc->axis_1);
    DISPATCH();

L_OP_PHASE_T_DAG:
    exec_phase_t_dag(state, pc->axis_1);
    DISPATCH();

L_OP_EXPAND_T:
    exec_expand_t(state, pc->axis_1);
    DISPATCH();

L_OP_EXPAND_T_DAG:
    exec_expand_t_dag(state, pc->axis_1);
    DISPATCH();

L_OP_PHASE_ROT:
    exec_phase_rot(state, pc->axis_1, pc->math.weight_re, pc->math.weight_im);
    DISPATCH();

L_OP_EXPAND_ROT:
    exec_expand_rot(state, pc->axis_1, pc->math.weight_re, pc->math.weight_im);
    DISPATCH();

L_OP_ARRAY_U2:
    exec_array_u2(state, program.constant_pool, pc->axis_1, pc->u2.cp_idx);
    DISPATCH();

L_OP_MEAS_DORMANT_STATIC:
    if (pc->flags & Instruction::FLAG_IDENTITY) {
        state.meas_record[pc->classical.classical_idx] =
            (pc->flags & Instruction::FLAG_SIGN) ? 1 : 0;
    } else {
        exec_meas_dormant_static(state, pc->axis_1, pc->classical.classical_idx,
                                 (pc->flags & Instruction::FLAG_SIGN) != 0);
    }
    DISPATCH();

L_OP_MEAS_DORMANT_RANDOM:
    exec_meas_dormant_random(state, pc->axis_1, pc->classical.classical_idx,
                             (pc->flags & Instruction::FLAG_SIGN) != 0);
    DISPATCH();

L_OP_MEAS_ACTIVE_DIAGONAL:
    exec_meas_active_diagonal(state, pc->axis_1, pc->classical.classical_idx,
                              (pc->flags & Instruction::FLAG_SIGN) != 0);
    DISPATCH();

L_OP_MEAS_ACTIVE_INTERFERE:
    exec_meas_active_interfere(state, pc->axis_1, pc->classical.classical_idx,
                               (pc->flags & Instruction::FLAG_SIGN) != 0);
    DISPATCH();

L_OP_SWAP_MEAS_INTERFERE:
    exec_swap_meas_interfere(state, pc->axis_1, pc->axis_2, pc->classical.classical_idx,
                             (pc->flags & Instruction::FLAG_SIGN) != 0);
    DISPATCH();

L_OP_APPLY_PAULI:
    exec_apply_pauli(state, program.constant_pool, pc->pauli.cp_mask_idx, pc->pauli.condition_idx);
    DISPATCH();

L_OP_NOISE:
    exec_noise(state, program.constant_pool, pc->pauli.cp_mask_idx);
    DISPATCH();

L_OP_NOISE_BLOCK:
    exec_noise_block(state, program.constant_pool, pc->pauli.cp_mask_idx, pc->pauli.condition_idx);
    DISPATCH();

L_OP_READOUT_NOISE:
    exec_readout_noise(state, program.constant_pool, pc->pauli.cp_mask_idx);
    DISPATCH();

L_OP_DETECTOR:
    exec_detector(state, program.constant_pool, pc->pauli.cp_mask_idx, pc->pauli.condition_idx,
                  (pc->flags & Instruction::FLAG_EXPECTED_ONE) != 0);
    DISPATCH();

L_OP_POSTSELECT:
    if (exec_postselect(state, program.constant_pool, pc->pauli.cp_mask_idx,
                        pc->pauli.condition_idx, (pc->flags & Instruction::FLAG_EXPECTED_ONE) != 0))
        return;
    DISPATCH();

L_OP_OBSERVABLE:
    exec_observable(state, program.constant_pool, pc->pauli.cp_mask_idx, pc->pauli.condition_idx);
    DISPATCH();

#pragma GCC diagnostic pop
#undef DISPATCH
#else
    // Fallback standard C++ switch loop for MSVC and non-GNU compilers
    for (const auto& instr : program.bytecode) {
        switch (instr.opcode) {
            case Opcode::OP_FRAME_CNOT:
                exec_frame_cnot(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_FRAME_CZ:
                exec_frame_cz(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_FRAME_H:
                exec_frame_h(state, instr.axis_1);
                break;
            case Opcode::OP_FRAME_S:
                exec_frame_s(state, instr.axis_1);
                break;
            case Opcode::OP_FRAME_S_DAG:
                exec_frame_s_dag(state, instr.axis_1);
                break;
            case Opcode::OP_FRAME_SWAP:
                exec_frame_swap(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_CNOT:
                exec_array_cnot(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_CZ:
                exec_array_cz(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_SWAP:
                exec_array_swap(state, instr.axis_1, instr.axis_2);
                break;
            case Opcode::OP_ARRAY_MULTI_CNOT:
                exec_array_multi_cnot(state, instr.axis_1, instr.multi_gate.mask);
                break;
            case Opcode::OP_ARRAY_MULTI_CZ:
                exec_array_multi_cz(state, instr.axis_1, instr.multi_gate.mask);
                break;
            case Opcode::OP_ARRAY_H:
                exec_array_h(state, instr.axis_1);
                break;
            case Opcode::OP_ARRAY_S:
                exec_array_s(state, instr.axis_1);
                break;
            case Opcode::OP_ARRAY_S_DAG:
                exec_array_s_dag(state, instr.axis_1);
                break;
            case Opcode::OP_EXPAND:
                exec_expand(state, instr.axis_1);
                break;
            case Opcode::OP_PHASE_T:
                exec_phase_t(state, instr.axis_1);
                break;
            case Opcode::OP_PHASE_T_DAG:
                exec_phase_t_dag(state, instr.axis_1);
                break;
            case Opcode::OP_EXPAND_T:
                exec_expand_t(state, instr.axis_1);
                break;
            case Opcode::OP_EXPAND_T_DAG:
                exec_expand_t_dag(state, instr.axis_1);
                break;
            case Opcode::OP_PHASE_ROT:
                exec_phase_rot(state, instr.axis_1, instr.math.weight_re, instr.math.weight_im);
                break;
            case Opcode::OP_EXPAND_ROT:
                exec_expand_rot(state, instr.axis_1, instr.math.weight_re, instr.math.weight_im);
                break;
            case Opcode::OP_ARRAY_U2:
                exec_array_u2(state, program.constant_pool, instr.axis_1, instr.u2.cp_idx);
                break;
            case Opcode::OP_MEAS_DORMANT_STATIC:
                if (instr.flags & Instruction::FLAG_IDENTITY) {
                    state.meas_record[instr.classical.classical_idx] =
                        (instr.flags & Instruction::FLAG_SIGN) ? 1 : 0;
                } else {
                    exec_meas_dormant_static(state, instr.axis_1, instr.classical.classical_idx,
                                             (instr.flags & Instruction::FLAG_SIGN) != 0);
                }
                break;
            case Opcode::OP_MEAS_DORMANT_RANDOM:
                exec_meas_dormant_random(state, instr.axis_1, instr.classical.classical_idx,
                                         (instr.flags & Instruction::FLAG_SIGN) != 0);
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
                exec_meas_active_diagonal(state, instr.axis_1, instr.classical.classical_idx,
                                          (instr.flags & Instruction::FLAG_SIGN) != 0);
                break;
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
                exec_meas_active_interfere(state, instr.axis_1, instr.classical.classical_idx,
                                           (instr.flags & Instruction::FLAG_SIGN) != 0);
                break;
            case Opcode::OP_SWAP_MEAS_INTERFERE:
                exec_swap_meas_interfere(state, instr.axis_1, instr.axis_2,
                                         instr.classical.classical_idx,
                                         (instr.flags & Instruction::FLAG_SIGN) != 0);
                break;
            case Opcode::OP_APPLY_PAULI:
                exec_apply_pauli(state, program.constant_pool, instr.pauli.cp_mask_idx,
                                 instr.pauli.condition_idx);
                break;
            case Opcode::OP_NOISE:
                exec_noise(state, program.constant_pool, instr.pauli.cp_mask_idx);
                break;
            case Opcode::OP_NOISE_BLOCK:
                exec_noise_block(state, program.constant_pool, instr.pauli.cp_mask_idx,
                                 instr.pauli.condition_idx);
                break;
            case Opcode::OP_READOUT_NOISE:
                exec_readout_noise(state, program.constant_pool, instr.pauli.cp_mask_idx);
                break;
            case Opcode::OP_DETECTOR:
                exec_detector(state, program.constant_pool, instr.pauli.cp_mask_idx,
                              instr.pauli.condition_idx,
                              (instr.flags & Instruction::FLAG_EXPECTED_ONE) != 0);
                break;
            case Opcode::OP_POSTSELECT:
                if (exec_postselect(state, program.constant_pool, instr.pauli.cp_mask_idx,
                                    instr.pauli.condition_idx,
                                    (instr.flags & Instruction::FLAG_EXPECTED_ONE) != 0))
                    return;
                break;
            case Opcode::OP_OBSERVABLE:
                exec_observable(state, program.constant_pool, instr.pauli.cp_mask_idx,
                                instr.pauli.condition_idx);
                break;
        }
    }
#endif
}

}  // namespace UCC_SIMD_NAMESPACE
}  // namespace ucc
