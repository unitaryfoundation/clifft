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
#include <cstring>
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

#if defined(__AVX512F__) && defined(__AVX512DQ__)
// 512-bit complex multiplication: multiplies 4 packed complex numbers (V) by
// 4 complex scalars whose real and imaginary parts are pre-broadcast:
//   V      = [Re0, Im0, Re1, Im1, Re2, Im2, Re3, Im3]
//   S_re   = [c0_re, c0_re, c1_re, c1_re, c2_re, c2_re, c3_re, c3_re]
//   S_im   = [c0_im, c0_im, c1_im, c1_im, c2_im, c2_im, c3_im, c3_im]
// Returns  [c0*z0, c1*z1, c2*z2, c3*z3] via fmaddsub.
static inline __m512d cmul_m512d(__m512d V, __m512d S_re, __m512d S_im) {
    __m512d V_swap = _mm512_permute_pd(V, 0x55);
    return _mm512_fmaddsub_pd(V, S_re, _mm512_mul_pd(V_swap, S_im));
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

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    {
        uint16_t lo = std::min(c, t);
        uint16_t hi = std::max(c, t);

        if (lo == 0 && hi == 1 && state.active_k >= kMinRankFor3DLoop) {
            // Both axes in bottom 2 bits. In register [C0,C1,C2,C3]:
            //   CNOT(0,1): control=axis0, swap C1<->C3 (|01> <-> |11>)
            //   CNOT(1,0): control=axis1, swap C2<->C3 (|10> <-> |11>)
            __m512i perm = (c == 0) ? _mm512_setr_epi64(0, 1, 6, 7, 4, 5, 2, 3)
                                    : _mm512_setr_epi64(0, 1, 2, 3, 6, 7, 4, 5);
            double* d = reinterpret_cast<double*>(state.v());
            uint64_t total = 1ULL << state.active_k;

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                _mm512_store_pd(p, _mm512_permutexvar_pd(perm, val));
            }
            exec_frame_cnot(state, c, t);
            return;
        }

        if (lo <= 1 && hi >= 2 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_hi = 1ULL << hi;
            uint64_t total = 1ULL << state.active_k;
            double* d = reinterpret_cast<double*>(state.v());

            if (c == hi) {
                // Control is the big axis. In the upper half (control=1),
                // swap the lo-bit pairs in each register.
                __m512i perm = (lo == 0) ? _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5)
                                         : _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
                for (uint64_t outer = 0; outer < total; outer += 2 * step_hi) {
                    double* q1 = d + (outer + step_hi) * 2;
                    for (uint64_t k = 0; k < step_hi; k += 4) {
                        double* p = q1 + k * 2;
                        _mm512_store_pd(p, _mm512_permutexvar_pd(perm, _mm512_load_pd(p)));
                    }
                }
            } else {
                // Control is the small axis. Swap lo-bit-set elements
                // between lower half (bit_hi=0) and upper half (bit_hi=1).
                // lo=0: swap odd complex slots; lo=1: swap upper 256 bits.
                __mmask8 swap_mask = (lo == 0) ? __mmask8(0xCC) : __mmask8(0xF0);
                for (uint64_t outer = 0; outer < total; outer += 2 * step_hi) {
                    double* q0 = d + outer * 2;
                    double* q1 = d + (outer + step_hi) * 2;
                    for (uint64_t k = 0; k < step_hi; k += 4) {
                        double* p0 = q0 + k * 2;
                        double* p1 = q1 + k * 2;
                        __m512d v0 = _mm512_load_pd(p0);
                        __m512d v1 = _mm512_load_pd(p1);
                        // Swap only the lanes where lo-bit is set.
                        _mm512_store_pd(p0, _mm512_mask_blend_pd(swap_mask, v0, v1));
                        _mm512_store_pd(p1, _mm512_mask_blend_pd(swap_mask, v1, v0));
                    }
                }
            }
            exec_frame_cnot(state, c, t);
            return;
        }

        if (lo >= 2 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();
            uint64_t step_a = (c == hi) ? step_hi : step_lo;

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* qa = reinterpret_cast<double*>(v + base + step_a);
                    double* qb = reinterpret_cast<double*>(v + base + step_hi + step_lo);

                    for (uint64_t k = 0; k < step_lo; k += 4) {
                        uint64_t off = k * 2;
                        __m512d va = _mm512_load_pd(qa + off);
                        __m512d vb = _mm512_load_pd(qb + off);
                        _mm512_store_pd(qa + off, vb);
                        _mm512_store_pd(qb + off, va);
                    }
                }
            }

            exec_frame_cnot(state, c, t);
            return;
        }
    }
#endif

#if defined(__AVX2__)
    {
        uint16_t lo = std::min(c, t);
        uint16_t hi = std::max(c, t);
        if (lo >= 1 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();
            uint64_t step_a = (c == hi) ? step_hi : step_lo;

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* qa = reinterpret_cast<double*>(v + base + step_a);
                    double* qb = reinterpret_cast<double*>(v + base + step_hi + step_lo);

                    for (uint64_t k = 0; k < step_lo; k += 2) {
                        uint64_t off = k * 2;
                        __m256d va = _mm256_load_pd(qa + off);
                        __m256d vb = _mm256_load_pd(qb + off);
                        _mm256_store_pd(qa + off, vb);
                        _mm256_store_pd(qb + off, va);
                    }
                }
            }

            exec_frame_cnot(state, c, t);
            return;
        }
    }
#endif

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
static inline void exec_array_cz(SchrodingerState& state, uint16_t c, uint16_t t) {
    assert(c != t && "Control and Target axes must be distinct");
    assert(c < state.active_k && c < 64 && "ARRAY_CZ: control axis out of range");
    assert(t < state.active_k && t < 64 && "ARRAY_CZ: target axis out of range");

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    {
        uint16_t lo = std::min(c, t);
        uint16_t hi = std::max(c, t);

        if (lo == 0 && hi == 1 && state.active_k >= kMinRankFor3DLoop) {
            // Both axes in the bottom 2 bits: |11> is C3 in each group of 4.
            // Negate only lanes 6,7 (C3's real,imag) via masked XOR.
            double* d = reinterpret_cast<double*>(state.v());
            uint64_t total = 1ULL << state.active_k;
            __m512d sign_mask = _mm512_set1_pd(-0.0);
            const __mmask8 mask_11 = 0xC0;  // bits: 11000000 = lanes 6,7

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                _mm512_store_pd(p, _mm512_mask_xor_pd(val, mask_11, val, sign_mask));
            }
            exec_frame_cz(state, c, t);
            return;
        }

        if (lo <= 1 && hi >= 2 && state.active_k >= kMinRankFor3DLoop) {
            // Mixed case: one axis in {0,1}, the other >= 2.
            // Negate |11> elements in the upper half (bit_hi set).
            // When lo=0: |11> = odd complex indices in upper half.
            // When lo=1: |11> = upper pair (C2,C3) of each 4-element group in upper half.
            uint64_t step_hi = 1ULL << hi;
            uint64_t total = 1ULL << state.active_k;
            double* d = reinterpret_cast<double*>(state.v());
            __m512d sign_mask = _mm512_set1_pd(-0.0);
            // Negate only elements where lo-bit is set:
            // lo=0: odd complex slots (C1,C3) -> lanes 2,3,6,7
            // lo=1: upper half of register (C2,C3) -> lanes 4,5,6,7
            __mmask8 neg_mask = (lo == 0) ? __mmask8(0xCC) : __mmask8(0xF0);

            for (uint64_t outer = 0; outer < total; outer += 2 * step_hi) {
                double* q1 = d + (outer + step_hi) * 2;
                for (uint64_t k = 0; k < step_hi; k += 4) {
                    double* p = q1 + k * 2;
                    __m512d val = _mm512_load_pd(p);
                    _mm512_store_pd(p, _mm512_mask_xor_pd(val, neg_mask, val, sign_mask));
                }
            }
            exec_frame_cz(state, c, t);
            return;
        }

        if (lo >= 2 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();
            __m512d sign_mask = _mm512_set1_pd(-0.0);

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* q11 = reinterpret_cast<double*>(v + base + step_hi + step_lo);

                    for (uint64_t k = 0; k < step_lo; k += 4) {
                        uint64_t off = k * 2;
                        __m512d val = _mm512_load_pd(q11 + off);
                        _mm512_store_pd(q11 + off, _mm512_xor_pd(val, sign_mask));
                    }
                }
            }

            exec_frame_cz(state, c, t);
            return;
        }
    }
#endif

#if defined(__AVX2__)
    {
        uint16_t lo = std::min(c, t);
        uint16_t hi = std::max(c, t);
        if (lo >= 1 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();
            __m256d sign_mask = _mm256_set1_pd(-0.0);

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* q11 = reinterpret_cast<double*>(v + base + step_hi + step_lo);

                    for (uint64_t k = 0; k < step_lo; k += 2) {
                        uint64_t off = k * 2;
                        __m256d val = _mm256_load_pd(q11 + off);
                        _mm256_store_pd(q11 + off, _mm256_xor_pd(val, sign_mask));
                    }
                }
            }

            exec_frame_cz(state, c, t);
            return;
        }
    }
#endif

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

// SWAP on active axes (a, b): permutes the amplitude array by swapping the
// |01> and |10> quadrants.
static inline void exec_array_swap(SchrodingerState& state, uint16_t a, uint16_t b) {
    assert(a != b && "ARRAY_SWAP: axes a and b must be distinct");
    assert(a < state.active_k && a < 64 && "ARRAY_SWAP: axis a out of range");
    assert(b < state.active_k && b < 64 && "ARRAY_SWAP: axis b out of range");

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    {
        uint16_t lo = std::min(a, b);
        uint16_t hi = std::max(a, b);

        if (lo == 0 && hi == 1 && state.active_k >= kMinRankFor3DLoop) {
            // Swap |01> (C1) and |10> (C2) in each group of 4 complex values.
            const __m512i perm = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
            double* d = reinterpret_cast<double*>(state.v());
            uint64_t total = 1ULL << state.active_k;

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                _mm512_store_pd(p, _mm512_permutexvar_pd(perm, val));
            }
            exec_frame_swap(state, a, b);
            return;
        }

        if (lo >= 2 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* q01 = reinterpret_cast<double*>(v + base + step_lo);
                    double* q10 = reinterpret_cast<double*>(v + base + step_hi);

                    for (uint64_t k = 0; k < step_lo; k += 4) {
                        uint64_t off = k * 2;
                        __m512d va = _mm512_load_pd(q01 + off);
                        __m512d vb = _mm512_load_pd(q10 + off);
                        _mm512_store_pd(q01 + off, vb);
                        _mm512_store_pd(q10 + off, va);
                    }
                }
            }

            exec_frame_swap(state, a, b);
            return;
        }
    }
#endif

#if defined(__AVX2__)
    {
        uint16_t lo = std::min(a, b);
        uint16_t hi = std::max(a, b);
        if (lo >= 1 && state.active_k >= kMinRankFor3DLoop) {
            uint64_t step_lo = 1ULL << lo;
            uint64_t step_hi = 1ULL << hi;
            auto* v = state.v();

            for (uint64_t i = 0; i < (1ULL << state.active_k); i += 2 * step_hi) {
                for (uint64_t j = 0; j < step_hi; j += 2 * step_lo) {
                    uint64_t base = i + j;
                    double* q01 = reinterpret_cast<double*>(v + base + step_lo);
                    double* q10 = reinterpret_cast<double*>(v + base + step_hi);

                    for (uint64_t k = 0; k < step_lo; k += 2) {
                        uint64_t off = k * 2;
                        __m256d va = _mm256_load_pd(q01 + off);
                        __m256d vb = _mm256_load_pd(q10 + off);
                        _mm256_store_pd(q01 + off, vb);
                        _mm256_store_pd(q10 + off, va);
                    }
                }
            }

            exec_frame_swap(state, a, b);
            return;
        }
    }
#endif

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
    auto* __restrict v = state.v();

#if defined(__AVX512F__) && defined(__AVX512DQ__) && UCC_HAS_PDEP
    // AVX-512 opmask path: process 4 complex doubles at a time.
    // Only safe when target >= 2 so consecutive idx values map to
    // contiguous memory addresses (stride >= 4 complex doubles).
    if (target >= 2 && state.active_k >= kMinRankFor3DLoop) {
        uint64_t mapped_cm = _pext_u64(ctrl_mask, pdep_mask);
        auto* v_dbl = reinterpret_cast<double*>(v);

        for (uint64_t idx = 0; idx < half; idx += 4) {
            uint64_t a0 = scatter_bits_1(idx, pdep_mask, target);

            bool p0 = (std::popcount((idx + 0) & mapped_cm) & 1) != 0;
            bool p1 = (std::popcount((idx + 1) & mapped_cm) & 1) != 0;
            bool p2 = (std::popcount((idx + 2) & mapped_cm) & 1) != 0;
            bool p3 = (std::popcount((idx + 3) & mapped_cm) & 1) != 0;

            // Each parity bit controls 2 double lanes (re, im of one complex).
            __mmask8 mask = static_cast<__mmask8>((p0 ? 0x03 : 0) | (p1 ? 0x0C : 0) |
                                                  (p2 ? 0x30 : 0) | (p3 ? 0xC0 : 0));

            __m512d va = _mm512_load_pd(v_dbl + (a0 << 1));
            __m512d vb = _mm512_load_pd(v_dbl + ((a0 | t_bit) << 1));

            // Where mask is 0: keep va in slot 0, vb in slot 1 (no swap).
            // Where mask is 1: put vb in slot 0, va in slot 1 (swap).
            _mm512_store_pd(v_dbl + (a0 << 1), _mm512_mask_blend_pd(mask, va, vb));
            _mm512_store_pd(v_dbl + ((a0 | t_bit) << 1), _mm512_mask_blend_pd(mask, vb, va));
        }

        for (uint16_t c = 0; c < state.active_k; ++c) {
            if (ctrl_mask & (1ULL << c)) {
                exec_frame_cnot(state, c, target);
            }
        }
        return;
    }
#endif

#if UCC_HAS_PDEP
    // ILP trick: map ctrl_mask from address-space into loop-counter-space
    // so popcount runs on idx (immediately available) in parallel with pdep.
    uint64_t mapped_cm = _pext_u64(ctrl_mask, pdep_mask);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, pdep_mask, target);
        bool parity = (std::popcount(idx & mapped_cm) & 1) != 0;
        auto a0 = v[actual];
        auto a1 = v[actual | t_bit];
        v[actual] = parity ? a1 : a0;
        v[actual | t_bit] = parity ? a0 : a1;
    }
#else
    uint64_t cm = ctrl_mask;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, pdep_mask, target);
        bool parity = (std::popcount(actual & cm) & 1) != 0;
        auto a0 = v[actual];
        auto a1 = v[actual | t_bit];
        v[actual] = parity ? a1 : a0;
        v[actual | t_bit] = parity ? a0 : a1;
    }
#endif

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
//
// Loop iterates over half the array (control bit always set), using
// scatter_bits_1 to skip the control-bit-unset indices entirely.
static inline void exec_array_multi_cz(SchrodingerState& state, uint16_t control,
                                       uint64_t target_mask) {
    assert(control < state.active_k && control < 64);

    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (state.active_k - 1);
    uint64_t pdep_mask = ~c_bit;
    auto* __restrict v = state.v();

#if defined(__AVX512F__) && defined(__AVX512DQ__) && UCC_HAS_PDEP
    // AVX-512 opmask path: process 4 complex doubles at a time.
    // Only safe when control >= 2 so consecutive idx values map to
    // contiguous memory addresses (stride >= 4 complex doubles).
    if (control >= 2 && state.active_k >= kMinRankFor3DLoop) {
        uint64_t mapped_tm = _pext_u64(target_mask, pdep_mask);
        auto* v_dbl = reinterpret_cast<double*>(v);
        __m512d sign_mask = _mm512_set1_pd(-0.0);

        for (uint64_t idx = 0; idx < half; idx += 4) {
            uint64_t a0 = scatter_bits_1(idx, pdep_mask, control) | c_bit;

            bool p0 = (std::popcount((idx + 0) & mapped_tm) & 1) != 0;
            bool p1 = (std::popcount((idx + 1) & mapped_tm) & 1) != 0;
            bool p2 = (std::popcount((idx + 2) & mapped_tm) & 1) != 0;
            bool p3 = (std::popcount((idx + 3) & mapped_tm) & 1) != 0;

            __mmask8 mask = static_cast<__mmask8>((p0 ? 0x03 : 0) | (p1 ? 0x0C : 0) |
                                                  (p2 ? 0x30 : 0) | (p3 ? 0xC0 : 0));

            __m512d val = _mm512_load_pd(v_dbl + (a0 << 1));
            // Negate masked lanes: XOR with sign bit where parity is odd.
            __m512d neg = _mm512_xor_pd(val, sign_mask);
            _mm512_store_pd(v_dbl + (a0 << 1), _mm512_mask_blend_pd(mask, val, neg));
        }

        for (uint16_t t = 0; t < state.active_k; ++t) {
            if (target_mask & (1ULL << t)) {
                exec_frame_cz(state, control, t);
            }
        }
        return;
    }
#endif

#if UCC_HAS_PDEP
    // ILP trick: map target_mask into loop-counter-space so popcount
    // runs on idx (immediately available) in parallel with pdep.
    uint64_t mapped_tm = _pext_u64(target_mask, pdep_mask);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, pdep_mask, control) | c_bit;
        bool negate = (std::popcount(idx & mapped_tm) & 1) != 0;
        v[actual] = negate ? -v[actual] : v[actual];
    }
#else
    uint64_t tm = target_mask;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, pdep_mask, control) | c_bit;
        bool negate = (std::popcount(actual & tm) & 1) != 0;
        v[actual] = negate ? -v[actual] : v[actual];
    }
#endif

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
    uint64_t v_bit = 1ULL << v;
    auto* __restrict arr = state.v();

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (state.active_k >= kMinRankFor3DLoop) {
        double* d = reinterpret_cast<double*>(arr);
        uint64_t total = 1ULL << state.active_k;
        __m512d inv_sqrt2 = _mm512_set1_pd(kInvSqrt2);

        if (v == 0) {
            // Axis 0: butterfly between adjacent complex pairs.
            // [C0,C1,C2,C3] -> C0'=(C0+C1)/sqrt2, C1'=(C0-C1)/sqrt2, etc.
            const __m512i swap_idx = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                __m512d swp = _mm512_permutexvar_pd(swap_idx, val);
                __m512d sum = _mm512_mul_pd(_mm512_add_pd(val, swp), inv_sqrt2);
                __m512d dif = _mm512_mul_pd(_mm512_sub_pd(swp, val), inv_sqrt2);
                // Even complex slots (C0,C2) get (a+b), odd (C1,C3) get (a-b).
                // dif = (swp-val)/sqrt2: at odd slots where swp holds a and val holds b,
                // this yields (a-b)/sqrt2 as required by the Hadamard.
                const __mmask8 blend = 0xCC;  // bits: 11001100
                _mm512_store_pd(p, _mm512_mask_blend_pd(blend, sum, dif));
            }
            exec_frame_h(state, v);
            return;
        }

        if (v == 1) {
            // Axis 1: butterfly between lower and upper 256-bit halves.
            // [C0,C1,C2,C3] -> lower'=(lower+upper)/sqrt2, upper'=(lower-upper)/sqrt2
            const __m512i swap_idx = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                __m512d swp = _mm512_permutexvar_pd(swap_idx, val);
                __m512d sum = _mm512_mul_pd(_mm512_add_pd(val, swp), inv_sqrt2);
                __m512d dif = _mm512_mul_pd(_mm512_sub_pd(swp, val), inv_sqrt2);
                // dif = (swp-val)/sqrt2: at upper half where swp holds the |0> amp
                // and val holds the |1> amp, this yields (a-b)/sqrt2.
                const __mmask8 blend = 0xF0;  // bits: 11110000
                _mm512_store_pd(p, _mm512_mask_blend_pd(blend, sum, dif));
            }
            exec_frame_h(state, v);
            return;
        }

        // Axis >= 2: structured stride loop with 2x unrolling.
        uint64_t step = v_bit;
        for (uint64_t outer = 0; outer < total; outer += 2 * step) {
            double* q0 = d + outer * 2;
            double* q1 = d + (outer + step) * 2;
            uint64_t k = 0;
            for (; k + 8 <= step; k += 8) {
                uint64_t off0 = k * 2;
                uint64_t off1 = (k + 4) * 2;
                __m512d a0 = _mm512_load_pd(q0 + off0);
                __m512d b0 = _mm512_load_pd(q1 + off0);
                __m512d a1 = _mm512_load_pd(q0 + off1);
                __m512d b1 = _mm512_load_pd(q1 + off1);
                _mm512_store_pd(q0 + off0, _mm512_mul_pd(_mm512_add_pd(a0, b0), inv_sqrt2));
                _mm512_store_pd(q1 + off0, _mm512_mul_pd(_mm512_sub_pd(a0, b0), inv_sqrt2));
                _mm512_store_pd(q0 + off1, _mm512_mul_pd(_mm512_add_pd(a1, b1), inv_sqrt2));
                _mm512_store_pd(q1 + off1, _mm512_mul_pd(_mm512_sub_pd(a1, b1), inv_sqrt2));
            }
            for (; k < step; k += 4) {
                uint64_t off = k * 2;
                __m512d a = _mm512_load_pd(q0 + off);
                __m512d b = _mm512_load_pd(q1 + off);
                _mm512_store_pd(q0 + off, _mm512_mul_pd(_mm512_add_pd(a, b), inv_sqrt2));
                _mm512_store_pd(q1 + off, _mm512_mul_pd(_mm512_sub_pd(a, b), inv_sqrt2));
            }
        }
        exec_frame_h(state, v);
        return;
    }
#endif

#if defined(__AVX2__)
    if (v >= 1 && state.active_k >= kMinRankFor3DLoop) {
        uint64_t step = v_bit;
        uint64_t total = 1ULL << state.active_k;
        __m256d inv_sqrt2 = _mm256_set1_pd(kInvSqrt2);
        double* d = reinterpret_cast<double*>(arr);

        for (uint64_t outer = 0; outer < total; outer += 2 * step) {
            double* q0 = d + outer * 2;
            double* q1 = d + (outer + step) * 2;
            for (uint64_t k = 0; k < step; k += 2) {
                uint64_t off = k * 2;
                __m256d a = _mm256_load_pd(q0 + off);
                __m256d b = _mm256_load_pd(q1 + off);
                _mm256_store_pd(q0 + off, _mm256_mul_pd(_mm256_add_pd(a, b), inv_sqrt2));
                _mm256_store_pd(q1 + off, _mm256_mul_pd(_mm256_sub_pd(a, b), inv_sqrt2));
            }
        }

        exec_frame_h(state, v);
        return;
    }
#endif

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t pdep_mask = ~v_bit;

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

// =============================================================================
// Shared Phase Waterfall Helper
// =============================================================================

// Multiplies arr[idx | v_bit] by the complex phase (phase_re, phase_im) for
// all indices where bit v is set. Uses AVX-512/AVX2 2D waterfall loops when
// v >= 2/1 (contiguous chunks), falling through to the scalar pdep loop.
static inline void apply_phase_waterfall(SchrodingerState& state, uint16_t v, double phase_re,
                                         double phase_im) {
    uint64_t v_bit = 1ULL << v;
    auto* __restrict arr = state.v();

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (state.active_k >= kMinRankFor3DLoop) {
        double* d = reinterpret_cast<double*>(arr);
        uint64_t total = 1ULL << state.active_k;

        if (v == 0) {
            // Axis 0: odd complex elements (index 1,3,5,...) get the phase.
            // In a 512-bit register [C0, C1, C2, C3], apply phase to C1,C3
            // and leave C0,C2 unchanged. Use blend after cmul.
            __m512d s_re = _mm512_set1_pd(phase_re);
            __m512d s_im = _mm512_set1_pd(phase_im);
            // Blend mask: keep lanes 0-1 (C0) and 4-5 (C2) from original,
            // take lanes 2-3 (C1) and 6-7 (C3) from the phase-multiplied version.
            const __mmask8 blend = 0xCC;  // bits: 11001100

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                __m512d phased = cmul_m512d(val, s_re, s_im);
                _mm512_store_pd(p, _mm512_mask_blend_pd(blend, val, phased));
            }
            return;
        }

        if (v == 1) {
            // Axis 1: upper-half complex elements (index 2,3 in each group
            // of 4) get the phase. In [C0, C1, C2, C3], apply to C2,C3.
            __m512d s_re = _mm512_set1_pd(phase_re);
            __m512d s_im = _mm512_set1_pd(phase_im);
            const __mmask8 blend = 0xF0;  // bits: 11110000

            for (uint64_t i = 0; i < total; i += 4) {
                double* p = d + (i << 1);
                __m512d val = _mm512_load_pd(p);
                __m512d phased = cmul_m512d(val, s_re, s_im);
                _mm512_store_pd(p, _mm512_mask_blend_pd(blend, val, phased));
            }
            return;
        }

        // Axis >= 2: structured stride loop with 2x unrolling.
        uint64_t step = v_bit;
        __m512d s_re = _mm512_set1_pd(phase_re);
        __m512d s_im = _mm512_set1_pd(phase_im);

        for (uint64_t outer = 0; outer < total; outer += 2 * step) {
            double* base = d + (outer + step) * 2;
            uint64_t k = 0;
            for (; k + 8 <= step; k += 8) {
                uint64_t off0 = k * 2;
                uint64_t off1 = (k + 4) * 2;
                __m512d v0 = _mm512_load_pd(base + off0);
                __m512d v1 = _mm512_load_pd(base + off1);
                _mm512_store_pd(base + off0, cmul_m512d(v0, s_re, s_im));
                _mm512_store_pd(base + off1, cmul_m512d(v1, s_re, s_im));
            }
            for (; k < step; k += 4) {
                uint64_t off = k * 2;
                __m512d val = _mm512_load_pd(base + off);
                _mm512_store_pd(base + off, cmul_m512d(val, s_re, s_im));
            }
        }
        return;
    }
#endif

#if defined(__AVX2__)
    if (v >= 1 && state.active_k >= kMinRankFor3DLoop) {
        uint64_t step = v_bit;
        uint64_t total = 1ULL << state.active_k;
        __m256d s_re = _mm256_set1_pd(phase_re);
        __m256d s_im = _mm256_set1_pd(phase_im);
        double* d = reinterpret_cast<double*>(arr);

        for (uint64_t outer = 0; outer < total; outer += 2 * step) {
            double* base = d + (outer + step) * 2;
            for (uint64_t k = 0; k < step; k += 2) {
                uint64_t off = k * 2;
                __m256d val = _mm256_load_pd(base + off);
                _mm256_store_pd(base + off, cmul_m256d(val, s_re, s_im));
            }
        }
        return;
    }
#endif

    uint64_t iters = 1ULL << (state.active_k - 1);
    uint64_t pdep_mask = ~v_bit;
    std::complex<double> phase(phase_re, phase_im);
    for (uint64_t i = 0; i < iters; ++i) {
        arr[scatter_bits_1(i, pdep_mask, v) | v_bit] *= phase;
    }
}

// ARRAY_S on active axis v: applies diag(1, i) to the amplitude array,
// then updates the Pauli frame identically to FRAME_S.
static inline void exec_array_s(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "ARRAY_S: axis out of range");
    apply_phase_waterfall(state, v, 0.0, 1.0);
    exec_frame_s(state, v);
}

// ARRAY_S_DAG on active axis v: applies diag(1, -i) to the amplitude array,
// then updates the Pauli frame identically to FRAME_S_DAG.
static inline void exec_array_s_dag(SchrodingerState& state, uint16_t v) {
    assert(v < state.active_k && v < 64 && "ARRAY_S_DAG: axis out of range");
    apply_phase_waterfall(state, v, 0.0, -1.0);
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
    (void)v;
    uint64_t half = 1ULL << state.active_k;
    auto* __restrict arr = state.v();

    std::memcpy(arr + half, arr, half * sizeof(std::complex<double>));

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));
}

// =============================================================================
// Flat 1D EXPAND+Phase Vectorization Helper
// =============================================================================

// Copies arr[0..half) to arr[half..2*half) while multiplying by a complex
// phase. Uses AVX-512/AVX2 flat 1D loops with aggressive thresholds since
// EXPAND targets v == active_k (perfectly contiguous, no outer loop needed).
static inline void expand_with_phase(std::complex<double>* __restrict arr, uint64_t half,
                                     double phase_re, double phase_im) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (half >= 4) {
        __m512d s_re = _mm512_set1_pd(phase_re);
        __m512d s_im = _mm512_set1_pd(phase_im);
        double* src = reinterpret_cast<double*>(arr);
        double* dst = reinterpret_cast<double*>(arr + half);
        uint64_t doubles = half * 2;

        for (uint64_t k = 0; k < doubles; k += 8) {
            __m512d val = _mm512_load_pd(src + k);
            _mm512_store_pd(dst + k, cmul_m512d(val, s_re, s_im));
        }
        return;
    }
#endif

#if defined(__AVX2__)
    if (half >= 2) {
        __m256d s_re = _mm256_set1_pd(phase_re);
        __m256d s_im = _mm256_set1_pd(phase_im);
        double* src = reinterpret_cast<double*>(arr);
        double* dst = reinterpret_cast<double*>(arr + half);
        uint64_t doubles = half * 2;

        for (uint64_t k = 0; k < doubles; k += 4) {
            __m256d val = _mm256_load_pd(src + k);
            _mm256_store_pd(dst + k, cmul_m256d(val, s_re, s_im));
        }
        return;
    }
#endif

    std::complex<double> phase(phase_re, phase_im);
    for (uint64_t i = 0; i < half; ++i) {
        arr[i + half] = arr[i] * phase;
    }
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

    if (px) {
        apply_phase_waterfall(state, v, kInvSqrt2, -kInvSqrt2);
        state.multiply_phase(kExpIPiOver4);
    } else {
        apply_phase_waterfall(state, v, kInvSqrt2, kInvSqrt2);
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

    if (px) {
        apply_phase_waterfall(state, v, kInvSqrt2, kInvSqrt2);
        state.multiply_phase(kExpMinusIPiOver4);
    } else {
        apply_phase_waterfall(state, v, kInvSqrt2, -kInvSqrt2);
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

    // If p_x[v]=1, T anticommutes with X -> array gets T_dag, gamma absorbs T.
    double ph_re = kInvSqrt2;
    double ph_im = px ? -kInvSqrt2 : kInvSqrt2;
    expand_with_phase(arr, half, ph_re, ph_im);

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
    double ph_re = kInvSqrt2;
    double ph_im = px ? kInvSqrt2 : -kInvSqrt2;
    expand_with_phase(arr, half, ph_re, ph_im);

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

    if (px) {
        apply_phase_waterfall(state, v, z_re, -z_im);
        state.multiply_phase(z);
    } else {
        apply_phase_waterfall(state, v, z_re, z_im);
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

    double ph_im = px ? -z_im : z_im;
    expand_with_phase(arr, half, z_re, ph_im);

    state.active_k++;
    state.scale_magnitude(1.0 / std::sqrt(2.0));

    if (px) {
        state.multiply_phase(std::complex<double>(z_re, z_im));
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

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (state.active_k >= kMinRankFor3DLoop) {
        auto* __restrict arr_dbl = reinterpret_cast<double*>(arr);

        if (axis == 0) {
            // Axis 0: |0> and |1> amplitudes are adjacent complex pairs.
            // A 512-bit register holds [C0, C1, C2, C3] where C0/C2 are
            // |0>-branch and C1/C3 are |1>-branch. Swap adjacent pairs
            // via permutexvar to get the cross-term, then FMA both halves.
            const __m512i swap_idx = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);

            // Broadcast matrix elements into lane-aware vectors:
            // lanes 0-1,4-5 (even complex slots = |0>) get m00/m01
            // lanes 2-3,6-7 (odd complex slots = |1>) get m11/m10
            // Note: at odd slots, v_in holds b (|1>) and v_swp holds a (|0>),
            // so c_self (applied to v_in=b) needs m11, c_swap (applied to v_swp=a) needs m10.
            __m512d c_self_re = _mm512_setr_pd(m00.real(), m00.real(), m11.real(), m11.real(),
                                               m00.real(), m00.real(), m11.real(), m11.real());
            __m512d c_self_im = _mm512_setr_pd(m00.imag(), m00.imag(), m11.imag(), m11.imag(),
                                               m00.imag(), m00.imag(), m11.imag(), m11.imag());
            __m512d c_swap_re = _mm512_setr_pd(m01.real(), m01.real(), m10.real(), m10.real(),
                                               m01.real(), m01.real(), m10.real(), m10.real());
            __m512d c_swap_im = _mm512_setr_pd(m01.imag(), m01.imag(), m10.imag(), m10.imag(),
                                               m01.imag(), m01.imag(), m10.imag(), m10.imag());

            uint64_t total = 1ULL << state.active_k;
            for (uint64_t i = 0; i < total; i += 4) {
                double* p = arr_dbl + (i << 1);
                __m512d v_in = _mm512_load_pd(p);
                __m512d v_swp = _mm512_permutexvar_pd(swap_idx, v_in);
                _mm512_store_pd(p, _mm512_add_pd(cmul_m512d(v_in, c_self_re, c_self_im),
                                                 cmul_m512d(v_swp, c_swap_re, c_swap_im)));
            }
            return;
        }

        if (axis == 1) {
            // Axis 1: |0> and |1> amplitudes are separated by 2 complex
            // doubles. In a 512-bit register [C0, C1, C2, C3], axis-1
            // pairs C0<->C2 and C1<->C3, i.e. swap lower and upper 256
            // bits. Use permutexvar to rearrange.
            const __m512i swap_idx = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);

            // Lower 256 bits (C0,C1) are |0>-branch, upper (C2,C3) are |1>-branch.
            __m512d c_self_re = _mm512_setr_pd(m00.real(), m00.real(), m00.real(), m00.real(),
                                               m11.real(), m11.real(), m11.real(), m11.real());
            __m512d c_self_im = _mm512_setr_pd(m00.imag(), m00.imag(), m00.imag(), m00.imag(),
                                               m11.imag(), m11.imag(), m11.imag(), m11.imag());
            __m512d c_swap_re = _mm512_setr_pd(m01.real(), m01.real(), m01.real(), m01.real(),
                                               m10.real(), m10.real(), m10.real(), m10.real());
            __m512d c_swap_im = _mm512_setr_pd(m01.imag(), m01.imag(), m01.imag(), m01.imag(),
                                               m10.imag(), m10.imag(), m10.imag(), m10.imag());

            uint64_t total = 1ULL << state.active_k;
            for (uint64_t i = 0; i < total; i += 4) {
                double* p = arr_dbl + (i << 1);
                __m512d v_in = _mm512_load_pd(p);
                __m512d v_swp = _mm512_permutexvar_pd(swap_idx, v_in);
                _mm512_store_pd(p, _mm512_add_pd(cmul_m512d(v_in, c_self_re, c_self_im),
                                                 cmul_m512d(v_swp, c_swap_re, c_swap_im)));
            }
            return;
        }

        // Axis >= 2: process 4 butterflies per iteration, 2x unrolled to
        // overlap DRAM latency. Issuing 4 loads before any math lets the
        // CPU's line fill buffers fetch cache lines in parallel.
        __m512d m00_re = _mm512_set1_pd(m00.real());
        __m512d m00_im = _mm512_set1_pd(m00.imag());
        __m512d m01_re = _mm512_set1_pd(m01.real());
        __m512d m01_im = _mm512_set1_pd(m01.imag());
        __m512d m10_re = _mm512_set1_pd(m10.real());
        __m512d m10_im = _mm512_set1_pd(m10.imag());
        __m512d m11_re = _mm512_set1_pd(m11.real());
        __m512d m11_im = _mm512_set1_pd(m11.imag());

        uint64_t i = 0;
        for (; i + 8 <= iters; i += 8) {
            uint64_t idx0_A = scatter_bits_1(i, pdep_mask, axis);
            uint64_t idx1_A = idx0_A | v_bit;
            uint64_t idx0_B = scatter_bits_1(i + 4, pdep_mask, axis);
            uint64_t idx1_B = idx0_B | v_bit;

            __m512d vA0 = _mm512_load_pd(arr_dbl + (idx0_A << 1));
            __m512d vA1 = _mm512_load_pd(arr_dbl + (idx1_A << 1));
            __m512d vB0 = _mm512_load_pd(arr_dbl + (idx0_B << 1));
            __m512d vB1 = _mm512_load_pd(arr_dbl + (idx1_B << 1));

            __m512d nA0 =
                _mm512_add_pd(cmul_m512d(vA0, m00_re, m00_im), cmul_m512d(vA1, m01_re, m01_im));
            __m512d nA1 =
                _mm512_add_pd(cmul_m512d(vA0, m10_re, m10_im), cmul_m512d(vA1, m11_re, m11_im));
            __m512d nB0 =
                _mm512_add_pd(cmul_m512d(vB0, m00_re, m00_im), cmul_m512d(vB1, m01_re, m01_im));
            __m512d nB1 =
                _mm512_add_pd(cmul_m512d(vB0, m10_re, m10_im), cmul_m512d(vB1, m11_re, m11_im));

            _mm512_store_pd(arr_dbl + (idx0_A << 1), nA0);
            _mm512_store_pd(arr_dbl + (idx1_A << 1), nA1);
            _mm512_store_pd(arr_dbl + (idx0_B << 1), nB0);
            _mm512_store_pd(arr_dbl + (idx1_B << 1), nB1);
        }
        for (; i < iters; i += 4) {
            uint64_t idx0 = scatter_bits_1(i, pdep_mask, axis);
            uint64_t idx1 = idx0 | v_bit;

            __m512d v_old0 = _mm512_load_pd(arr_dbl + (idx0 << 1));
            __m512d v_old1 = _mm512_load_pd(arr_dbl + (idx1 << 1));

            __m512d new0 = _mm512_add_pd(cmul_m512d(v_old0, m00_re, m00_im),
                                         cmul_m512d(v_old1, m01_re, m01_im));
            __m512d new1 = _mm512_add_pd(cmul_m512d(v_old0, m10_re, m10_im),
                                         cmul_m512d(v_old1, m11_re, m11_im));

            _mm512_store_pd(arr_dbl + (idx0 << 1), new0);
            _mm512_store_pd(arr_dbl + (idx1 << 1), new1);
        }
        return;
    }
#endif

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

            __m256d v_old0 = _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(base));
            __m256d v_old1 = _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(base + 2));

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
// Fused 2-Axis U4 Opcode
// =============================================================================

// OP_ARRAY_U4: applies a pre-computed 4x4 unitary matrix from the ConstantPool.
// The 16-state FSM selects the correct matrix based on the incoming Pauli frame
// bits (px_lo, pz_lo, px_hi, pz_hi) on the two target axes. This replaces an
// entire tile run of 1Q+2Q ops with one array sweep.
//
// Basis ordering: |b_hi, b_lo> with lo=LSB. The four basis states at each
// position are indexed as: |00>=0, |01>=1, |10>=2, |11>=3.
static inline void exec_array_u4(SchrodingerState& state, const ConstantPool& pool,
                                 uint16_t axis_lo, uint16_t axis_hi, uint32_t cp_idx) {
    assert(axis_lo < axis_hi && "U4: axis_lo must be less than axis_hi");
    assert(axis_hi < state.active_k && axis_hi < 64 && "U4: axis_hi out of range");

    const auto& node = pool.fused_u4_nodes[cp_idx];

    // Compute 4-bit incoming frame state: (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo
    uint8_t px_lo = bit_get(state.p_x, axis_lo) ? 1 : 0;
    uint8_t pz_lo = bit_get(state.p_z, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(state.p_x, axis_hi) ? 1 : 0;
    uint8_t pz_hi = bit_get(state.p_z, axis_hi) ? 1 : 0;
    uint8_t in_state = static_cast<uint8_t>((pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo);

    const auto& entry = node.entries[in_state];
    state.multiply_phase(entry.gamma_multiplier);

    uint8_t out = entry.out_state;
    bit_set(state.p_x, axis_lo, (out & 1) != 0);
    bit_set(state.p_z, axis_lo, (out & 2) != 0);
    bit_set(state.p_x, axis_hi, (out & 4) != 0);
    bit_set(state.p_z, axis_hi, (out & 8) != 0);

    // Array sweep: apply the 4x4 matrix to every block of 4 elements.
    // Block indices: {base, base|lo_bit, base|hi_bit, base|lo_bit|hi_bit}
    uint64_t lo_bit = 1ULL << axis_lo;
    uint64_t hi_bit = 1ULL << axis_hi;
    uint64_t both_bits = lo_bit | hi_bit;
    auto* __restrict arr = state.v();

    // Extract matrix elements (row-major: mat[row][col])
    const auto& mat = entry.matrix;

    // Iterate over all 2^(k-2) blocks where both axis bits are zero.
    uint64_t iters = 1ULL << (state.active_k - 2);
    uint64_t pdep_mask = ~both_bits;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    // Structured stride loop for axis_lo >= 2: consecutive loop indices
    // produce contiguous memory. Process 4 blocks per AVX-512 iteration.
    if (axis_lo >= 2 && state.active_k >= kMinRankFor3DLoop) {
        auto* d = reinterpret_cast<double*>(arr);
        uint64_t step_lo = lo_bit;
        uint64_t step_hi = hi_bit;

        // Broadcast all 16 matrix coefficients
        __m512d m00_re = _mm512_set1_pd(mat[0][0].real());
        __m512d m00_im = _mm512_set1_pd(mat[0][0].imag());
        __m512d m01_re = _mm512_set1_pd(mat[0][1].real());
        __m512d m01_im = _mm512_set1_pd(mat[0][1].imag());
        __m512d m02_re = _mm512_set1_pd(mat[0][2].real());
        __m512d m02_im = _mm512_set1_pd(mat[0][2].imag());
        __m512d m03_re = _mm512_set1_pd(mat[0][3].real());
        __m512d m03_im = _mm512_set1_pd(mat[0][3].imag());
        __m512d m10_re = _mm512_set1_pd(mat[1][0].real());
        __m512d m10_im = _mm512_set1_pd(mat[1][0].imag());
        __m512d m11_re = _mm512_set1_pd(mat[1][1].real());
        __m512d m11_im = _mm512_set1_pd(mat[1][1].imag());
        __m512d m12_re = _mm512_set1_pd(mat[1][2].real());
        __m512d m12_im = _mm512_set1_pd(mat[1][2].imag());
        __m512d m13_re = _mm512_set1_pd(mat[1][3].real());
        __m512d m13_im = _mm512_set1_pd(mat[1][3].imag());
        __m512d m20_re = _mm512_set1_pd(mat[2][0].real());
        __m512d m20_im = _mm512_set1_pd(mat[2][0].imag());
        __m512d m21_re = _mm512_set1_pd(mat[2][1].real());
        __m512d m21_im = _mm512_set1_pd(mat[2][1].imag());
        __m512d m22_re = _mm512_set1_pd(mat[2][2].real());
        __m512d m22_im = _mm512_set1_pd(mat[2][2].imag());
        __m512d m23_re = _mm512_set1_pd(mat[2][3].real());
        __m512d m23_im = _mm512_set1_pd(mat[2][3].imag());
        __m512d m30_re = _mm512_set1_pd(mat[3][0].real());
        __m512d m30_im = _mm512_set1_pd(mat[3][0].imag());
        __m512d m31_re = _mm512_set1_pd(mat[3][1].real());
        __m512d m31_im = _mm512_set1_pd(mat[3][1].imag());
        __m512d m32_re = _mm512_set1_pd(mat[3][2].real());
        __m512d m32_im = _mm512_set1_pd(mat[3][2].imag());
        __m512d m33_re = _mm512_set1_pd(mat[3][3].real());
        __m512d m33_im = _mm512_set1_pd(mat[3][3].imag());

        for (uint64_t i3 = 0; i3 < (1ULL << state.active_k); i3 += 2 * step_hi) {
            for (uint64_t i2 = 0; i2 < step_hi; i2 += 2 * step_lo) {
                uint64_t base = i3 + i2;
                double* p00 = d + base * 2;
                double* p01 = d + (base + step_lo) * 2;
                double* p10 = d + (base + step_hi) * 2;
                double* p11 = d + (base + step_hi + step_lo) * 2;

                for (uint64_t k = 0; k < step_lo; k += 4) {
                    uint64_t off = k * 2;
                    __m512d v0 = _mm512_load_pd(p00 + off);
                    __m512d v1 = _mm512_load_pd(p01 + off);
                    __m512d v2 = _mm512_load_pd(p10 + off);
                    __m512d v3 = _mm512_load_pd(p11 + off);

                    __m512d n0 = _mm512_add_pd(_mm512_add_pd(cmul_m512d(v0, m00_re, m00_im),
                                                             cmul_m512d(v1, m01_re, m01_im)),
                                               _mm512_add_pd(cmul_m512d(v2, m02_re, m02_im),
                                                             cmul_m512d(v3, m03_re, m03_im)));
                    __m512d n1 = _mm512_add_pd(_mm512_add_pd(cmul_m512d(v0, m10_re, m10_im),
                                                             cmul_m512d(v1, m11_re, m11_im)),
                                               _mm512_add_pd(cmul_m512d(v2, m12_re, m12_im),
                                                             cmul_m512d(v3, m13_re, m13_im)));
                    __m512d n2 = _mm512_add_pd(_mm512_add_pd(cmul_m512d(v0, m20_re, m20_im),
                                                             cmul_m512d(v1, m21_re, m21_im)),
                                               _mm512_add_pd(cmul_m512d(v2, m22_re, m22_im),
                                                             cmul_m512d(v3, m23_re, m23_im)));
                    __m512d n3 = _mm512_add_pd(_mm512_add_pd(cmul_m512d(v0, m30_re, m30_im),
                                                             cmul_m512d(v1, m31_re, m31_im)),
                                               _mm512_add_pd(cmul_m512d(v2, m32_re, m32_im),
                                                             cmul_m512d(v3, m33_re, m33_im)));

                    _mm512_store_pd(p00 + off, n0);
                    _mm512_store_pd(p01 + off, n1);
                    _mm512_store_pd(p10 + off, n2);
                    _mm512_store_pd(p11 + off, n3);
                }
            }
        }
        return;
    }
#endif

#if defined(__AVX2__)
    // AVX2 structured stride loop for axis_lo >= 1: process 2 blocks per
    // iteration using 256-bit loads (2 complex doubles per register).
    if (axis_lo >= 1) {
        auto* d = reinterpret_cast<double*>(arr);
        uint64_t step_lo = lo_bit;
        uint64_t step_hi = hi_bit;

        // Broadcast all 16 matrix coefficients into 256-bit registers
        __m256d m00_re = _mm256_set1_pd(mat[0][0].real());
        __m256d m00_im = _mm256_set1_pd(mat[0][0].imag());
        __m256d m01_re = _mm256_set1_pd(mat[0][1].real());
        __m256d m01_im = _mm256_set1_pd(mat[0][1].imag());
        __m256d m02_re = _mm256_set1_pd(mat[0][2].real());
        __m256d m02_im = _mm256_set1_pd(mat[0][2].imag());
        __m256d m03_re = _mm256_set1_pd(mat[0][3].real());
        __m256d m03_im = _mm256_set1_pd(mat[0][3].imag());
        __m256d m10_re = _mm256_set1_pd(mat[1][0].real());
        __m256d m10_im = _mm256_set1_pd(mat[1][0].imag());
        __m256d m11_re = _mm256_set1_pd(mat[1][1].real());
        __m256d m11_im = _mm256_set1_pd(mat[1][1].imag());
        __m256d m12_re = _mm256_set1_pd(mat[1][2].real());
        __m256d m12_im = _mm256_set1_pd(mat[1][2].imag());
        __m256d m13_re = _mm256_set1_pd(mat[1][3].real());
        __m256d m13_im = _mm256_set1_pd(mat[1][3].imag());
        __m256d m20_re = _mm256_set1_pd(mat[2][0].real());
        __m256d m20_im = _mm256_set1_pd(mat[2][0].imag());
        __m256d m21_re = _mm256_set1_pd(mat[2][1].real());
        __m256d m21_im = _mm256_set1_pd(mat[2][1].imag());
        __m256d m22_re = _mm256_set1_pd(mat[2][2].real());
        __m256d m22_im = _mm256_set1_pd(mat[2][2].imag());
        __m256d m23_re = _mm256_set1_pd(mat[2][3].real());
        __m256d m23_im = _mm256_set1_pd(mat[2][3].imag());
        __m256d m30_re = _mm256_set1_pd(mat[3][0].real());
        __m256d m30_im = _mm256_set1_pd(mat[3][0].imag());
        __m256d m31_re = _mm256_set1_pd(mat[3][1].real());
        __m256d m31_im = _mm256_set1_pd(mat[3][1].imag());
        __m256d m32_re = _mm256_set1_pd(mat[3][2].real());
        __m256d m32_im = _mm256_set1_pd(mat[3][2].imag());
        __m256d m33_re = _mm256_set1_pd(mat[3][3].real());
        __m256d m33_im = _mm256_set1_pd(mat[3][3].imag());

        for (uint64_t i3 = 0; i3 < (1ULL << state.active_k); i3 += 2 * step_hi) {
            for (uint64_t i2 = 0; i2 < step_hi; i2 += 2 * step_lo) {
                uint64_t base = i3 + i2;
                double* p00 = d + base * 2;
                double* p01 = d + (base + step_lo) * 2;
                double* p10 = d + (base + step_hi) * 2;
                double* p11 = d + (base + step_hi + step_lo) * 2;

                for (uint64_t k = 0; k < step_lo; k += 2) {
                    uint64_t off = k * 2;
                    __m256d v0 = _mm256_load_pd(p00 + off);
                    __m256d v1 = _mm256_load_pd(p01 + off);
                    __m256d v2 = _mm256_load_pd(p10 + off);
                    __m256d v3 = _mm256_load_pd(p11 + off);

                    __m256d n0 = _mm256_add_pd(_mm256_add_pd(cmul_m256d(v0, m00_re, m00_im),
                                                             cmul_m256d(v1, m01_re, m01_im)),
                                               _mm256_add_pd(cmul_m256d(v2, m02_re, m02_im),
                                                             cmul_m256d(v3, m03_re, m03_im)));
                    __m256d n1 = _mm256_add_pd(_mm256_add_pd(cmul_m256d(v0, m10_re, m10_im),
                                                             cmul_m256d(v1, m11_re, m11_im)),
                                               _mm256_add_pd(cmul_m256d(v2, m12_re, m12_im),
                                                             cmul_m256d(v3, m13_re, m13_im)));
                    __m256d n2 = _mm256_add_pd(_mm256_add_pd(cmul_m256d(v0, m20_re, m20_im),
                                                             cmul_m256d(v1, m21_re, m21_im)),
                                               _mm256_add_pd(cmul_m256d(v2, m22_re, m22_im),
                                                             cmul_m256d(v3, m23_re, m23_im)));
                    __m256d n3 = _mm256_add_pd(_mm256_add_pd(cmul_m256d(v0, m30_re, m30_im),
                                                             cmul_m256d(v1, m31_re, m31_im)),
                                               _mm256_add_pd(cmul_m256d(v2, m32_re, m32_im),
                                                             cmul_m256d(v3, m33_re, m33_im)));

                    _mm256_store_pd(p00 + off, n0);
                    _mm256_store_pd(p01 + off, n1);
                    _mm256_store_pd(p10 + off, n2);
                    _mm256_store_pd(p11 + off, n3);
                }
            }
        }
        return;
    }
#endif

    // Scalar fallback: iterate over 2^(k-2) blocks using bit scattering.
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, pdep_mask, axis_lo, axis_hi);
        uint64_t i0 = base;
        uint64_t i1 = base | lo_bit;
        uint64_t i2 = base | hi_bit;
        uint64_t i3 = base | both_bits;

        auto v0 = arr[i0];
        auto v1 = arr[i1];
        auto v2 = arr[i2];
        auto v3 = arr[i3];

        arr[i0] = mat[0][0] * v0 + mat[0][1] * v1 + mat[0][2] * v2 + mat[0][3] * v3;
        arr[i1] = mat[1][0] * v0 + mat[1][1] * v1 + mat[1][2] * v2 + mat[1][3] * v3;
        arr[i2] = mat[2][0] * v0 + mat[2][1] * v1 + mat[2][2] * v2 + mat[2][3] * v3;
        arr[i3] = mat[3][0] * v0 + mat[3][1] * v1 + mat[3][2] * v2 + mat[3][3] * v3;
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

    // Channel selection: PRNG roll determines which Pauli fires.
    // This is unchanged for forced faults -- we only control *whether*
    // the site fires, not *which* channel within it.
    double rand = state.random_double() * prob_sum;
    double cumulative = 0.0;
    for (const auto& ch : site.channels) {
        cumulative += ch.prob;
        if (rand < cumulative) {
            apply_pauli_to_frame(state, ch.destab_mask, ch.stab_mask, false);
            break;
        }
    }

    // Advance to next firing site: forced-fault cursor or gap sampler.
    if (state.forced_faults.active) {
        state.advance_forced_noise();
    } else {
        state.next_noise_idx++;
        state.draw_next_noise(pool.noise_hazards);
    }
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
// In forced-fault mode, a two-pointer comparison replaces the PRNG roll:
// the bit flips iff entry_idx matches the next forced readout index.
static inline void exec_readout_noise(SchrodingerState& state, const ConstantPool& pool,
                                      uint32_t entry_idx) {
    assert(entry_idx < pool.readout_noise.size());
    const auto& entry = pool.readout_noise[entry_idx];

    bool fire = false;
    if (state.forced_faults.active) {
        auto& ff = state.forced_faults;
        // The backend emits OP_READOUT_NOISE with strictly increasing
        // entry_idx, so a simple equality check suffices here.
        if (ff.readout_pos < ff.readout_indices.size() &&
            ff.readout_indices[ff.readout_pos] == entry_idx) {
            fire = true;
            ff.readout_pos++;
        }
    } else {
        fire = (state.random_double() < entry.prob);
    }
    if (fire) {
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
        [static_cast<uint8_t>(Opcode::OP_ARRAY_U4)] = &&L_OP_ARRAY_U4,

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

L_OP_ARRAY_U4:
    exec_array_u4(state, program.constant_pool, pc->axis_1, pc->axis_2, pc->u4.cp_idx);
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
            case Opcode::OP_ARRAY_U4:
                exec_array_u4(state, program.constant_pool, instr.axis_1, instr.axis_2,
                              instr.u4.cp_idx);
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
