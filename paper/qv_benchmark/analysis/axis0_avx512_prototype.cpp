// Test 6 Final: In-register AVX-512 operations on axis==0
// Demonstrates the key intrinsics for eliminating the scalar fallback.

#include <cmath>
#include <complex>
#include <cstdio>
#include <immintrin.h>

// Swap adjacent complex<double> pairs within a 512-bit register.
// [c0, c1, c2, c3] -> [c1, c0, c3, c2]
static inline __m512d swap_adj_cx(__m512d v) {
    const __m512i idx = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);
    return _mm512_permutexvar_pd(idx, v);
}

// Hadamard butterfly on axis 0 (stride-1 pairs).
// For pairs (a, b): a' = (a+b)/sqrt(2), b' = (a-b)/sqrt(2).
// Register layout: [a0, b0, a1, b1]
static inline __m512d h_axis0(__m512d v) {
    __m512d swapped = swap_adj_cx(v);
    // v       = [a0, b0, a1, b1]
    // swapped = [b0, a0, b1, a1]
    // For EVEN complex (a-positions): want a+b = v + swapped
    // For ODD  complex (b-positions): want a-b = swapped_from_even - odd_original
    // Actually: (v + swapped) at even = a+b, at odd = b+a (same)
    //           (v - swapped) at even = a-b, at odd = b-a (negated)
    // So blend: even takes (v+swapped), odd takes (swapped-v)
    __m512d add = _mm512_add_pd(v, swapped);
    __m512d sub = _mm512_sub_pd(swapped, v);  // [b-a, a-b, b-a, a-b]
    // Wait, sub at even = b0-a0 (wrong), sub at odd = a0-b0 (right!)
    // So: even positions want add, odd positions want sub
    __mmask8 odd_cx = 0xCC;  // lanes 2,3,6,7 = odd complex indices
    __m512d result = _mm512_mask_blend_pd(odd_cx, add, sub);
    return _mm512_mul_pd(result, _mm512_set1_pd(M_SQRT1_2));
}

// U2 gate on axis 0: apply arbitrary 2x2 matrix to each (a,b) pair.
// mat = {{m00,m01},{m10,m11}}
// a' = m00*a + m01*b, b' = m10*a + m11*b
static inline __m512d u2_axis0(__m512d v, std::complex<double> m00, std::complex<double> m01,
                               std::complex<double> m10, std::complex<double> m11) {
    __m512d swapped = swap_adj_cx(v);
    // v       = [a0, b0, a1, b1]
    // swapped = [b0, a0, b1, a1]
    //
    // For even complex (a-positions): want m00*a + m01*b
    //   = m00 * v[even] + m01 * swapped[even]
    // For odd complex (b-positions):  want m10*a + m11*b
    //   = m10 * swapped[odd] + m11 * v[odd]
    //
    // Build coefficient vectors that alternate m00/m10 and m01/m11:
    __m512d c_self_re = _mm512_setr_pd(m00.real(), m00.real(), m11.real(), m11.real(), m00.real(),
                                       m00.real(), m11.real(), m11.real());
    __m512d c_self_im = _mm512_setr_pd(m00.imag(), m00.imag(), m11.imag(), m11.imag(), m00.imag(),
                                       m00.imag(), m11.imag(), m11.imag());
    __m512d c_swap_re = _mm512_setr_pd(m01.real(), m01.real(), m10.real(), m10.real(), m01.real(),
                                       m01.real(), m10.real(), m10.real());
    __m512d c_swap_im = _mm512_setr_pd(m01.imag(), m01.imag(), m10.imag(), m10.imag(), m01.imag(),
                                       m01.imag(), m10.imag(), m10.imag());

    // Complex multiply helper: cmul(V, re, im)
    auto cmul = [](__m512d V, __m512d re, __m512d im) -> __m512d {
        __m512d V_swap = _mm512_permute_pd(V, 0x55);
        return _mm512_fmaddsub_pd(V, re, _mm512_mul_pd(V_swap, im));
    };

    return _mm512_add_pd(cmul(v, c_self_re, c_self_im), cmul(swapped, c_swap_re, c_swap_im));
}

// CNOT(ctrl=higher_axis, tgt=0): swap adjacent pairs where control bit is set.
// For CNOT(1,0): control bit 1 set => indices 2,3 (double lanes 4-7).
static inline __m512d cnot_c1_t0(__m512d v) {
    __m512d swapped = swap_adj_cx(v);
    __mmask8 mask = 0xF0;  // lanes 4,5,6,7
    return _mm512_mask_blend_pd(mask, v, swapped);
}

// Phase rotation on axis 0: multiply odd-indexed complex by phase.
// [a0, b0, a1, b1] -> [a0, b0*phase, a1, b1*phase]
static inline __m512d phase_axis0(__m512d v, std::complex<double> phase) {
    __m512d swapped = swap_adj_cx(v);
    // v       = [a0, b0, a1, b1]
    // swapped = [b0, a0, b1, a1]
    // We want: even slots unchanged (identity), odd slots *= phase.
    // Use u2 with m00=1, m01=0, m10=0, m11=phase
    return u2_axis0(v, {1, 0}, {0, 0}, {0, 0}, phase);
}

void print_state(const char* label, std::complex<double>* data, int n) {
    printf("%s\n", label);
    for (int i = 0; i < n; i++)
        printf("  v[%d] = (%+.6f, %+.6f)\n", i, data[i].real(), data[i].imag());
}

int main() {
    alignas(64) std::complex<double> data[4];
    double err;
    __m512d* reg = reinterpret_cast<__m512d*>(data);
    int pass = 0, total = 0;

    // Test 1: H(axis=0) on |00>
    data[0] = {1, 0};
    data[1] = {0, 0};
    data[2] = {0, 0};
    data[3] = {0, 0};
    *reg = h_axis0(*reg);
    err = std::abs(data[0] - std::complex<double>(M_SQRT1_2, 0)) +
          std::abs(data[1] - std::complex<double>(M_SQRT1_2, 0)) + std::abs(data[2]) +
          std::abs(data[3]);
    total++;
    if (err < 1e-14)
        pass++;
    printf("Test 1 H|00>:  err=%.2e %s\n", err, err < 1e-14 ? "PASS" : "FAIL");
    print_state("  Result:", data, 4);

    // Test 2: H on non-trivial state
    data[0] = {0.6, 0};
    data[1] = {0.8, 0};
    data[2] = {0, 0};
    data[3] = {0, 0};
    *reg = h_axis0(*reg);
    err = std::abs(data[0] - std::complex<double>(1.4 * M_SQRT1_2, 0)) +
          std::abs(data[1] - std::complex<double>(-0.2 * M_SQRT1_2, 0)) + std::abs(data[2]) +
          std::abs(data[3]);
    total++;
    if (err < 1e-14)
        pass++;
    printf("Test 2 H|psi>: err=%.2e %s\n", err, err < 1e-14 ? "PASS" : "FAIL");

    // Test 3: CNOT(1,0)
    data[0] = {1, 0};
    data[1] = {0, 0};
    data[2] = {0.3, 0.4};
    data[3] = {0.5, 0.6};
    *reg = cnot_c1_t0(*reg);
    err = std::abs(data[0] - std::complex<double>(1, 0)) + std::abs(data[1]) +
          std::abs(data[2] - std::complex<double>(0.5, 0.6)) +
          std::abs(data[3] - std::complex<double>(0.3, 0.4));
    total++;
    if (err < 1e-14)
        pass++;
    printf("Test 3 CNOT:   err=%.2e %s\n", err, err < 1e-14 ? "PASS" : "FAIL");

    // Test 4: U2(H) via generic u2
    data[0] = {1, 0};
    data[1] = {0, 0};
    data[2] = {0, 0};
    data[3] = {0, 0};
    *reg = u2_axis0(*reg, {M_SQRT1_2, 0}, {M_SQRT1_2, 0}, {M_SQRT1_2, 0}, {-M_SQRT1_2, 0});
    err = std::abs(data[0] - std::complex<double>(M_SQRT1_2, 0)) +
          std::abs(data[1] - std::complex<double>(M_SQRT1_2, 0)) + std::abs(data[2]) +
          std::abs(data[3]);
    total++;
    if (err < 1e-14)
        pass++;
    printf("Test 4 U2(H):  err=%.2e %s\n", err, err < 1e-14 ? "PASS" : "FAIL");

    // Test 5: S gate via phase
    data[0] = {0.6, 0};
    data[1] = {0, 0.8};
    data[2] = {0.3, 0.4};
    data[3] = {0.5, 0.6};
    *reg = phase_axis0(*reg, {0, 1});  // S = diag(1, i)
    err = std::abs(data[0] - std::complex<double>(0.6, 0)) +
          std::abs(data[1] - std::complex<double>(-0.8, 0)) +
          std::abs(data[2] - std::complex<double>(0.3, 0.4)) +
          std::abs(data[3] - std::complex<double>(-0.6, 0.5));
    total++;
    if (err < 1e-14)
        pass++;
    printf("Test 5 S:      err=%.2e %s\n", err, err < 1e-14 ? "PASS" : "FAIL");

    printf("\n%d/%d passed\n", pass, total);
    return pass == total ? 0 : 1;
}
