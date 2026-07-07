// cpu_kernels.hpp -- portable CPU implementations of clifft's active-block
// kernels, faithful to the scalar reference paths in svm_kernels.inl.
//
// These are written as clean loops (not AVX intrinsics) and compiled with
// -O3 -march=native so the compiler autovectorizes to NEON (ARM / Apple
// Silicon / Grace) or AVX (x86). On ARM this mirrors what clifft actually
// runs, since clifft's hand-written AVX paths are x86-only.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>

#include "bench_common.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mb {

using cd = std::complex<double>;

// Loops parallelize only above clifft's threshold (k >= 18) via the
// `if (k >= kMinRankForThreads)` clause on each omp pragma, matching clifft's
// policy that thread-startup overhead dominates for small arrays.

// --- Single-statevector kernels (active rank k, dimension d = 2^k) ---------

// Hadamard butterfly on axis v: a' = (a+b)/sqrt2, b' = (a-b)/sqrt2.
inline void op_h(cd* __restrict arr, unsigned k, unsigned v) {
    const uint64_t half = uint64_t(1) << (k - 1);
    const uint64_t vbit = uint64_t(1) << v;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(half); ++ii) {
        uint64_t i0 = insert_zero_bit(uint64_t(ii), v);
        uint64_t i1 = i0 | vbit;
        cd a = arr[i0], b = arr[i1];
        arr[i0] = (a + b) * kInvSqrt2;
        arr[i1] = (a - b) * kInvSqrt2;
    }
}

// T phase on axis v: multiply amplitudes with bit v set by e^{i*pi/4}.
inline void op_t(cd* __restrict arr, unsigned k, unsigned v) {
    const uint64_t half = uint64_t(1) << (k - 1);
    const uint64_t vbit = uint64_t(1) << v;
    const cd phase(kTPhaseRe, kTPhaseIm);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(half); ++ii) {
        uint64_t idx = insert_zero_bit(uint64_t(ii), v) | vbit;
        arr[idx] *= phase;
    }
}

// CZ on (c,t): negate amplitudes where both bits set.
inline void op_cz(cd* __restrict arr, unsigned k, unsigned c, unsigned t) {
    const unsigned lo = c < t ? c : t, hi = c < t ? t : c;
    const uint64_t quarter = uint64_t(1) << (k - 2);
    const uint64_t cbit = uint64_t(1) << c, tbit = uint64_t(1) << t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(quarter); ++ii) {
        uint64_t idx = insert_two_zero_bits(uint64_t(ii), lo, hi) | cbit | tbit;
        arr[idx] = -arr[idx];
    }
}

// CNOT on (c,t): swap amplitudes c-bit-set differing in t-bit.
inline void op_cnot(cd* __restrict arr, unsigned k, unsigned c, unsigned t) {
    const unsigned lo = c < t ? c : t, hi = c < t ? t : c;
    const uint64_t quarter = uint64_t(1) << (k - 2);
    const uint64_t cbit = uint64_t(1) << c, tbit = uint64_t(1) << t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(quarter); ++ii) {
        uint64_t base0 = insert_two_zero_bits(uint64_t(ii), lo, hi) | cbit;
        cd tmp = arr[base0];
        arr[base0] = arr[base0 | tbit];
        arr[base0 | tbit] = tmp;
    }
}

// Fused dense 1q matrix on axis v (clifft OP_ARRAY_U2 scalar path, the
// single-axis fusion output): (a,b) -> (m00 a + m01 b, m10 a + m11 b).
// mat = 4 complex entries, row-major.
inline void op_u2(cd* __restrict arr, unsigned k, unsigned v, const cd* __restrict mat) {
    const uint64_t half = uint64_t(1) << (k - 1);
    const uint64_t vbit = uint64_t(1) << v;
    const cd m00 = mat[0], m01 = mat[1], m10 = mat[2], m11 = mat[3];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(half); ++ii) {
        uint64_t i0 = insert_zero_bit(uint64_t(ii), v);
        uint64_t i1 = i0 | vbit;
        cd a = arr[i0], b = arr[i1];
        arr[i0] = m00 * a + m01 * b;
        arr[i1] = m10 * a + m11 * b;
    }
}

// Fused dense 2q matrix on axes (lo < hi) (clifft OP_ARRAY_U4 scalar path,
// the tile-fusion output): 4x4 complex matrix applied to every block
// {base, base|lo, base|hi, base|lo|hi}. mat = 16 complex entries, row-major.
inline void op_u4(cd* __restrict arr, unsigned k, unsigned lo, unsigned hi,
                  const cd* __restrict mat) {
    const uint64_t quarter = uint64_t(1) << (k - 2);
    const uint64_t lobit = uint64_t(1) << lo, hibit = uint64_t(1) << hi;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t ii = 0; ii < int64_t(quarter); ++ii) {
        uint64_t base = insert_two_zero_bits(uint64_t(ii), lo, hi);
        uint64_t idx[4] = {base, base | lobit, base | hibit, base | lobit | hibit};
        cd in[4] = {arr[idx[0]], arr[idx[1]], arr[idx[2]], arr[idx[3]]};
        for (int r = 0; r < 4; ++r)
            arr[idx[r]] = mat[4 * r + 0] * in[0] + mat[4 * r + 1] * in[1] +
                          mat[4 * r + 2] * in[2] + mat[4 * r + 3] * in[3];
    }
}

// EXPAND at rank k: copy [0,2^k) into [2^k, 2^{k+1}). (gamma scaling is O(1)
// in clifft and omitted from the timed work.)
inline void op_expand(cd* __restrict arr, unsigned k) {
    const uint64_t half = uint64_t(1) << k;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
    for (int64_t i = 0; i < int64_t(half); ++i) arr[half + i] = arr[i];
#else
    std::memcpy(arr + half, arr, half * sizeof(cd));
#endif
}

// EXPAND_T at rank k: copy with T phase on the new axis (fused, matches
// exec_expand_t).
inline void op_expand_t(cd* __restrict arr, unsigned k) {
    const uint64_t half = uint64_t(1) << k;
    const cd phase(kTPhaseRe, kTPhaseIm);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t i = 0; i < int64_t(half); ++i) arr[half + i] = arr[i] * phase;
}

// Z-basis measurement at rank k (axis k-1): reduce branch probs, sample,
// compact surviving half. Returns sampled bit. Touches 2^k amplitudes.
// force_bit >= 0 pins the outcome: used (a) by validation so CPU and GPU
// follow the same trajectory and (b) by the per-op benchmark rows with
// force_bit=1 so the compact always runs -- matching the GPU row, which
// always compacts (review fix: the two rows previously measured different
// work, CPU compacting only ~50% of calls).
inline int op_meas_diag(cd* __restrict arr, unsigned k, Rng& rng, int force_bit = -1) {
    const uint64_t half = uint64_t(1) << (k - 1);
    double p0 = 0.0, p1 = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : p0, p1) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t i = 0; i < int64_t(half); ++i) {
        p0 += std::norm(arr[i]);
        p1 += std::norm(arr[i + half]);
    }
    double total = p0 + p1;
    int b = force_bit >= 0 ? force_bit : ((rng.next_unit() * total < p0) ? 0 : 1);
    if (b == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
        for (int64_t i = 0; i < int64_t(half); ++i) arr[i] = arr[i + half];
    }
    return b;
}

// X-basis "interfere" measurement at rank k (axis k-1): reduce |a+-b|^2,
// sample, fold. Touches 2^k amplitudes. force_bit as in op_meas_diag.
inline int op_meas_interfere(cd* __restrict arr, unsigned k, Rng& rng, int force_bit = -1) {
    const uint64_t half = uint64_t(1) << (k - 1);
    double pp = 0.0, pm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : pp, pm) if (k >= mb::kMinRankForThreads)
#endif
    for (int64_t i = 0; i < int64_t(half); ++i) {
        cd s = arr[i] + arr[i + half];
        cd d = arr[i] - arr[i + half];
        pp += std::norm(s);
        pm += std::norm(d);
    }
    double total = pp + pm;
    int b = force_bit >= 0 ? force_bit : ((rng.next_unit() * total < pp) ? 0 : 1);
    if (b == 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
        for (int64_t i = 0; i < int64_t(half); ++i) arr[i] = (arr[i] + arr[i + half]) * kInvSqrt2;
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (k >= mb::kMinRankForThreads)
#endif
        for (int64_t i = 0; i < int64_t(half); ++i) arr[i] = (arr[i] - arr[i + half]) * kInvSqrt2;
    }
    return b;
}

}  // namespace mb
