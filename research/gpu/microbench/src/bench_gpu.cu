// bench_gpu.cu -- CUDA (complex128) counterpart to bench_cpu.cpp.
//
// Implements clifft's active-block ops as hand-written CUDA kernels in double
// precision and measures:
//   (1) single-statevector per-op throughput swept over rank k
//   (2) batched-across-shots throughput at fixed small k  <- key hypothesis
//   (3) host<->device transfer cost (quantifies PCIe vs NVLink-C2C on GH200)
// plus a CPU-vs-GPU correctness check on the single-statevector schedule.
//
// Build for the GH200 Hopper GPU: -arch=sm_90 (set via CMake CUDA_ARCH).
//
// Usage: bench_gpu [kmin kmax] [bk1,bk2,...] [layers]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "cpu_kernels.hpp"  // for validation only (host code)

using namespace mb;

#define CHECK(x)                                                                          \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,  \
                    __LINE__);                                                            \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

// --- device complex128 ------------------------------------------------------
struct dc {
    double r, i;
};
__host__ __device__ inline dc operator+(dc a, dc b) { return {a.r + b.r, a.i + b.i}; }
__host__ __device__ inline dc operator-(dc a, dc b) { return {a.r - b.r, a.i - b.i}; }
__host__ __device__ inline dc operator*(dc a, double s) { return {a.r * s, a.i * s}; }
__host__ __device__ inline dc cmul(dc a, dc b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
__host__ __device__ inline double cnorm(dc a) { return a.r * a.r + a.i * a.i; }

__device__ inline uint64_t dev_insert_zero_bit(uint64_t x, unsigned p) {
    uint64_t mask = (uint64_t(1) << p) - 1;
    return (x & mask) | ((x & ~mask) << 1);
}
__device__ inline uint64_t dev_insert_two(uint64_t x, unsigned a, unsigned b) {
    return dev_insert_zero_bit(dev_insert_zero_bit(x, a), b);
}

constexpr double kPhaseRe = 0.7071067811865476;
constexpr double kPhaseIm = 0.7071067811865476;

// =============================================================================
// Single-statevector kernels (dimension d = 2^k). nThreads as noted.
// =============================================================================
__global__ void k_h(dc* a, uint64_t half, unsigned v) {  // nThreads=half
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    uint64_t i0 = dev_insert_zero_bit(tid, v);
    uint64_t i1 = i0 | (uint64_t(1) << v);
    dc x = a[i0], y = a[i1];
    a[i0] = (x + y) * mb::kInvSqrt2;
    a[i1] = (x - y) * mb::kInvSqrt2;
}
__global__ void k_t(dc* a, uint64_t half, unsigned v) {  // nThreads=half
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    uint64_t idx = dev_insert_zero_bit(tid, v) | (uint64_t(1) << v);
    a[idx] = cmul(a[idx], {kPhaseRe, kPhaseIm});
}
__global__ void k_cz(dc* a, uint64_t quarter, unsigned lo, unsigned hi, unsigned c, unsigned t) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= quarter) return;
    uint64_t idx = dev_insert_two(tid, lo, hi) | (uint64_t(1) << c) | (uint64_t(1) << t);
    a[idx] = a[idx] * -1.0;
}
__global__ void k_cnot(dc* a, uint64_t quarter, unsigned lo, unsigned hi, unsigned c, unsigned t) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= quarter) return;
    uint64_t base0 = dev_insert_two(tid, lo, hi) | (uint64_t(1) << c);
    uint64_t base1 = base0 | (uint64_t(1) << t);
    dc tmp = a[base0];
    a[base0] = a[base1];
    a[base1] = tmp;
}
struct U2m { dc m00, m01, m10, m11; };
struct U4m { dc m[16]; };  // row-major, 256 B by value (within kernel-param limits)

__global__ void k_u2(dc* a, uint64_t half, unsigned v, U2m u) {  // nThreads=half
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    uint64_t i0 = dev_insert_zero_bit(tid, v);
    uint64_t i1 = i0 | (uint64_t(1) << v);
    dc x = a[i0], y = a[i1];
    a[i0] = cmul(u.m00, x) + cmul(u.m01, y);
    a[i1] = cmul(u.m10, x) + cmul(u.m11, y);
}
__global__ void k_u4(dc* a, uint64_t quarter, unsigned lo, unsigned hi, U4m u) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= quarter) return;
    uint64_t base = dev_insert_two(tid, lo, hi);
    uint64_t i1 = base | (uint64_t(1) << lo);
    uint64_t i2 = base | (uint64_t(1) << hi);
    uint64_t i3 = i1 | (uint64_t(1) << hi);
    dc in0 = a[base], in1 = a[i1], in2 = a[i2], in3 = a[i3];
    a[base] = cmul(u.m[0], in0) + cmul(u.m[1], in1) + cmul(u.m[2], in2) + cmul(u.m[3], in3);
    a[i1] = cmul(u.m[4], in0) + cmul(u.m[5], in1) + cmul(u.m[6], in2) + cmul(u.m[7], in3);
    a[i2] = cmul(u.m[8], in0) + cmul(u.m[9], in1) + cmul(u.m[10], in2) + cmul(u.m[11], in3);
    a[i3] = cmul(u.m[12], in0) + cmul(u.m[13], in1) + cmul(u.m[14], in2) + cmul(u.m[15], in3);
}

__global__ void k_expand(dc* a, uint64_t half) {  // nThreads=half
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    a[half + tid] = a[tid];
}
__global__ void k_expand_t(dc* a, uint64_t half) {  // nThreads=half
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    a[half + tid] = cmul(a[tid], {kPhaseRe, kPhaseIm});
}

// Block-reduced probability sums for measurement.
__global__ void k_reduce_diag(const dc* a, uint64_t half, double* part0, double* part1) {
    extern __shared__ double sh[];  // 2*blockDim
    double* s0 = sh;
    double* s1 = sh + blockDim.x;
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    double l0 = 0, l1 = 0;
    for (uint64_t i = tid; i < half; i += stride) {
        l0 += cnorm(a[i]);
        l1 += cnorm(a[i + half]);
    }
    s0[threadIdx.x] = l0;
    s1[threadIdx.x] = l1;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part0[blockIdx.x] = s0[0];
        part1[blockIdx.x] = s1[0];
    }
}
__global__ void k_reduce_interfere(const dc* a, uint64_t half, double* partp, double* partm) {
    extern __shared__ double sh[];
    double* sp = sh;
    double* sm = sh + blockDim.x;
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    double lp = 0, lm = 0;
    for (uint64_t i = tid; i < half; i += stride) {
        lp += cnorm(a[i] + a[i + half]);
        lm += cnorm(a[i] - a[i + half]);
    }
    sp[threadIdx.x] = lp;
    sm[threadIdx.x] = lm;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sp[threadIdx.x] += sp[threadIdx.x + s];
            sm[threadIdx.x] += sm[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partp[blockIdx.x] = sp[0];
        partm[blockIdx.x] = sm[0];
    }
}
__global__ void k_compact_upper(dc* a, uint64_t half) {  // diag b==1
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    a[tid] = a[tid + half];
}
__global__ void k_fold(dc* a, uint64_t half, int sign) {  // interfere
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= half) return;
    dc x = a[tid], y = a[tid + half];
    a[tid] = (sign >= 0 ? (x + y) : (x - y)) * mb::kInvSqrt2;
}

// =============================================================================
// Batched kernels: B states of dimension d laid out contiguously (stride d).
// Thread g maps to (shot, work-index). Only H/T/CZ/CNOT are batched (the
// fixed-rank layer schedule; honors cuStateVec's uniform-qubit-count rule).
// =============================================================================
__global__ void kb_h(dc* a, uint64_t d, uint64_t half, unsigned v, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;  // total = B*half
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    dc* s = a + b * d;
    uint64_t i0 = dev_insert_zero_bit(j, v), i1 = i0 | (uint64_t(1) << v);
    dc x = s[i0], y = s[i1];
    s[i0] = (x + y) * mb::kInvSqrt2;
    s[i1] = (x - y) * mb::kInvSqrt2;
}
__global__ void kb_t(dc* a, uint64_t d, uint64_t half, unsigned v, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    dc* s = a + b * d;
    uint64_t idx = dev_insert_zero_bit(j, v) | (uint64_t(1) << v);
    s[idx] = cmul(s[idx], {kPhaseRe, kPhaseIm});
}
__global__ void kb_cz(dc* a, uint64_t d, uint64_t quarter, unsigned lo, unsigned hi, unsigned c,
                      unsigned t, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;  // total = B*quarter
    if (g >= total) return;
    uint64_t b = g / quarter, j = g % quarter;
    dc* s = a + b * d;
    uint64_t idx = dev_insert_two(j, lo, hi) | (uint64_t(1) << c) | (uint64_t(1) << t);
    s[idx] = s[idx] * -1.0;
}
__global__ void kb_cnot(dc* a, uint64_t d, uint64_t quarter, unsigned lo, unsigned hi, unsigned c,
                        unsigned t, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / quarter, j = g % quarter;
    dc* s = a + b * d;
    uint64_t base0 = dev_insert_two(j, lo, hi) | (uint64_t(1) << c);
    uint64_t base1 = base0 | (uint64_t(1) << t);
    dc tmp = s[base0];
    s[base0] = s[base1];
    s[base1] = tmp;
}

__global__ void kb_u2(dc* a, uint64_t d, uint64_t half, unsigned v, U2m u, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    dc* s = a + b * d;
    uint64_t i0 = dev_insert_zero_bit(j, v), i1 = i0 | (uint64_t(1) << v);
    dc x = s[i0], y = s[i1];
    s[i0] = cmul(u.m00, x) + cmul(u.m01, y);
    s[i1] = cmul(u.m10, x) + cmul(u.m11, y);
}
__global__ void kb_u4(dc* a, uint64_t d, uint64_t quarter, unsigned lo, unsigned hi, U4m u,
                      uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / quarter, j = g % quarter;
    dc* s = a + b * d;
    uint64_t base = dev_insert_two(j, lo, hi);
    uint64_t i1 = base | (uint64_t(1) << lo);
    uint64_t i2 = base | (uint64_t(1) << hi);
    uint64_t i3 = i1 | (uint64_t(1) << hi);
    dc in0 = s[base], in1 = s[i1], in2 = s[i2], in3 = s[i3];
    s[base] = cmul(u.m[0], in0) + cmul(u.m[1], in1) + cmul(u.m[2], in2) + cmul(u.m[3], in3);
    s[i1] = cmul(u.m[4], in0) + cmul(u.m[5], in1) + cmul(u.m[6], in2) + cmul(u.m[7], in3);
    s[i2] = cmul(u.m[8], in0) + cmul(u.m[9], in1) + cmul(u.m[10], in2) + cmul(u.m[11], in3);
    s[i3] = cmul(u.m[12], in0) + cmul(u.m[13], in1) + cmul(u.m[14], in2) + cmul(u.m[15], in3);
}

// --- Batched measurement round ----------------------------------------------
// One BLOCK per shot: reduce that shot's two branch-probability sums.
__global__ void kb_reduce_diag(const dc* a, uint64_t d, uint64_t half, double* p0, double* p1) {
    extern __shared__ double sh[];
    double* s0 = sh;
    double* s1 = sh + blockDim.x;
    const dc* s = a + (uint64_t)blockIdx.x * d;
    double l0 = 0, l1 = 0;
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        l0 += cnorm(s[i]);
        l1 += cnorm(s[i + half]);
    }
    s0[threadIdx.x] = l0;
    s1[threadIdx.x] = l1;
    __syncthreads();
    for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            s0[threadIdx.x] += s0[threadIdx.x + st];
            s1[threadIdx.x] += s1[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p0[blockIdx.x] = s0[0];
        p1[blockIdx.x] = s1[0];
    }
}
// Per-shot-outcome-selected compact: outcome[b]==1 copies upper half down,
// outcome[b]==0 leaves the (already-live) lower half -- a per-lane predicate,
// not a divergent code path (mirrors clifft's keep-low/keep-high collapse).
__global__ void kb_compact_sel(dc* a, uint64_t d, uint64_t half, const unsigned char* outcome,
                               uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    if (outcome[b]) {
        dc* s = a + b * d;
        s[j] = s[j + half];
    }
}
// Per-shot-sign-selected interfere fold.
__global__ void kb_fold_sel(dc* a, uint64_t d, uint64_t half, const signed char* sign,
                            uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    dc* s = a + b * d;
    dc x = s[j], y = s[j + half];
    double sg = sign[b] >= 0 ? 1.0 : -1.0;
    s[j] = dc{x.r + sg * y.r, x.i + sg * y.i} * mb::kInvSqrt2;
}
// Per-shot expand (rank r -> r+1 within a fixed-stride slice; half = 2^r).
__global__ void kb_expand(dc* a, uint64_t d, uint64_t half, uint64_t total) {
    uint64_t g = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= total) return;
    uint64_t b = g / half, j = g % half;
    dc* s = a + b * d;
    s[half + j] = s[j];
}

// =============================================================================
// Arm 3: on-device measurement sampling. Arm 4: MIMD per-shot interpreter.
//
// hash_unit is a stateless counter-based RNG (splitmix64 finalizer), computable
// identically on host and device, so device-sampled outcomes are exactly
// reproducible for validation. clifft's per-shot RNG is equally cheap, so this
// models the real integration honestly.
// =============================================================================
__host__ __device__ inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
__host__ __device__ inline double hash_unit(uint64_t seed, uint64_t shot, uint64_t round) {
    return (double)(mix64(seed ^ mix64(shot + mix64(round))) >> 11) *
           (1.0 / 9007199254740992.0);
}

// Per-shot outcome sampled ON DEVICE from the reduced branch sums -- the
// clifft-cuda design lesson: no D2H/H2D legs, the host never touches
// mid-circuit state (kb_reduce_diag -> this -> kb_compact_sel -> kb_expand).
__global__ void kb_sample_dev(const double* p0, const double* p1, unsigned char* out, unsigned B,
                              uint64_t seed, uint64_t round) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    double tot = p0[b] + p1[b];
    out[b] = (unsigned char)((hash_unit(seed, b, round) * tot < p0[b]) ? 0 : 1);
}

// -----------------------------------------------------------------------------
// MIMD per-shot interpreter (the clifft-cuda architecture, in FP64): one BLOCK
// per shot walks the whole layered run -- gates, in-block reduction, in-block
// outcome sampling, collapse, re-expand -- in a single kernel launch with zero
// host involvement and zero cross-shot cooperation. Contrast arm: the lockstep
// SoA kernels above launch one grid-wide kernel per op. Same math, same
// initial state, same RNG, same schedule; only the execution architecture
// differs, so (mimdgpu vs batcheddevgpu) isolates the architecture choice.
// -----------------------------------------------------------------------------
__device__ void mimd_shot(dc* s, unsigned k, const ScheduledOp* ops, unsigned nops,
                          unsigned layers, uint64_t seed, int force_parity, unsigned shot,
                          double* red, const U2m& u2, const U4m& u4) {
    uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned o = 0; o < nops; ++o) {
            ScheduledOp sc = ops[o];
            unsigned lo = sc.a < sc.b ? sc.a : sc.b, hi = sc.a < sc.b ? sc.b : sc.a;
            switch (sc.op) {
                case Op::H:
                    for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) {
                        uint64_t i0 = dev_insert_zero_bit(j, sc.a);
                        uint64_t i1 = i0 | (uint64_t(1) << sc.a);
                        dc x = s[i0], y = s[i1];
                        s[i0] = (x + y) * mb::kInvSqrt2;
                        s[i1] = (x - y) * mb::kInvSqrt2;
                    }
                    break;
                case Op::T:
                    for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) {
                        uint64_t idx = dev_insert_zero_bit(j, sc.a) | (uint64_t(1) << sc.a);
                        s[idx] = cmul(s[idx], {kPhaseRe, kPhaseIm});
                    }
                    break;
                case Op::CZ:
                    for (uint64_t j = threadIdx.x; j < quarter; j += blockDim.x) {
                        uint64_t idx = dev_insert_two(j, lo, hi) | (uint64_t(1) << sc.a) |
                                       (uint64_t(1) << sc.b);
                        s[idx] = s[idx] * -1.0;
                    }
                    break;
                case Op::CNOT:
                    for (uint64_t j = threadIdx.x; j < quarter; j += blockDim.x) {
                        uint64_t b0 = dev_insert_two(j, lo, hi) | (uint64_t(1) << sc.a);
                        uint64_t b1 = b0 | (uint64_t(1) << sc.b);
                        dc tmp = s[b0];
                        s[b0] = s[b1];
                        s[b1] = tmp;
                    }
                    break;
                case Op::U2:
                    for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) {
                        uint64_t i0 = dev_insert_zero_bit(j, sc.a);
                        uint64_t i1 = i0 | (uint64_t(1) << sc.a);
                        dc x = s[i0], y = s[i1];
                        s[i0] = cmul(u2.m00, x) + cmul(u2.m01, y);
                        s[i1] = cmul(u2.m10, x) + cmul(u2.m11, y);
                    }
                    break;
                case Op::U4:
                    for (uint64_t j = threadIdx.x; j < quarter; j += blockDim.x) {
                        uint64_t base = dev_insert_two(j, lo, hi);
                        uint64_t i1 = base | (uint64_t(1) << lo);
                        uint64_t i2 = base | (uint64_t(1) << hi);
                        uint64_t i3 = i1 | (uint64_t(1) << hi);
                        dc in0 = s[base], in1 = s[i1], in2 = s[i2], in3 = s[i3];
                        s[base] = cmul(u4.m[0], in0) + cmul(u4.m[1], in1) + cmul(u4.m[2], in2) +
                                  cmul(u4.m[3], in3);
                        s[i1] = cmul(u4.m[4], in0) + cmul(u4.m[5], in1) + cmul(u4.m[6], in2) +
                                cmul(u4.m[7], in3);
                        s[i2] = cmul(u4.m[8], in0) + cmul(u4.m[9], in1) + cmul(u4.m[10], in2) +
                                cmul(u4.m[11], in3);
                        s[i3] = cmul(u4.m[12], in0) + cmul(u4.m[13], in1) + cmul(u4.m[14], in2) +
                                cmul(u4.m[15], in3);
                    }
                    break;
                default: break;
            }
            __syncthreads();
        }
        // Completed measurement round on the top axis: reduce, sample (or force
        // outcome = shot parity for validation), collapse, re-expand -- keeps
        // the rank, and the comparison with the SoA arm, fixed every layer.
        double l0 = 0, l1 = 0;
        for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) {
            l0 += cnorm(s[j]);
            l1 += cnorm(s[j + half]);
        }
        red[threadIdx.x] = l0;
        red[blockDim.x + threadIdx.x] = l1;
        __syncthreads();
        for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) {
                red[threadIdx.x] += red[threadIdx.x + st];
                red[blockDim.x + threadIdx.x] += red[blockDim.x + threadIdx.x + st];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            double p0 = red[0], p1 = red[blockDim.x];
            int b = (force_parity >= 0)
                        ? (int)(shot & 1u)
                        : ((hash_unit(seed, shot, l) * (p0 + p1) < p0) ? 0 : 1);
            red[0] = (double)b;
        }
        __syncthreads();
        int b = (int)red[0];
        __syncthreads();
        if (b) {
            for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) s[j] = s[j + half];
        }
        __syncthreads();
        for (uint64_t j = threadIdx.x; j < half; j += blockDim.x) s[half + j] = s[j];
        __syncthreads();
    }
}

// Global-memory variant: each block works its slice in place.
__global__ void k_mimd_global(dc* a, uint64_t d, unsigned k, const ScheduledOp* ops,
                              unsigned nops, unsigned layers, uint64_t seed, int force_parity,
                              U2m u2, U4m u4) {
    extern __shared__ double red[];  // 2 * blockDim
    mimd_shot(a + (uint64_t)blockIdx.x * d, k, ops, nops, layers, seed, force_parity,
              blockIdx.x, red, u2, u4);
}

// Shared-memory variant (the clifft-cuda coop design): the whole slice lives
// in dynamic shared memory; global traffic is one load and one store.
__global__ void k_mimd_shared(dc* a, uint64_t d, unsigned k, const ScheduledOp* ops,
                              unsigned nops, unsigned layers, uint64_t seed, int force_parity,
                              U2m u2, U4m u4) {
    extern __shared__ unsigned char shraw[];
    dc* sv = reinterpret_cast<dc*>(shraw);
    double* red = reinterpret_cast<double*>(shraw + (size_t)d * sizeof(dc));
    dc* g = a + (uint64_t)blockIdx.x * d;
    for (uint64_t j = threadIdx.x; j < d; j += blockDim.x) sv[j] = g[j];
    __syncthreads();
    mimd_shot(sv, k, ops, nops, layers, seed, force_parity, blockIdx.x, red, u2, u4);
    for (uint64_t j = threadIdx.x; j < d; j += blockDim.x) g[j] = sv[j];
}

// --- launch helpers ---------------------------------------------------------
static const int kTPB = 256;
static inline unsigned grid(uint64_t n) { return (unsigned)((n + kTPB - 1) / kTPB); }

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void start() { cudaEventRecord(a); }
    double stop_ms() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

// time a launch lambda adaptively (>= ~30ms total), return sec/call
template <class F>
static double time_kernel(F launch) {
    launch();
    CHECK(cudaDeviceSynchronize());
    long reps = 1;
    GpuTimer tm;
    for (;;) {
        tm.start();
        for (long i = 0; i < reps; ++i) launch();
        double ms = tm.stop_ms();
        if (ms > 30.0 || reps > (1L << 22)) return (ms / 1e3) / reps;
        reps *= 2;
    }
}

static void init_host(std::vector<dc>& h, uint64_t dim, Rng& rng) {
    double n = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        double re = rng.next_unit() * 2 - 1, im = rng.next_unit() * 2 - 1;
        h[i] = {re, im};
        n += re * re + im * im;
    }
    double inv = 1.0 / std::sqrt(n);
    for (uint64_t i = 0; i < dim; ++i) h[i] = h[i] * inv;
}

// Initialize shot b's slice [b*dim, b*dim+dim) of a batched host buffer.
static void init_host_slice(std::vector<dc>& h, unsigned b, uint64_t dim, Rng& rng) {
    dc* p = h.data() + (size_t)b * dim;
    double n = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        double re = rng.next_unit() * 2 - 1, im = rng.next_unit() * 2 - 1;
        p[i] = {re, im};
        n += re * re + im * im;
    }
    double inv = 1.0 / std::sqrt(n);
    for (uint64_t i = 0; i < dim; ++i) p[i] = p[i] * inv;
}

static U2m host_u2() {
    return U2m{{mb::kU2[0], mb::kU2[1]}, {mb::kU2[2], mb::kU2[3]},
               {mb::kU2[4], mb::kU2[5]}, {mb::kU2[6], mb::kU2[7]}};
}
static U4m host_u4() {
    U4m u;
    for (int i = 0; i < 16; ++i) u.m[i] = {mb::kU4[2 * i], mb::kU4[2 * i + 1]};
    return u;
}
static const cd* cpu_u2_mat() {
    static const cd m[4] = {cd(kU2[0], kU2[1]), cd(kU2[2], kU2[3]), cd(kU2[4], kU2[5]),
                            cd(kU2[6], kU2[7])};
    return m;
}
static const cd* cpu_u4_mat() {
    static cd m[16];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 16; ++i) m[i] = cd(kU4[2 * i], kU4[2 * i + 1]);
        init = true;
    }
    return m;
}

// Host-side reference application of one scheduled gate op (validation only).
static void apply_cpu_gate(cd* arr, unsigned k, const ScheduledOp& s) {
    switch (s.op) {
        case Op::H: op_h(arr, k, s.a); break;
        case Op::T: op_t(arr, k, s.a); break;
        case Op::CZ: op_cz(arr, k, s.a, s.b); break;
        case Op::CNOT: op_cnot(arr, k, s.a, s.b); break;
        case Op::U2: op_u2(arr, k, s.a, cpu_u2_mat()); break;
        case Op::U4: op_u4(arr, k, s.a, s.b, cpu_u4_mat()); break;
        default: break;
    }
}

// GPU-side launch of one scheduled gate op on a single statevector.
static void launch_gpu_gate(dc* d, uint64_t dim, unsigned /*k*/, const ScheduledOp& s) {
    uint64_t half = dim / 2, quarter = dim / 4;
    unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
    switch (s.op) {
        case Op::H: k_h<<<grid(half), kTPB>>>(d, half, s.a); break;
        case Op::T: k_t<<<grid(half), kTPB>>>(d, half, s.a); break;
        case Op::CZ: k_cz<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, s.a, s.b); break;
        case Op::CNOT: k_cnot<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, s.a, s.b); break;
        case Op::U2: k_u2<<<grid(half), kTPB>>>(d, half, s.a, host_u2()); break;
        case Op::U4: k_u4<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, host_u4()); break;
        default: break;
    }
}

static double max_abs_diff(const std::vector<dc>& g, const std::vector<cd>& c, uint64_t n) {
    double m = 0;
    for (uint64_t i = 0; i < n; ++i) {
        m = std::max(m, std::abs(g[i].r - c[i].real()));
        m = std::max(m, std::abs(g[i].i - c[i].imag()));
    }
    return m;
}

// Sum block partials on the host (helper for single-SV reduce cross-checks).
static double host_sum(const double* p, unsigned n) {
    double s = 0;
    for (unsigned i = 0; i < n; ++i) s += p[i];
    return s;
}

static void validate(unsigned k) {
    // Coverage: gate layers incl. U2/U4, EXPAND_T (rank k -> k+1), forced MEAS_DIAG
    // (k+1 -> k) and forced MEAS_INTERFERE (k -> k-1), with the GPU reduce
    // sums cross-checked against the CPU probabilities at both measurements.
    uint64_t dim = uint64_t(1) << k;
    uint64_t cap = dim * 2;  // room for the EXPAND_T
    std::vector<dc> h(cap);
    Rng rng(2024);
    init_host(h, dim, rng);
    std::vector<cd> hc(cap);
    for (uint64_t i = 0; i < dim; ++i) hc[i] = cd(h[i].r, h[i].i);

    dc* d = nullptr;
    CHECK(cudaMalloc(&d, cap * sizeof(dc)));
    CHECK(cudaMemcpy(d, h.data(), dim * sizeof(dc), cudaMemcpyHostToDevice));
    const int maxblocks = 4096;
    double* dp0 = nullptr;
    double* dp1 = nullptr;
    CHECK(cudaMalloc(&dp0, maxblocks * sizeof(double)));
    CHECK(cudaMalloc(&dp1, maxblocks * sizeof(double)));
    std::vector<double> hp0(maxblocks), hp1(maxblocks);
    size_t shmem = 2 * kTPB * sizeof(double);
    Rng crng(7);  // CPU meas rng (outcomes forced, so only consumed on parity)
    double worst_prob = 0;

    // 1) gate layers at rank k
    for (auto& s : make_layer_schedule(k, 2)) {
        launch_gpu_gate(d, dim, k, s);
        apply_cpu_gate(hc.data(), k, s);
    }
    // 2) EXPAND_T: rank k -> k+1
    k_expand_t<<<grid(dim), kTPB>>>(d, dim);
    op_expand_t(hc.data(), k);
    unsigned k1 = k + 1;
    uint64_t dim1 = dim * 2;
    // 3) one gate layer at rank k+1
    for (auto& s : make_layer_schedule(k1, 1)) {
        launch_gpu_gate(d, dim1, k1, s);
        apply_cpu_gate(hc.data(), k1, s);
    }
    // 4) MEAS_DIAG forced b=1 at rank k+1 (with reduce cross-check)
    {
        uint64_t half = dim1 / 2;
        unsigned nb = (unsigned)std::min<uint64_t>(maxblocks, grid(half));
        k_reduce_diag<<<nb, kTPB, shmem>>>(d, half, dp0, dp1);
        CHECK(cudaMemcpy(hp0.data(), dp0, nb * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hp1.data(), dp1, nb * sizeof(double), cudaMemcpyDeviceToHost));
        double g0 = host_sum(hp0.data(), nb), g1 = host_sum(hp1.data(), nb);
        double c0 = 0, c1 = 0;
        for (uint64_t i = 0; i < half; ++i) {
            c0 += std::norm(hc[i]);
            c1 += std::norm(hc[i + half]);
        }
        worst_prob = std::max(worst_prob, std::abs(g0 - c0) + std::abs(g1 - c1));
        k_compact_upper<<<grid(half), kTPB>>>(d, half);
        op_meas_diag(hc.data(), k1, crng, /*force_bit=*/1);
    }
    // 5) MEAS_INTERFERE forced sign -1 at rank k (with reduce cross-check)
    {
        uint64_t half = dim / 2;
        unsigned nb = (unsigned)std::min<uint64_t>(maxblocks, grid(half));
        k_reduce_interfere<<<nb, kTPB, shmem>>>(d, half, dp0, dp1);
        CHECK(cudaMemcpy(hp0.data(), dp0, nb * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hp1.data(), dp1, nb * sizeof(double), cudaMemcpyDeviceToHost));
        double gp = host_sum(hp0.data(), nb), gm = host_sum(hp1.data(), nb);
        double cp = 0, cm = 0;
        for (uint64_t i = 0; i < half; ++i) {
            cp += std::norm(hc[i] + hc[i + half]);
            cm += std::norm(hc[i] - hc[i + half]);
        }
        worst_prob = std::max(worst_prob, std::abs(gp - cp) + std::abs(gm - cm));
        k_fold<<<grid(half), kTPB>>>(d, half, -1);
        op_meas_interfere(hc.data(), k, crng, /*force_bit=*/1);  // b=1 => minus fold
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h.data(), d, (dim / 2) * sizeof(dc), cudaMemcpyDeviceToHost));
    double maxerr = max_abs_diff(h, hc, dim / 2);
    cudaFree(dp0);
    cudaFree(dp1);
    cudaFree(d);
    fprintf(stderr,
            "[validate] k=%u  gates+U2/U4+expand+meas: max|cpu-gpu| = %.3e, reduce-sum err = "
            "%.3e  %s\n",
            k, maxerr, worst_prob, (maxerr < 1e-9 && worst_prob < 1e-9) ? "OK" : "FAIL");
}

static void validate_batched(unsigned k, unsigned B) {
    // Batched kernels + one COMPLETED batched measurement round vs per-shot
    // CPU reference. Outcomes forced to (shot & 1) so both predicate arms of
    // kb_compact_sel are exercised deterministically; probabilities from
    // kb_reduce_diag are cross-checked per shot.
    uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
    std::vector<dc> h((size_t)B * dim);
    Rng rng(4242);
    for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
    std::vector<cd> hc((size_t)B * dim);
    for (size_t i = 0; i < h.size(); ++i) hc[i] = cd(h[i].r, h[i].i);

    dc* d = nullptr;
    CHECK(cudaMalloc(&d, (size_t)B * dim * sizeof(dc)));
    CHECK(cudaMemcpy(d, h.data(), (size_t)B * dim * sizeof(dc), cudaMemcpyHostToDevice));
    uint64_t totH = (uint64_t)B * half, totQ = (uint64_t)B * quarter;

    auto sched = make_layer_schedule(k, 2);
    for (auto& s : sched) {
        unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
        switch (s.op) {
            case Op::H: kb_h<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
            case Op::T: kb_t<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
            case Op::CZ: kb_cz<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ); break;
            case Op::CNOT:
                kb_cnot<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                break;
            case Op::U2: kb_u2<<<grid(totH), kTPB>>>(d, dim, half, s.a, host_u2(), totH); break;
            case Op::U4:
                kb_u4<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, host_u4(), totQ);
                break;
            default: break;
        }
    }
    // completed batched measurement round: reduce -> D2H -> outcomes -> H2D
    // -> selected compact -> per-shot expand (restores rank)
    double* dp0 = nullptr;
    double* dp1 = nullptr;
    unsigned char* dout = nullptr;
    CHECK(cudaMalloc(&dp0, B * sizeof(double)));
    CHECK(cudaMalloc(&dp1, B * sizeof(double)));
    CHECK(cudaMalloc(&dout, B));
    std::vector<double> hp0(B), hp1(B);
    std::vector<unsigned char> hout(B);
    size_t shmem = 2 * kTPB * sizeof(double);
    kb_reduce_diag<<<B, kTPB, shmem>>>(d, dim, half, dp0, dp1);
    CHECK(cudaMemcpy(hp0.data(), dp0, B * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hp1.data(), dp1, B * sizeof(double), cudaMemcpyDeviceToHost));
    for (unsigned b = 0; b < B; ++b) hout[b] = (unsigned char)(b & 1);
    CHECK(cudaMemcpy(dout, hout.data(), B, cudaMemcpyHostToDevice));
    kb_compact_sel<<<grid(totH), kTPB>>>(d, dim, half, dout, totH);
    // expand back: rank k-1 -> k within each slice
    uint64_t totE = (uint64_t)B * half;
    kb_expand<<<grid(totE), kTPB>>>(d, dim, half, totE);
    // break the expand's half-symmetry (else the minus fold is trivially 0)
    kb_t<<<grid(totH), kTPB>>>(d, dim, half, k - 1, totH);
    // and one per-shot-sign interfere fold (rank k -> k-1), signs alternating
    // so both arms of kb_fold_sel are exercised
    signed char* dsign = nullptr;
    CHECK(cudaMalloc(&dsign, B));
    std::vector<signed char> hsign(B);
    for (unsigned b = 0; b < B; ++b) hsign[b] = (b & 1) ? -1 : +1;
    CHECK(cudaMemcpy(dsign, hsign.data(), B, cudaMemcpyHostToDevice));
    kb_fold_sel<<<grid(totH), kTPB>>>(d, dim, half, dsign, totH);
    CHECK(cudaDeviceSynchronize());

    // CPU reference, per shot
    double worst_prob = 0;
    Rng crng(5);
    for (unsigned b = 0; b < B; ++b) {
        cd* slice = hc.data() + (size_t)b * dim;
        for (auto& s : sched) apply_cpu_gate(slice, k, s);
        double c0 = 0, c1 = 0;
        for (uint64_t i = 0; i < half; ++i) {
            c0 += std::norm(slice[i]);
            c1 += std::norm(slice[i + half]);
        }
        worst_prob = std::max(worst_prob, std::abs(hp0[b] - c0) + std::abs(hp1[b] - c1));
        op_meas_diag(slice, k, crng, /*force_bit=*/(int)(b & 1));
        op_expand(slice, k - 1);
        op_t(slice, k, k - 1);  // mirror the symmetry-breaking T
        // CPU interfere fold: force_bit 0 = plus (sign +1), 1 = minus (sign -1)
        op_meas_interfere(slice, k, crng, /*force_bit=*/(int)(b & 1));
    }
    CHECK(cudaMemcpy(h.data(), d, (size_t)B * dim * sizeof(dc), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (unsigned b = 0; b < B; ++b)
        for (uint64_t i = 0; i < half; ++i) {  // live prefix after the fold
            size_t idx = (size_t)b * dim + i;
            maxerr = std::max(maxerr, std::abs(h[idx].r - hc[idx].real()));
            maxerr = std::max(maxerr, std::abs(h[idx].i - hc[idx].imag()));
        }
    cudaFree(dp0);
    cudaFree(dp1);
    cudaFree(dout);
    cudaFree(dsign);
    cudaFree(d);
    fprintf(stderr,
            "[validate-batched] k=%u B=%u  kb_* + completed meas round: max|cpu-gpu| = %.3e, "
            "per-shot prob err = %.3e  %s\n",
            k, B, maxerr, worst_prob, (maxerr < 1e-9 && worst_prob < 1e-9) ? "OK" : "FAIL");
}

static void validate_devmeas(unsigned k, unsigned B) {
    // Arm-3 validation: gate layers, then ONE measurement round with the
    // outcome sampled ON DEVICE. Checks (a) the device-sampled outcomes
    // reproduce the host recomputation from the device-reduced sums exactly
    // (same doubles, same stateless hash -- bit-for-bit), (b) per-shot branch
    // sums match the CPU reference, (c) the collapsed+re-expanded state
    // matches the CPU reference forced to the device's outcomes.
    uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
    const uint64_t kSeed = 0xDEC0DEull;
    std::vector<dc> h((size_t)B * dim);
    Rng rng(777);
    for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
    std::vector<cd> hc((size_t)B * dim);
    for (size_t i = 0; i < h.size(); ++i) hc[i] = cd(h[i].r, h[i].i);

    dc* d = nullptr;
    CHECK(cudaMalloc(&d, (size_t)B * dim * sizeof(dc)));
    CHECK(cudaMemcpy(d, h.data(), (size_t)B * dim * sizeof(dc), cudaMemcpyHostToDevice));
    uint64_t totH = (uint64_t)B * half, totQ = (uint64_t)B * quarter;

    auto sched = make_layer_schedule(k, 2);
    for (auto& s : sched) {
        unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
        switch (s.op) {
            case Op::H: kb_h<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
            case Op::T: kb_t<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
            case Op::CZ: kb_cz<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ); break;
            case Op::CNOT:
                kb_cnot<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                break;
            case Op::U2: kb_u2<<<grid(totH), kTPB>>>(d, dim, half, s.a, host_u2(), totH); break;
            case Op::U4:
                kb_u4<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, host_u4(), totQ);
                break;
            default: break;
        }
    }
    double* dp0 = nullptr;
    double* dp1 = nullptr;
    unsigned char* dout = nullptr;
    CHECK(cudaMalloc(&dp0, B * sizeof(double)));
    CHECK(cudaMalloc(&dp1, B * sizeof(double)));
    CHECK(cudaMalloc(&dout, B));
    size_t shmem = 2 * kTPB * sizeof(double);
    kb_reduce_diag<<<B, kTPB, shmem>>>(d, dim, half, dp0, dp1);
    kb_sample_dev<<<grid(B), kTPB>>>(dp0, dp1, dout, B, kSeed, /*round=*/0);
    std::vector<double> hp0(B), hp1(B);
    std::vector<unsigned char> hout(B);
    CHECK(cudaMemcpy(hp0.data(), dp0, B * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hp1.data(), dp1, B * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hout.data(), dout, B, cudaMemcpyDeviceToHost));
    unsigned rng_mismatch = 0;
    for (unsigned b = 0; b < B; ++b) {
        double tot = hp0[b] + hp1[b];
        unsigned char expect = (unsigned char)((hash_unit(kSeed, b, 0) * tot < hp0[b]) ? 0 : 1);
        if (expect != hout[b]) ++rng_mismatch;
    }
    kb_compact_sel<<<grid(totH), kTPB>>>(d, dim, half, dout, totH);
    kb_expand<<<grid(totH), kTPB>>>(d, dim, half, totH);
    CHECK(cudaDeviceSynchronize());

    // CPU reference forced to the device's outcomes.
    double worst_prob = 0;
    Rng crng(5);
    for (unsigned b = 0; b < B; ++b) {
        cd* slice = hc.data() + (size_t)b * dim;
        for (auto& s : sched) apply_cpu_gate(slice, k, s);
        double c0 = 0, c1 = 0;
        for (uint64_t i = 0; i < half; ++i) {
            c0 += std::norm(slice[i]);
            c1 += std::norm(slice[i + half]);
        }
        worst_prob = std::max(worst_prob, std::abs(hp0[b] - c0) + std::abs(hp1[b] - c1));
        op_meas_diag(slice, k, crng, /*force_bit=*/(int)hout[b]);
        op_expand(slice, k - 1);
    }
    CHECK(cudaMemcpy(h.data(), d, (size_t)B * dim * sizeof(dc), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        maxerr = std::max(maxerr, std::abs(h[i].r - hc[i].real()));
        maxerr = std::max(maxerr, std::abs(h[i].i - hc[i].imag()));
    }
    cudaFree(dp0);
    cudaFree(dp1);
    cudaFree(dout);
    cudaFree(d);
    fprintf(stderr,
            "[validate-devmeas] k=%u B=%u  on-device sampling: rng mismatches = %u, "
            "max|cpu-gpu| = %.3e, prob err = %.3e  %s\n",
            k, B, rng_mismatch, maxerr, worst_prob,
            (rng_mismatch == 0 && maxerr < 1e-9 && worst_prob < 1e-9) ? "OK" : "FAIL");
}

static void validate_mimd(unsigned k, unsigned B, unsigned layers) {
    // Arm-4 validation: the MIMD interpreter (global variant, plus the shared
    // variant when the slice fits) with outcomes forced to shot parity, vs the
    // identical per-shot CPU trajectory. Exercises both compact arms.
    uint64_t dim = uint64_t(1) << k;
    auto sched1 = make_layer_schedule(k, 1);
    std::vector<dc> h0((size_t)B * dim);
    Rng rng(1313);
    for (unsigned b = 0; b < B; ++b) init_host_slice(h0, b, dim, rng);

    // CPU reference trajectory (shared by both variants).
    std::vector<cd> hc((size_t)B * dim);
    for (size_t i = 0; i < h0.size(); ++i) hc[i] = cd(h0[i].r, h0[i].i);
    Rng crng(5);
    for (unsigned b = 0; b < B; ++b) {
        cd* slice = hc.data() + (size_t)b * dim;
        for (unsigned l = 0; l < layers; ++l) {
            for (auto& s : sched1) apply_cpu_gate(slice, k, s);
            op_meas_diag(slice, k, crng, /*force_bit=*/(int)(b & 1));
            op_expand(slice, k - 1);
        }
    }

    ScheduledOp* dops = nullptr;
    CHECK(cudaMalloc(&dops, sched1.size() * sizeof(ScheduledOp)));
    CHECK(cudaMemcpy(dops, sched1.data(), sched1.size() * sizeof(ScheduledOp),
                     cudaMemcpyHostToDevice));
    dc* d = nullptr;
    CHECK(cudaMalloc(&d, (size_t)B * dim * sizeof(dc)));
    std::vector<dc> h((size_t)B * dim);

    size_t red_bytes = 2 * kTPB * sizeof(double);
    size_t sh_bytes = (size_t)dim * sizeof(dc) + red_bytes;
    int sh_optin = 0;
    CHECK(cudaDeviceGetAttribute(&sh_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    bool shared_fits = sh_bytes <= (size_t)sh_optin;

    for (int variant = 0; variant < (shared_fits ? 2 : 1); ++variant) {
        CHECK(cudaMemcpy(d, h0.data(), (size_t)B * dim * sizeof(dc), cudaMemcpyHostToDevice));
        if (variant == 0) {
            k_mimd_global<<<B, kTPB, red_bytes>>>(d, dim, k, dops, (unsigned)sched1.size(),
                                                  layers, 0, /*force_parity=*/1, host_u2(),
                                                  host_u4());
        } else {
            CHECK(cudaFuncSetAttribute(k_mimd_shared,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)sh_bytes));
            k_mimd_shared<<<B, kTPB, sh_bytes>>>(d, dim, k, dops, (unsigned)sched1.size(),
                                                 layers, 0, /*force_parity=*/1, host_u2(),
                                                 host_u4());
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h.data(), d, (size_t)B * dim * sizeof(dc), cudaMemcpyDeviceToHost));
        double maxerr = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            maxerr = std::max(maxerr, std::abs(h[i].r - hc[i].real()));
            maxerr = std::max(maxerr, std::abs(h[i].i - hc[i].imag()));
        }
        fprintf(stderr, "[validate-mimd] k=%u B=%u layers=%u %s: max|cpu-gpu| = %.3e  %s\n", k,
                B, layers, variant == 0 ? "global" : "shared", maxerr,
                maxerr < 1e-9 ? "OK" : "FAIL");
    }
    if (!shared_fits)
        fprintf(stderr, "[validate-mimd] k=%u shared variant skipped (needs %zu B > opt-in %d B)\n",
                k, sh_bytes, sh_optin);
    cudaFree(dops);
    cudaFree(d);
}

static void transfer_cost(unsigned k) {
    uint64_t dim = uint64_t(1) << k;
    size_t bytes = dim * sizeof(dc);
    std::vector<dc> h(dim);
    Rng rng(1);
    init_host(h, dim, rng);
    dc* d = nullptr;
    CHECK(cudaMalloc(&d, bytes));
    GpuTimer tm;
    int reps = 20;
    tm.start();
    for (int i = 0; i < reps; ++i)
        cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice);
    double h2d = tm.stop_ms() / reps / 1e3;
    tm.start();
    for (int i = 0; i < reps; ++i)
        cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost);
    double d2h = tm.stop_ms() / reps / 1e3;
    cudaFree(d);
    printf("transfer,%u,%llu,%.3f,%.3f\n", k, (unsigned long long)bytes, bytes / h2d / 1e9,
           bytes / d2h / 1e9);
    fprintf(stderr, "[transfer] k=%u  %.1f MB  H2D %.1f GB/s  D2H %.1f GB/s\n", k,
            bytes / 1e6, bytes / h2d / 1e9, bytes / d2h / 1e9);
}

static void bench_single(unsigned kmin, unsigned kmax) {
    fprintf(stderr, "\n=== Single-statevector per-op throughput (GPU, FP64) ===\n");
    fprintf(stderr, "%-4s %-15s %-12s %-12s %-10s\n", "k", "op", "dim", "ns/op", "Gamp/s");
    uint64_t cap = uint64_t(1) << (kmax + 1);
    dc* d = nullptr;
    CHECK(cudaMalloc(&d, cap * sizeof(dc)));
    {
        std::vector<dc> h(cap);
        Rng rng(99);
        init_host(h, cap, rng);
        CHECK(cudaMemcpy(d, h.data(), cap * sizeof(dc), cudaMemcpyHostToDevice));
    }
    double* part0 = nullptr;
    double* part1 = nullptr;
    int maxblocks = 4096;
    CHECK(cudaMalloc(&part0, maxblocks * sizeof(double)));
    CHECK(cudaMalloc(&part1, maxblocks * sizeof(double)));
    std::vector<double> hpart0(maxblocks), hpart1(maxblocks);

    for (unsigned k = kmin; k <= kmax; ++k) {
        uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
        unsigned v = k / 2, c = 0, t = (k >= 2) ? 1u : 0u;
        unsigned lo = c < t ? c : t, hi = c < t ? t : c;

        struct Row { Op op; double spc; };
        std::vector<Row> rows;
        rows.push_back({Op::H, time_kernel([&] { k_h<<<grid(half), kTPB>>>(d, half, v); })});
        rows.push_back({Op::T, time_kernel([&] { k_t<<<grid(half), kTPB>>>(d, half, v); })});
        if (k >= 2) {
            rows.push_back({Op::CZ, time_kernel([&] {
                                k_cz<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, c, t);
                            })});
            rows.push_back({Op::CNOT, time_kernel([&] {
                                k_cnot<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, c, t);
                            })});
        }
        rows.push_back({Op::U2, time_kernel([&] { k_u2<<<grid(half), kTPB>>>(d, half, v, host_u2()); })});
        if (k >= 2) {
            rows.push_back({Op::U4, time_kernel([&] {
                                k_u4<<<grid(quarter), kTPB>>>(d, quarter, lo, hi, host_u4());
                            })});
        }
        rows.push_back({Op::EXPAND, time_kernel([&] { k_expand<<<grid(dim), kTPB>>>(d, dim); })});
        rows.push_back(
            {Op::EXPAND_T, time_kernel([&] { k_expand_t<<<grid(dim), kTPB>>>(d, dim); })});
        unsigned nb = (unsigned)std::min<uint64_t>(maxblocks, grid(half));
        size_t shmem = 2 * kTPB * sizeof(double);
        rows.push_back({Op::MEAS_DIAG, time_kernel([&] {
                            k_reduce_diag<<<nb, kTPB, shmem>>>(d, half, part0, part1);
                            k_compact_upper<<<grid(half), kTPB>>>(d, half);
                        })});
        rows.push_back({Op::MEAS_INTERFERE, time_kernel([&] {
                            k_reduce_interfere<<<nb, kTPB, shmem>>>(d, half, part0, part1);
                            k_fold<<<grid(half), kTPB>>>(d, half, +1);
                        })});

        for (auto& r : rows) {
            double gamps = (double)amps_touched(r.op, dim) / r.spc / 1e9;
            printf("singlegpu,%u,%s,%llu,%.1f,%.3f\n", k, op_name(r.op),
                   (unsigned long long)dim, r.spc * 1e9, gamps);
            fprintf(stderr, "%-4u %-15s %-12llu %-12.1f %-10.3f\n", k, op_name(r.op),
                    (unsigned long long)dim, r.spc * 1e9, gamps);
        }
    }
    cudaFree(part0);
    cudaFree(part1);
    cudaFree(d);
}

static void bench_batched(const std::vector<unsigned>& ks, unsigned layers) {
    fprintf(stderr, "\n=== Batched-across-shots throughput (GPU, FP64) ===\n");
    fprintf(stderr, "%-4s %-8s %-14s %-12s\n", "k", "B", "shots/s", "Mshot-lyr/s");
    std::vector<unsigned> Bs = {256, 1024, 4096, 16384, 65536};
    for (unsigned k : ks) {
        uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
        auto sched = make_layer_schedule(k, layers);
        for (unsigned B : Bs) {
            size_t bytes = (size_t)B * dim * sizeof(dc);
            if (bytes > 16.0e9) continue;  // keep within GH200 HBM headroom
            dc* d = nullptr;
            if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
            {
                std::vector<dc> h((size_t)B * dim);
                Rng rng(31 + k);
                for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
                CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
            }
            uint64_t totH = (uint64_t)B * half, totQ = (uint64_t)B * quarter;
            double spc = time_kernel([&] {
                for (auto& s : sched) {
                    unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
                    switch (s.op) {
                        case Op::H: kb_h<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                        case Op::T: kb_t<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                        case Op::CZ:
                            kb_cz<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                            break;
                        case Op::CNOT:
                            kb_cnot<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                            break;
                        case Op::U2:
                            kb_u2<<<grid(totH), kTPB>>>(d, dim, half, s.a, host_u2(), totH);
                            break;
                        case Op::U4:
                            kb_u4<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, host_u4(), totQ);
                            break;
                        default: break;
                    }
                }
            });
            cudaFree(d);
            double shots_per_s = B / spc;
            double mshot_layer = shots_per_s * layers / 1e6;
            printf("batchedgpu,%u,%u,%u,%.1f,%.3f\n", k, B, B, shots_per_s, mshot_layer);
            fprintf(stderr, "%-4u %-8u %-14.1f %-12.3f\n", k, B, shots_per_s, mshot_layer);
        }
    }
}

// The decisive go/no-go number: batched throughput with one COMPLETED
// measurement per layer -- reduce, D2H of per-shot probabilities, host
// sampling, H2D of outcomes, outcome-selected compact, per-shot expand (rank
// restored each layer, mirroring clifft's static k trajectory). Timed with
// wall-clock including every sync (cuda events would hide the host legs).
// Also times one gate-free measurement round alone (tag `measround`).
static void bench_batched_meas(const std::vector<unsigned>& ks, unsigned layers) {
    fprintf(stderr, "\n=== Batched shots WITH completed measurement per layer (GPU) ===\n");
    fprintf(stderr, "%-4s %-8s %-14s %-14s\n", "k", "B", "shots/s", "us/meas-round");
    std::vector<unsigned> Bs = {256, 1024, 4096, 16384, 65536};
    for (unsigned k : ks) {
        uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
        auto gates = make_layer_schedule(k, 1);
        for (unsigned B : Bs) {
            size_t bytes = (size_t)B * dim * sizeof(dc);
            if (bytes > 16.0e9) continue;
            dc* d = nullptr;
            if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
            {
                std::vector<dc> h((size_t)B * dim);
                Rng rng(53 + k);
                for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
                CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
            }
            double* dp0 = nullptr;
            double* dp1 = nullptr;
            unsigned char* dout = nullptr;
            CHECK(cudaMalloc(&dp0, B * sizeof(double)));
            CHECK(cudaMalloc(&dp1, B * sizeof(double)));
            CHECK(cudaMalloc(&dout, B));
            // pinned host staging for the per-round D2H/H2D legs
            double* hp0 = nullptr;
            double* hp1 = nullptr;
            unsigned char* hout = nullptr;
            CHECK(cudaHostAlloc(&hp0, B * sizeof(double), cudaHostAllocDefault));
            CHECK(cudaHostAlloc(&hp1, B * sizeof(double), cudaHostAllocDefault));
            CHECK(cudaHostAlloc(&hout, B, cudaHostAllocDefault));
            uint64_t totH = (uint64_t)B * half, totQ = (uint64_t)B * quarter;
            size_t shmem = 2 * kTPB * sizeof(double);
            Rng srng(0xC0FFEE ^ k);

            auto meas_round = [&] {
                kb_reduce_diag<<<B, kTPB, shmem>>>(d, dim, half, dp0, dp1);
                CHECK(cudaMemcpy(hp0, dp0, B * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hp1, dp1, B * sizeof(double), cudaMemcpyDeviceToHost));
                for (unsigned b = 0; b < B; ++b) {
                    double tot = hp0[b] + hp1[b];
                    hout[b] = (unsigned char)((srng.next_unit() * tot < hp0[b]) ? 0 : 1);
                }
                CHECK(cudaMemcpy(dout, hout, B, cudaMemcpyHostToDevice));
                kb_compact_sel<<<grid(totH), kTPB>>>(d, dim, half, dout, totH);
                kb_expand<<<grid(totH), kTPB>>>(d, dim, half, totH);
            };
            auto run_layer_batch = [&] {
                for (unsigned l = 0; l < layers; ++l) {
                    for (auto& s : gates) {
                        unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
                        switch (s.op) {
                            case Op::H: kb_h<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                            case Op::T: kb_t<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                            case Op::CZ:
                                kb_cz<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                                break;
                            case Op::CNOT:
                                kb_cnot<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b,
                                                              totQ);
                                break;
                            case Op::U2:
                                kb_u2<<<grid(totH), kTPB>>>(d, dim, half, s.a, host_u2(), totH);
                                break;
                            case Op::U4:
                                kb_u4<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, host_u4(),
                                                            totQ);
                                break;
                            default: break;
                        }
                    }
                    meas_round();
                }
                CHECK(cudaDeviceSynchronize());
            };

            // (a) whole-batch throughput with measurements
            run_layer_batch();  // warmup
            std::vector<double> samples;
            for (int r = 0; r < 5; ++r) {
                auto t0 = Clock::now();
                run_layer_batch();
                samples.push_back(seconds_since(t0));
            }
            double sec = median(samples);
            double shots_per_s = B / sec;
            // (b) one gate-free measurement round alone (latency of the
            // reduce -> D2H -> sample -> H2D -> collapse chain)
            meas_round();
            CHECK(cudaDeviceSynchronize());
            std::vector<double> ms;
            for (int r = 0; r < 20; ++r) {
                auto t0 = Clock::now();
                meas_round();
                CHECK(cudaDeviceSynchronize());
                ms.push_back(seconds_since(t0));
            }
            double us_round = median(ms) * 1e6;
            printf("batchedmeasgpu,%u,%u,%u,%.1f,%.3f\n", k, B, B, shots_per_s, us_round);
            printf("measround,%u,%u,%u,%.3f,%.3f\n", k, B, B, us_round, us_round / B);
            fprintf(stderr, "%-4u %-8u %-14.1f %-14.2f\n", k, B, shots_per_s, us_round);
            cudaFreeHost(hp0);
            cudaFreeHost(hp1);
            cudaFreeHost(hout);
            cudaFree(dp0);
            cudaFree(dp1);
            cudaFree(dout);
            cudaFree(d);
        }
    }
}

// Arm 3: batched shots with the measurement outcome sampled ON DEVICE -- the
// same schedule as bench_batched_meas but with zero host legs; the whole
// layered run is an uninterrupted kernel stream with one final sync. The
// (batchedmeasgpu vs batcheddevgpu) delta prices the host round-trip itself.
static void bench_batched_meas_dev(const std::vector<unsigned>& ks, unsigned layers) {
    fprintf(stderr, "\n=== Batched shots, measurement sampled on device (GPU) ===\n");
    fprintf(stderr, "%-4s %-8s %-14s %-14s\n", "k", "B", "shots/s", "us/meas-round");
    std::vector<unsigned> Bs = {256, 1024, 4096, 16384, 65536};
    for (unsigned k : ks) {
        uint64_t dim = uint64_t(1) << k, half = dim / 2, quarter = dim / 4;
        auto gates = make_layer_schedule(k, 1);
        for (unsigned B : Bs) {
            size_t bytes = (size_t)B * dim * sizeof(dc);
            if (bytes > 16.0e9) continue;
            dc* d = nullptr;
            if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
            {
                std::vector<dc> h((size_t)B * dim);
                Rng rng(67 + k);
                for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
                CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
            }
            double* dp0 = nullptr;
            double* dp1 = nullptr;
            unsigned char* dout = nullptr;
            CHECK(cudaMalloc(&dp0, B * sizeof(double)));
            CHECK(cudaMalloc(&dp1, B * sizeof(double)));
            CHECK(cudaMalloc(&dout, B));
            uint64_t totH = (uint64_t)B * half, totQ = (uint64_t)B * quarter;
            size_t shmem = 2 * kTPB * sizeof(double);
            uint64_t seed = 0xC0FFEEull ^ k;

            auto meas_round_dev = [&](unsigned round) {
                kb_reduce_diag<<<B, kTPB, shmem>>>(d, dim, half, dp0, dp1);
                kb_sample_dev<<<grid(B), kTPB>>>(dp0, dp1, dout, B, seed, round);
                kb_compact_sel<<<grid(totH), kTPB>>>(d, dim, half, dout, totH);
                kb_expand<<<grid(totH), kTPB>>>(d, dim, half, totH);
            };
            auto run_layer_batch = [&] {
                for (unsigned l = 0; l < layers; ++l) {
                    for (auto& s : gates) {
                        unsigned lo = s.a < s.b ? s.a : s.b, hi = s.a < s.b ? s.b : s.a;
                        switch (s.op) {
                            case Op::H: kb_h<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                            case Op::T: kb_t<<<grid(totH), kTPB>>>(d, dim, half, s.a, totH); break;
                            case Op::CZ:
                                kb_cz<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b, totQ);
                                break;
                            case Op::CNOT:
                                kb_cnot<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, s.a, s.b,
                                                              totQ);
                                break;
                            case Op::U2:
                                kb_u2<<<grid(totH), kTPB>>>(d, dim, half, s.a, host_u2(), totH);
                                break;
                            case Op::U4:
                                kb_u4<<<grid(totQ), kTPB>>>(d, dim, quarter, lo, hi, host_u4(),
                                                            totQ);
                                break;
                            default: break;
                        }
                    }
                    meas_round_dev(l);
                }
                CHECK(cudaDeviceSynchronize());
            };

            run_layer_batch();  // warmup
            std::vector<double> samples;
            for (int r = 0; r < 5; ++r) {
                auto t0 = Clock::now();
                run_layer_batch();
                samples.push_back(seconds_since(t0));
            }
            double sec = median(samples);
            double shots_per_s = B / sec;
            // Gate-free device measurement round. The per-round sync makes this
            // a LATENCY number (the stream never syncs mid-run in real use, so
            // the whole-batch figure above is the honest throughput).
            meas_round_dev(0);
            CHECK(cudaDeviceSynchronize());
            std::vector<double> ms;
            for (int r = 0; r < 20; ++r) {
                auto t0 = Clock::now();
                meas_round_dev(r);
                CHECK(cudaDeviceSynchronize());
                ms.push_back(seconds_since(t0));
            }
            double us_round = median(ms) * 1e6;
            printf("batcheddevgpu,%u,%u,%u,%.1f,%.3f\n", k, B, B, shots_per_s, us_round);
            printf("measrounddev,%u,%u,%u,%.3f,%.3f\n", k, B, B, us_round, us_round / B);
            fprintf(stderr, "%-4u %-8u %-14.1f %-14.2f\n", k, B, shots_per_s, us_round);
            cudaFree(dp0);
            cudaFree(dp1);
            cudaFree(dout);
            cudaFree(d);
        }
    }
}

// Arm 4: the MIMD per-shot interpreter (clifft-cuda's architecture) on the
// SAME schedule, state, and RNG as arm 3 -- one kernel launch for the whole
// layered run, one block per shot. metric2: 1 = shared-memory variant.
static void bench_mimd(const std::vector<unsigned>& ks, unsigned layers) {
    fprintf(stderr, "\n=== MIMD per-shot interpreter (GPU, one block/shot) ===\n");
    fprintf(stderr, "%-4s %-8s %-14s %-8s\n", "k", "B", "shots/s", "variant");
    std::vector<unsigned> Bs = {256, 1024, 4096, 16384, 65536};
    int sh_optin = 0;
    CHECK(cudaDeviceGetAttribute(&sh_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    for (unsigned k : ks) {
        uint64_t dim = uint64_t(1) << k;
        auto sched1 = make_layer_schedule(k, 1);
        ScheduledOp* dops = nullptr;
        CHECK(cudaMalloc(&dops, sched1.size() * sizeof(ScheduledOp)));
        CHECK(cudaMemcpy(dops, sched1.data(), sched1.size() * sizeof(ScheduledOp),
                         cudaMemcpyHostToDevice));
        size_t red_bytes = 2 * kTPB * sizeof(double);
        size_t sh_bytes = (size_t)dim * sizeof(dc) + red_bytes;
        bool use_shared = sh_bytes <= (size_t)sh_optin;
        if (use_shared) {
            CHECK(cudaFuncSetAttribute(k_mimd_shared,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)sh_bytes));
        }
        for (unsigned B : Bs) {
            size_t bytes = (size_t)B * dim * sizeof(dc);
            if (bytes > 16.0e9) continue;
            dc* d = nullptr;
            if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
            {
                std::vector<dc> h((size_t)B * dim);
                Rng rng(89 + k);
                for (unsigned b = 0; b < B; ++b) init_host_slice(h, b, dim, rng);
                CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
            }
            uint64_t seed = 0xC0FFEEull ^ k;
            auto run = [&] {
                if (use_shared) {
                    k_mimd_shared<<<B, kTPB, sh_bytes>>>(d, dim, k, dops,
                                                         (unsigned)sched1.size(), layers, seed,
                                                         /*force_parity=*/-1, host_u2(),
                                                         host_u4());
                } else {
                    k_mimd_global<<<B, kTPB, red_bytes>>>(d, dim, k, dops,
                                                          (unsigned)sched1.size(), layers, seed,
                                                          /*force_parity=*/-1, host_u2(),
                                                          host_u4());
                }
                CHECK(cudaDeviceSynchronize());
            };
            run();  // warmup
            std::vector<double> samples;
            for (int r = 0; r < 5; ++r) {
                auto t0 = Clock::now();
                run();
                samples.push_back(seconds_since(t0));
            }
            double sec = median(samples);
            double shots_per_s = B / sec;
            printf("mimdgpu,%u,%u,%u,%.1f,%d\n", k, B, B, shots_per_s, use_shared ? 1 : 0);
            fprintf(stderr, "%-4u %-8u %-14.1f %-8s\n", k, B, shots_per_s,
                    use_shared ? "shared" : "global");
            cudaFree(d);
        }
        cudaFree(dops);
    }
}

int main(int argc, char** argv) {
    unsigned kmin = 10, kmax = 26, layers = 4;
    std::vector<unsigned> bks = {12, 16, 18, 20};
    if (argc >= 3) { kmin = atoi(argv[1]); kmax = atoi(argv[2]); }
    if (argc >= 4) {
        bks.clear();
        std::string s = argv[3];
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            bks.push_back((unsigned)atoi(s.substr(pos, comma - pos).c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    if (argc >= 5) layers = atoi(argv[4]);

    int dev = 0;
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "GPU: %s  sm_%d%d  %.1f GB HBM\n", prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / 1e9);

    printf("# tag,k,op_or_B,dim_or_shots_or_bytes,metric1,metric2\n");
    validate(12);
    validate(18);
    validate_batched(12, 8);
    validate_batched(16, 8);
    validate_devmeas(12, 8);
    validate_devmeas(16, 8);
    validate_mimd(12, 8, 3);
    validate_mimd(16, 8, 3);
    for (unsigned k = kmin; k <= kmax; k += 4) transfer_cost(k);
    bench_single(kmin, kmax);
    bench_batched(bks, layers);
    bench_batched_meas(bks, layers);
    bench_batched_meas_dev(bks, layers);
    bench_mimd(bks, layers);
    return 0;
}
