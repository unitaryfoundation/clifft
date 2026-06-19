// bench_common.hpp -- shared timing, RNG, index helpers, and workload
// definitions for the clifft GPU microbenchmark.
//
// Pure portable C++17 (no CUDA, no clifft dependency) so it is included by
// both the CPU benchmark (bench_cpu.cpp) and the CUDA benchmark (bench_gpu.cu).
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mb {

// ---------------------------------------------------------------------------
// Bit-scatter helpers (portable replacements for clifft's PDEP-based
// scatter_bits_1 / scatter_bits_2; on ARM clifft itself uses the same
// insert-zero-bit fallback).
// ---------------------------------------------------------------------------

// Insert a zero bit at position p into x (bits >= p shift up by one).
inline uint64_t insert_zero_bit(uint64_t x, unsigned p) {
    uint64_t mask = (uint64_t(1) << p) - 1;
    return (x & mask) | ((x & ~mask) << 1);
}

// Insert two zero bits at positions a < b.
inline uint64_t insert_two_zero_bits(uint64_t x, unsigned a, unsigned b) {
    return insert_zero_bit(insert_zero_bit(x, a), b);
}

// ---------------------------------------------------------------------------
// Constants matching clifft semantics.
// ---------------------------------------------------------------------------
inline constexpr double kInvSqrt2 = 0.7071067811865476;
// e^{i*pi/4} = (cos45, sin45) = (kInvSqrt2, kInvSqrt2)
inline constexpr double kTPhaseRe = kInvSqrt2;
inline constexpr double kTPhaseIm = kInvSqrt2;

// clifft only spins up threads above this active rank (kMinRankForThreads=18).
inline constexpr unsigned kMinRankForThreads = 18;

// ---------------------------------------------------------------------------
// Deterministic RNG (splitmix64) for measurement branch sampling.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed) {}
    inline uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    inline double next_unit() {  // uniform [0,1)
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

// ---------------------------------------------------------------------------
// Timing.
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// median of a vector of samples (sorts a copy)
double median(std::vector<double> xs);

// ---------------------------------------------------------------------------
// Op codes for the synthetic workload.
// ---------------------------------------------------------------------------
enum class Op { H, T, CZ, CNOT, EXPAND, EXPAND_T, MEAS_DIAG, MEAS_INTERFERE };

const char* op_name(Op op);

// A fixed "active-block layer" replayed L times at active rank k. One layer:
//   H on every axis, then T on every axis, then a CNOT chain, then one CZ.
// Returns the op/axis schedule for a given k. Used by both CPU (loop over
// shots) and GPU (one kernel launch per scheduled op across the whole batch).
struct ScheduledOp {
    Op op;
    unsigned a;  // primary axis (control for 2q)
    unsigned b;  // secondary axis (target for 2q); ignored otherwise
};
std::vector<ScheduledOp> make_layer_schedule(unsigned k, unsigned layers);

}  // namespace mb
