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
// Op codes for the synthetic workload. U2/U4 are clifft's fused dense 1q/2q
// matrix opcodes (OP_ARRAY_U2 / OP_ARRAY_U4) -- the hot path the optimizer's
// tile/single-axis fusion passes emit, with ~4-8x the arithmetic intensity of
// H/T at the same memory traffic; without them the op mix would not match
// what clifft's compiler actually leaves in a program.
// ---------------------------------------------------------------------------
// FRAME_TICK models one frame/bookkeeping bytecode instruction (frame
// Cliffords, dormant measurements, Pauli/noise ops): O(1) per shot, touches no
// amplitudes. Real compiled programs are 23-77% frame ops (see
// ../opcode_census.md) -- free for the CPU and the MIMD interpreter, but one
// (near-empty) kernel launch each for the lockstep-SoA design, so the real-mix
// schedule includes them to price that launch tax.
enum class Op { H, T, CZ, CNOT, U2, U4, EXPAND, EXPAND_T, MEAS_DIAG, MEAS_INTERFERE, FRAME_TICK };

const char* op_name(Op op);

// Amplitudes actually touched by one application of `op` on a rank-k block of
// dimension dim = 2^k (e.g. CZ touches dim/4, EXPAND reads dim and writes
// dim). CPU/GPU halves must both use this so cross-op AND cross-device
// Gamp/s comparisons are meaningful.
inline uint64_t amps_touched(Op op, uint64_t dim) {
    switch (op) {
        case Op::H: return dim;                 // butterfly rw on all
        case Op::T: return dim / 2;             // phases the v-set half
        case Op::CZ: return dim / 4;            // negates the both-set quarter
        case Op::CNOT: return dim / 2;          // swaps 2 amps per quarter-block
        case Op::U2: return dim;                // dense 2x2 on all pairs
        case Op::U4: return dim;                // dense 4x4 on all blocks
        case Op::EXPAND: return 2 * dim;        // reads dim, writes dim
        case Op::EXPAND_T: return 2 * dim;      // reads dim, writes dim (phased)
        case Op::MEAS_DIAG: return 2 * dim;     // reduce reads dim + compact r/w dim
        case Op::MEAS_INTERFERE: return 2 * dim;  // reduce reads dim + fold r/w dim
        case Op::FRAME_TICK: return 0;          // O(1) per shot, no amplitudes
    }
    return dim;
}

// Fixed generic unitaries for the U2/U4 model ops and validation (interleaved
// re,im; row-major). kU2 = u3(0.6, 0.35, 0.8); kU4 = kron(kU2, u3(1.1,-0.4,0.25)).
inline constexpr double kU2[8] = {
    0.9553364891256060, 0.0000000000000000, -0.2058909107286161, -0.2119932202323976,
    0.2776036182326806, 0.1013332309229552, 0.3902429576261744, 0.8719966980889404};
inline constexpr double kU4[32] = {
    0.8144477837978134, 0.0000000000000000, -0.4838188430151684, -0.1235392328984321,
    -0.1755270502653098, -0.1807294187584803, 0.0768571317485036, 0.1339862144585600,
    0.4599246066823143, -0.1944530048360987, 0.8053024131083846, -0.1217095558080070,
    -0.1422713529342881, -0.0601513632413207, -0.2005639375200707, -0.1524695876103295,
    0.2366638919558055, 0.0863890642613379, -0.1274851669135874, -0.0872172952031696,
    0.3326916909373423, 0.7433985682757410, -0.0848717640376650, -0.4920766186182839,
    0.1542715973372545, -0.0077200142845794, 0.2469162319870369, 0.0500523981537599,
    0.3653631164730068, 0.3403709859003868, 0.4400480155951548, 0.6853341787069546};

// A fixed "active-block layer" replayed L times at active rank k. One layer:
//   H on every axis, T on every axis, a CNOT chain, one CZ, one fused U2, and
//   one fused U4 (the post-optimizer opcodes).
// Returns the op/axis schedule for a given k. Used by both CPU (loop over
// shots) and GPU (one kernel launch per scheduled op across the whole batch).
struct ScheduledOp {
    Op op;
    unsigned a;  // primary axis (control for 2q)
    unsigned b;  // secondary axis (target for 2q); ignored otherwise
};
std::vector<ScheduledOp> make_layer_schedule(unsigned k, unsigned layers);

// The "real shot shape" schedule: the gates-only batched workload is an
// idealized upper bound -- real clifft shots interleave measurements, whose
// reduce -> host-sample -> per-shot-conditional collapse is the dominant
// unknown at k < 20. One layer = the gate layer above + one
// completed MEAS_DIAG (rank k -> k-1) + one EXPAND (k-1 -> k), so the rank --
// and therefore the batch shape -- is identical every layer (this mirrors
// clifft's static, shot-invariant k trajectory).
std::vector<ScheduledOp> make_layer_schedule_meas(unsigned k, unsigned layers);

// The census-calibrated "real op mix" schedule (see ../opcode_census.md, built
// from real clifft-compiled programs): the gates-only and gate-heavy-meas
// schedules above overweight gates ~3-30x vs compiled bytecode. One layer =
// 8 gates (2 H, 2 T, 3 CNOT incl. one max-stride, 1 CZ; census gates-per-meas
// band 1.7-15.3, median ~8), one completed X-basis MEAS_INTERFERE (k -> k-1)
// and one EXPAND_T (k-1 -> k) -- the measurement/expand flavors real programs
// actually use (MEAS_DIAG/plain EXPAND are ~absent in compiled code) -- plus
// 12 FRAME_TICKs (frame:dense ~1.2, census band 0.3-3.3). This is the
// schedule the go/no-go ratio is quoted on.
std::vector<ScheduledOp> make_layer_schedule_real(unsigned k, unsigned layers);

}  // namespace mb
