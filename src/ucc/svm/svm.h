#pragma once

#include "ucc/backend/backend.h"

#include <bit>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace ucc {

// =============================================================================
// Scientific PRNG: xoshiro256++ (Seeded by SplitMix64)
// =============================================================================
//
// Reference implementation of xoshiro256++ 1.0 by David Blackman and
// Sebastiano Vigna (vigna@acm.org), described in:
//
//   Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom
//   Number Generators." ACM Trans. Math. Softw. 47(4), Article 36.
//   https://doi.org/10.1145/3460772
//
// Source: https://prng.di.unimi.it/xoshiro256plusplus.c
// Seeding: https://prng.di.unimi.it/splitmix64.c
// License: Public domain (CC0)
//
// Period: 2^256-1. State: 32 bytes (vs MT19937's 2504 bytes), making per-shot
// reseeding ~100x cheaper. Uses pure bitwise math to guarantee identical
// sequences across GCC/Clang/MSVC.

class Xoshiro256PlusPlus {
  public:
    explicit Xoshiro256PlusPlus(uint64_t seed_val = 0) { seed(seed_val); }

    inline void seed(uint64_t seed_val) {
        // SplitMix64 expands 64-bit seed into 256 bits of state.
        // Sequential seeds (shot 1, shot 2) avalanche into radically
        // different starting points in the 2^256 period.
        uint64_t z = seed_val;
        s_[0] = splitmix64(z);
        s_[1] = splitmix64(z);
        s_[2] = splitmix64(z);
        s_[3] = splitmix64(z);
    }

    inline uint64_t operator()() {
        const uint64_t result = std::rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;

        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];

        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);

        return result;
    }

  private:
    uint64_t s_[4];

    static inline uint64_t splitmix64(uint64_t& state) {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// =============================================================================
// Schrodinger Virtual Machine State
// =============================================================================
//
// Holds the quantum state during execution: a sparse coefficient array indexed
// by GF(2) coordinates, plus Pauli frame signs for Clifford tracking.
//
// Memory layout:
//   - v_: 64-byte aligned array of 2^rank complex amplitudes
//   - destab_signs/stab_signs: packed sign bits for Pauli frame
//   - meas_record: measurement outcomes (0 or 1)

class SchrodingerState {
  public:
    // Allocate state for given peak_rank (determines array size 2^rank).
    // Seed controls the deterministic PRNG for measurement sampling.
    explicit SchrodingerState(uint32_t peak_rank, uint32_t num_measurements,
                              uint32_t num_detectors = 0, uint32_t num_observables = 0,
                              uint64_t seed = 0);

    ~SchrodingerState();

    // Non-copyable (owns aligned memory)
    SchrodingerState(const SchrodingerState&) = delete;
    SchrodingerState& operator=(const SchrodingerState&) = delete;

    // Movable
    SchrodingerState(SchrodingerState&& other) noexcept;
    SchrodingerState& operator=(SchrodingerState&& other) noexcept;

    // Reset to |0...0> state for next shot (reuses allocation)
    void reset(uint64_t seed);

    // Access coefficient array
    [[nodiscard]] std::complex<double>* v() { return v_; }
    [[nodiscard]] const std::complex<double>* v() const { return v_; }
    [[nodiscard]] uint64_t array_size() const { return array_size_; }

    // Generate random double in [0, 1) using deterministic bit manipulation.
    // CRITICAL: Do NOT use std::uniform_real_distribution  --  its output is
    // implementation-defined and varies across compilers (GCC vs Clang vs MSVC).
    // This bit-manipulation approach ensures identical sequences across platforms
    // given the same seed, enabling reproducible simulation results.
    [[nodiscard]] double random_double() { return static_cast<double>(rng_() >> 11) * 0x1.0p-53; }

    // Pauli frame signs (X-part and Z-part)
    uint64_t destab_signs = 0;
    uint64_t stab_signs = 0;

    // Measurement record
    std::vector<uint8_t> meas_record;

    // Detector record (one bit per DETECTOR instruction)
    std::vector<uint8_t> det_record;

    // Observable record (one bit per logical observable index)
    std::vector<uint8_t> obs_record;

  private:
    std::complex<double>* v_ = nullptr;  // 64-byte aligned
    uint64_t array_size_ = 0;            // 2^peak_rank
    uint32_t peak_rank_ = 0;
    Xoshiro256PlusPlus rng_;
};

// =============================================================================
// SVM Execution
// =============================================================================

/// Execute a compiled program for one shot, populating state with results.
void execute(const CompiledModule& program, SchrodingerState& state);

// =============================================================================
// Execution Tracing
// =============================================================================

/// Snapshot of SVM state captured after each instruction during traced execution.
struct TraceEntry {
    uint32_t pc;                          // Program counter of the instruction
    Opcode opcode;                        // The opcode that was executed
    uint32_t rank_after;                  // Current rank after this instruction
    uint64_t destab_signs;                // Pauli frame X-signs after
    uint64_t stab_signs;                  // Pauli frame Z-signs after
    std::vector<std::complex<double>> v;  // Copy of v[0..2^rank) after
    std::string detail;                   // Human-readable detail (outcome, etc.)
};

/// Execute with tracing: runs the real execute() logic but captures a TraceEntry
/// after each instruction. The trace vector is populated in instruction order.
void execute_traced(const CompiledModule& program, SchrodingerState& state,
                    std::vector<TraceEntry>& trace);

/// Results from sampling a circuit with noise and QEC annotations.
struct SampleResult {
    std::vector<uint8_t> measurements;  // Shape: [shots * num_measurements]
    std::vector<uint8_t> detectors;     // Shape: [shots * num_detectors]
    std::vector<uint8_t> observables;   // Shape: [shots * num_observables]
};

/// Run multiple shots and return all records (measurements, detectors, observables).
/// Each vector is flattened row-major: [shots, record_size].
SampleResult sample(const CompiledModule& program, uint32_t shots, uint64_t seed = 0);

// =============================================================================
// Statevector Expansion
// =============================================================================

/// Expand the SVM's sparse representation into a dense 2^N statevector.
/// This applies the final Clifford tableau to convert from the Heisenberg
/// picture back to the Schrodinger picture.
///
/// Parameters:
///   state: The SVM state after execution (contains v[], destab_signs, stab_signs)
///   gf2_basis: The GF(2) basis vectors from compilation
///   final_tableau: The forward Clifford frame at circuit end
///   global_weight: Accumulated global phase factor
///
/// Returns:
///   Dense statevector of size 2^N where N = final_tableau.num_qubits
std::vector<std::complex<double>> get_statevector(
    const SchrodingerState& state, const std::vector<stim::bitword<kStimWidth>>& gf2_basis,
    const stim::Tableau<kStimWidth>& final_tableau, std::complex<double> global_weight);

}  // namespace ucc
