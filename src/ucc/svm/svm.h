#pragma once

#include "ucc/backend/backend.h"

#include "stim.h"

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
// Maps exactly to the Factored State Representation:
//   |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
//
// Memory layout:
//   - v_: 64-byte aligned array of 2^peak_rank complex amplitudes
//   - p_x, p_z: Pauli frame P (stim::bitword for 512-qubit scalability)
//   - gamma: global scalar (phase + deferred normalization)
//   - active_k: current active dimension k

class SchrodingerState {
  public:
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
    [[nodiscard]] uint64_t v_size() const { return 1ULL << active_k; }
    [[nodiscard]] uint64_t array_size() const { return array_size_; }

    // Generate random double in [0, 1) using deterministic bit manipulation.
    // CRITICAL: Do NOT use std::uniform_real_distribution -- its output is
    // implementation-defined and varies across compilers (GCC vs Clang vs MSVC).
    [[nodiscard]] double random_double() { return static_cast<double>(rng_() >> 11) * 0x1.0p-53; }

    // --- Factored State Components ---

    // The Pauli Frame (P): tracks stochastic bit-flips and phase-flips.
    // Uses stim::bitword<kStimWidth> for future 512-qubit scalability.
    stim::bitword<kStimWidth> p_x = 0;
    stim::bitword<kStimWidth> p_z = 0;

    // Global Scalar (gamma): continuous global phase + deferred normalization
    std::complex<double> gamma = {1.0, 0.0};

    // Current active dimension k (v_ holds 2^active_k meaningful entries)
    uint32_t active_k = 0;

    // Classical Memory
    std::vector<uint8_t> meas_record;
    std::vector<uint8_t> det_record;
    std::vector<uint8_t> obs_record;

  private:
    std::complex<double>* v_ = nullptr;  // 64-byte aligned
    uint64_t array_size_ = 0;            // 2^peak_rank (allocated capacity)
    uint32_t peak_rank_ = 0;
    Xoshiro256PlusPlus rng_;
};

// =============================================================================
// SVM Execution
// =============================================================================

/// Execute a compiled program for one shot, populating state with results.
void execute(const CompiledModule& program, SchrodingerState& state);

/// Results from sampling a circuit.
struct SampleResult {
    std::vector<uint8_t> measurements;  // Shape: [shots * num_measurements]
    std::vector<uint8_t> detectors;     // Shape: [shots * num_detectors]
    std::vector<uint8_t> observables;   // Shape: [shots * num_observables]
};

/// Run multiple shots and return all records.
SampleResult sample(const CompiledModule& program, uint32_t shots, uint64_t seed = 0);

// =============================================================================
// Statevector Expansion
// =============================================================================

/// Expand the factored state |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
/// into a dense 2^n statevector for validation.
/// Capped at n <= 10 qubits (8 MB unitary matrix) to prevent OOM.
std::vector<std::complex<double>> get_statevector(const CompiledModule& program,
                                                  const SchrodingerState& state);

}  // namespace ucc
