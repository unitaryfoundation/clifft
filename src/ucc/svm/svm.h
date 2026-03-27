#pragma once

#include "ucc/backend/backend.h"

#include "stim.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
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

    // Seed from a single 64-bit value via SplitMix64 expansion.
    inline void seed(uint64_t seed_val) {
        uint64_t z = seed_val;
        s_[0] = splitmix64(z);
        s_[1] = splitmix64(z);
        s_[2] = splitmix64(z);
        s_[3] = splitmix64(z);
    }

    // Seed all 256 bits directly (e.g. from hardware entropy).
    inline void seed_full(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
        s_[0] = s0;
        s_[1] = s1;
        s_[2] = s2;
        s_[3] = s3;
    }

    // Seed from OS hardware entropy (std::random_device).
    // Uses the glibc /dev/urandom workaround from Stim for
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94087
    void seed_from_entropy();

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

struct StateConfig {
    uint32_t peak_rank;
    uint32_t num_measurements;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    std::optional<uint64_t> seed = std::nullopt;
};

class SchrodingerState {
  public:
    explicit SchrodingerState(StateConfig cfg);
    SchrodingerState(uint32_t peak_rank, uint32_t num_measurements)
        : SchrodingerState(
              StateConfig{.peak_rank = peak_rank, .num_measurements = num_measurements}) {}

    ~SchrodingerState();

    // Non-copyable (owns aligned memory)
    SchrodingerState(const SchrodingerState&) = delete;
    SchrodingerState& operator=(const SchrodingerState&) = delete;

    // Movable
    SchrodingerState(SchrodingerState&& other) noexcept;
    SchrodingerState& operator=(SchrodingerState&& other) noexcept;

    // Reset to |0...0> state for next shot (reuses allocation).
    // Does NOT reseed the PRNG -- the RNG streams forward naturally.
    void reset();

    // Explicitly reseed the PRNG (for deterministic test replay).
    void reseed(uint64_t seed) { rng_.seed(seed); }
    void reseed_from_entropy() { rng_.seed_from_entropy(); }

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
    // Uses BitMask<kMaxInlineQubits> for compile-time 512-qubit scalability.
    PauliBitMask p_x;
    PauliBitMask p_z;

    // Global Scalar (gamma): continuous global phase + deferred normalization
    [[nodiscard]] std::complex<double> gamma() const { return gamma_; }
    void set_gamma(std::complex<double> g) { gamma_ = g; }

    // Multiply gamma by a unit-magnitude phase factor.
    void multiply_phase(std::complex<double> phase) {
        assert(std::abs(std::norm(phase) - 1.0) < 1e-9 && "Phase must be unitary");
        gamma_ *= phase;
    }

    // Multiply gamma by a real scale factor, triggering renormalization
    // if gamma drifts toward overflow or underflow.
    // Uses std::abs (not std::norm) to avoid squaring-induced underflow:
    // norm() squares the magnitude, so values near 1e-154 underflow to 0,
    // while abs() uses hypot() which stays representable down to ~5e-308.
    void scale_magnitude(double scale) {
        gamma_ *= scale;
        double g_mag = std::abs(gamma_);
        if (g_mag > 1e100 || (g_mag < 1e-100 && g_mag > 0.0)) {
            uint64_t sz = v_size();
            for (uint64_t ri = 0; ri < sz; ++ri)
                v_[ri] *= g_mag;
            gamma_ /= g_mag;
        }
    }

    // Current active dimension k (v_ holds 2^active_k meaningful entries)
    uint32_t active_k = 0;

    // Post-selection: true if this shot was discarded by OP_POSTSELECT.
    bool discarded = false;

    // Classical Memory
    std::vector<uint8_t> meas_record;
    std::vector<uint8_t> det_record;
    std::vector<uint8_t> obs_record;

    // Gap-based noise sampling: index of next noise site that might fire.
    // Sites with index < next_noise_idx are guaranteed silent (identity).
    uint32_t next_noise_idx = 0;

    // Forced-fault state for importance sampling (k-fault conditioning).
    // When active, the gap sampler is bypassed: only sites listed in
    // noise_indices fire, and only readout entries in readout_indices flip.
    struct ForcedFaults {
        bool active = false;
        std::vector<uint32_t> noise_indices;    // Sorted quantum site indices to force
        std::vector<uint32_t> readout_indices;  // Sorted readout entry indices to force
        uint32_t noise_pos = 0;                 // Two-pointer cursor for noise
        uint32_t readout_pos = 0;               // Two-pointer cursor for readout
    } forced_faults;

    // Advance the forced-fault noise cursor to the next forced site.
    void advance_forced_noise() {
        auto& ff = forced_faults;
        if (ff.noise_pos < ff.noise_indices.size()) {
            next_noise_idx = ff.noise_indices[ff.noise_pos++];
        } else {
            next_noise_idx = static_cast<uint32_t>(-1);  // Sentinel: no more faults
        }
    }

    // Advance next_noise_idx by sampling an exponential gap.
    // Uses the cumulative hazard table to skip silent noise sites in O(1).
    void draw_next_noise(const std::vector<double>& hazards) {
        // Gap exhaustion fast-path: when the sampled exponential gap
        // exceeds the total accumulated hazard, std::upper_bound returns
        // end(), making next_noise_idx == size() (out-of-bounds). This is
        // mathematically correct: a gap larger than the remaining circuit
        // hazard means no further noise events fire in this shot, so the
        // VM skips all subsequent OP_NOISE sites in O(1) via the
        // site_idx != next_noise_idx guard in exec_noise().
        if (hazards.empty() || next_noise_idx >= hazards.size()) {
            next_noise_idx = static_cast<uint32_t>(-1);
            return;
        }
        double current_hazard = (next_noise_idx == 0) ? 0.0 : hazards[next_noise_idx - 1];
        double gap = -std::log(1.0 - random_double());
        double target_hazard = current_hazard + gap;
        auto it = std::upper_bound(hazards.begin(), hazards.end(), target_hazard);
        next_noise_idx = static_cast<uint32_t>(std::distance(hazards.begin(), it));
    }

    // Telemetry: count of times the epsilon threshold caught floating-point
    // dust in active measurements, forcing a deterministic branch instead of
    // a spurious PRNG roll. Accumulates across shots (not reset per shot).
    uint64_t dust_clamps = 0;

  private:
    std::complex<double> gamma_ = {1.0, 0.0};
    std::complex<double>* v_ = nullptr;  // page-aligned
    uint64_t array_size_ = 0;            // 2^peak_rank (allocated capacity)
    size_t v_alloc_bytes_ = 0;           // actual allocation size in bytes
    uint32_t peak_rank_ = 0;
    bool v_is_mmap_ = false;  // true if v_ allocated via mmap
    Xoshiro256PlusPlus rng_;

    // --- Cold fields (rare per-shot probes) ---
    // Placed after rng_ to preserve cache-line packing of hot fields
    // (gamma_, v_, rng_) which are accessed on every opcode.
  public:
    // Expectation value record: one double per EXP_VAL probe per shot.
    std::vector<double> exp_vals;
};

// =============================================================================
// SVM Execution
// =============================================================================

/// Execute a compiled program for one shot, populating state with results.
void execute(const CompiledModule& program, SchrodingerState& state);

/// Return the name of the active SVM dispatch backend ("avx512", "avx2", or "scalar").
/// Reflects the resolved CPUID path or UCC_FORCE_ISA override.
const char* svm_backend();

/// Results from sampling a circuit.
struct SampleResult {
    std::vector<uint8_t> measurements;  // Shape: [shots * num_measurements]
    std::vector<uint8_t> detectors;     // Shape: [shots * num_detectors]
    std::vector<uint8_t> observables;   // Shape: [shots * num_observables]
    std::vector<double> exp_vals;       // Shape: [shots * num_exp_vals]
};

/// Run multiple shots and return all records.
/// The PRNG is seeded once at the start of the batch and streams forward
/// across shots (no per-shot reseeding). If seed is nullopt, 256 bits of
/// OS hardware entropy are used; if provided, a deterministic SplitMix64
/// expansion initializes the xoshiro256++ state for reproducible results.
SampleResult sample(const CompiledModule& program, uint32_t shots,
                    std::optional<uint64_t> seed = std::nullopt);

/// Results from survivor-only sampling (post-selection aware).
/// Only shots that pass all OP_POSTSELECT checks contribute to the arrays.
struct SurvivorResult {
    uint32_t total_shots = 0;
    uint32_t passed_shots = 0;

    // Number of surviving shots where at least one observable was flipped.
    // This is what Sinter expects as the "errors" count.
    uint32_t logical_errors = 0;

    // Per-observable count of how many surviving shots had obs[i] == 1.
    // Length: num_observables. Useful for detailed per-observable tracking.
    std::vector<uint64_t> observable_ones;

    // Flat arrays for surviving shots only. Empty when keep_records=false.
    std::vector<uint8_t> measurements;  // Shape: [passed_shots * num_measurements]
    std::vector<uint8_t> detectors;     // Shape: [passed_shots * num_detectors]
    std::vector<uint8_t> observables;   // Shape: [passed_shots * num_observables]
    std::vector<double> exp_vals;       // Shape: [passed_shots * num_exp_vals]
};

/// Sample shots and return results only for survivors (non-discarded shots).
/// When keep_records=false, only counts are populated (zero array allocation).
/// PRNG seeding follows the same seed-once-and-stream convention as sample().
SurvivorResult sample_survivors(const CompiledModule& program, uint32_t shots,
                                std::optional<uint64_t> seed = std::nullopt,
                                bool keep_records = false);

/// Sample with exactly k forced faults per shot.
/// Sites are sampled from the exact conditional Poisson-Binomial
/// distribution. When all site probabilities are uniform, an O(k)
/// Fisher-Yates sampler is used automatically.
SampleResult sample_k(const CompiledModule& program, uint32_t shots, uint32_t k,
                      std::optional<uint64_t> seed = std::nullopt);

/// Sample survivors with exactly k forced faults per shot.
SurvivorResult sample_k_survivors(const CompiledModule& program, uint32_t shots, uint32_t k,
                                  std::optional<uint64_t> seed = std::nullopt,
                                  bool keep_records = false);

/// Return per-site total fault probabilities for importance sampling.
/// The returned vector has length N_q + N_r: first the quantum noise sites
/// (sum of channel probs), then the readout noise entries.
std::vector<double> noise_site_probabilities(const CompiledModule& program);

// =============================================================================
// Statevector Expansion
// =============================================================================

/// Expand the factored state |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
/// into a dense 2^n statevector for validation.
/// Capped at n <= 10 qubits (8 MB unitary matrix) to prevent OOM.
std::vector<std::complex<double>> get_statevector(const CompiledModule& program,
                                                  const SchrodingerState& state);

}  // namespace ucc
