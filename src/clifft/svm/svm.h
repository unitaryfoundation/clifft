#pragma once

#include "clifft/backend/backend.h"
#include "clifft/util/xoshiro.h"

#include "stim.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace clifft {

// =============================================================================
// Schrodinger Virtual Machine State
// =============================================================================
//
// Maps exactly to the Factored State Representation:
//   |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
//
// Memory layout:
//   - v_: 64-byte aligned array of 2^peak_rank complex amplitudes
//   - p_x, p_z: Pauli frame P, runtime-sized to ceil(num_qubits / 64) words
//   - gamma: global scalar (phase + deferred normalization)
//   - active_k: current active dimension k

struct StateConfig {
    uint32_t peak_rank;
    uint32_t num_measurements;
    uint32_t num_qubits = 0;  // Sizes the p_x / p_z Pauli frame storage.
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    std::optional<uint64_t> seed = std::nullopt;
};

class SchrodingerState {
  public:
    explicit SchrodingerState(StateConfig cfg);
    SchrodingerState(uint32_t peak_rank, uint32_t num_measurements, uint32_t num_qubits = 0)
        : SchrodingerState(StateConfig{.peak_rank = peak_rank,
                                       .num_measurements = num_measurements,
                                       .num_qubits = num_qubits}) {}

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

    // Generate random double in [0, 1). The extraction lives on the
    // generator so the SVM and the noncomp sampler draw it identically;
    // it deliberately avoids std::uniform_real_distribution, whose output
    // varies across compilers.
    [[nodiscard]] double random_double() { return rng_.next_double(); }

    // --- Factored State Components ---

    // The Pauli Frame (P): tracks stochastic bit-flips and phase-flips.
    // Sized to ceil(num_qubits / 64) 64-bit words at construction.
    std::vector<uint64_t> p_x;
    std::vector<uint64_t> p_z;

    // Number of physical qubits the frame is sized for.
    uint32_t num_qubits = 0;

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

    // True when the program has EXP_VAL probes. Fits in existing padding
    // after discarded (3 bytes free before the next 8-byte-aligned field),
    // so this does not shift any downstream struct layout.
    bool has_exp_vals = false;

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
    /// Allocate a zero-filled amplitude array for 2^peak_rank entries,
    /// setting v_, array_size_, v_alloc_bytes_, v_is_mmap_, peak_rank_.
    void allocate_array(uint32_t peak_rank);
    void free_array() noexcept;

    /// Grow the amplitude array to hold 2^peak_rank entries, preserving
    /// the live 2^active_k amplitudes (the region above stays zero); no-op
    /// when the allocation already suffices. This is the single sanctioned
    /// exception to the allocate-once invariant, so it is reachable only
    /// from resume(), only between dispatch entries, and only while a trap
    /// is pending (asserted). Never shrinks.
    void grow_for_continuation(uint32_t peak_rank);
    friend void resume(const CompiledModule& program, SchrodingerState& state, uint32_t offset);

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

    // --- Resumable trap state ---
    //
    // Set when an instrument fire cannot be resolved in-line: any form
    // firing to a leaked/lost destination, or any fire at a neglect-mode
    // dormant-random site (whose collapse belongs to the continuation).
    // execute() halts at the site with the state intact (the carrier
    // already collapsed onto the drawn source where the form allows it)
    // and the host continues via resume() in a recompiled module. Dormant
    // in ordinary sampling; reset() clears it.
    struct InstrumentTrap {
        uint32_t site_id = 0;  // CompiledInstrumentSite::site_id
        uint8_t source = 0;    // Drawn physical source level (0 = |0>)

        // True at a neglect-form site: no destination has been drawn at
        // all, so the host draws from the site's full column --
        // computational destinations included -- and the continuation
        // performs the collapse. False elsewhere: the destination class
        // is already drawn as leaked/lost, and the host draws only which
        // noncomputational level from the trap remainder.
        bool destination_pending = false;
    };
    std::optional<InstrumentTrap> pending_trap;

    // --- Forced-execution state ---
    //
    // Dormant in sampling mode. Set per record by the forced-execution
    // path, where each measurement kernel reads its outcome from
    // forced_record[classical_idx] instead of sampling. The kernels
    // accumulate the log-probability of the forced outcome into
    // forced_log_probability under the same dust-clamping policy
    // sample_branch() uses: a branch with prob <= kDustEpsilon * total
    // is treated as exactly zero, so forcing the dust outcome sets
    // forced_reachable = false and forcing the surviving outcome
    // contributes 0 (not log(prob/total)). Non-dust branches contribute
    // log(prob_b / total). When forced_reachable becomes false the
    // dispatcher short-circuits the rest of the bytecode.
    //
    // reset() clears all three to their dormant defaults.
    std::span<const uint8_t> forced_record;
    double forced_log_probability = 0.0;
    bool forced_reachable = true;
};

// =============================================================================
// SVM Execution
// =============================================================================

/// Execute a compiled program for one shot, populating state with results.
/// If an instrument fire cannot be resolved in-line (a leaked/lost
/// destination on any form, or any fire at a neglect-form site),
/// execution halts at the site with state.pending_trap set; continue via
/// resume().
void execute(const CompiledModule& program, SchrodingerState& state);

/// Continue a trapped shot in `program` starting at bytecode index
/// `offset` (for a trap at site s, program.instrument_offsets[s] + 1).
/// Requires state.pending_trap to be set and clears it. The program's
/// bytecode before `offset` must be bit-identical to the code the state
/// already executed -- the compiler's determinism plus the instrument
/// barrier contract guarantee this for a recompiled continuation of the
/// same circuit prefix. Grows the amplitude array and measurement-record
/// buffer if the continuation needs more than the state was built with
/// (the sanctioned trap-boundary exception to the allocate-once
/// invariant; a driver reusing one state across shots amortizes growth to
/// the chain maximum), and re-anchors the noise-gap cursor at the entry
/// offset (exact, because exponential gaps are memoryless). May itself
/// halt on a later trap.
void resume(const CompiledModule& program, SchrodingerState& state, uint32_t offset);

/// Return the name of the active SVM dispatch backend. Reflects the
/// resolved CPUID path or the CLIFFT_FORCE_ISA environment override.
///
/// In normal operation returns one of:
///   - "avx512" — the AVX-512 kernel is active.
///   - "avx2"   — the AVX-2 kernel is active.
///   - "scalar" — the portable scalar fallback is active.
///
/// If CLIFFT_FORCE_ISA names an ISA the host CPU cannot execute, or is
/// set to an unrecognized value, the backend reports one of:
///   - "trap:avx512"  — CLIFFT_FORCE_ISA=avx512 but host lacks
///                      avx2/bmi2/fma/avx512f/avx512dq.
///   - "trap:avx2"    — CLIFFT_FORCE_ISA=avx2 but host lacks
///                      avx2/bmi2/fma.
///   - "trap:unknown" — CLIFFT_FORCE_ISA is set to a value other than
///                      avx512/avx2/scalar (case-insensitive).
/// In each trap case the next execute() call throws std::runtime_error
/// with a message naming the missing features or accepted values.
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

/// Return exact |<x|psi>|^2 computational-basis probabilities for unitary
/// full-register basis states. EXP_VAL probes are ignored; measurement,
/// feedback, noise, detector, postselection, and observable opcodes are
/// rejected. Each basis mask is word-packed little-endian by qubit index.
std::vector<double> basis_probabilities(const CompiledModule& program,
                                        std::span<const uint64_t> basis_masks,
                                        size_t num_basis_masks, size_t words_per_basis_mask);

/// Sentinel returned by `record_probabilities()` for records the program
/// cannot emit. Equal to `std::numeric_limits<double>::lowest()` (-DBL_MAX).
/// A finite value is used (rather than `-infinity`) so the contract survives
/// `-ffast-math` builds, which assume infinities cannot occur and fold
/// away `std::isinf`/`std::isfinite`. Exponentiating it underflows to 0.
inline constexpr double kUnreachableLogProb = std::numeric_limits<double>::lowest();

/// Return exact log-probabilities for a batch of measurement records under
/// the compiled program. Computes the probability that sample() would emit
/// each record, modulo dust-clamping. The program must contain at least one
/// measurement and may include any unitary gate, EXP_VAL probe, or
/// classical feedback (OP_APPLY_PAULI); noise, detectors, observables, and
/// post-selection are rejected.
///
/// `records` is a packed buffer of `num_records * program.num_measurements`
/// bytes (0 or 1 per slot, in execution order -- the same order
/// sample().measurements uses). Records the program cannot emit return
/// `kUnreachableLogProb`; other entries are natural-log probabilities.
std::vector<double> record_probabilities(const CompiledModule& program,
                                         std::span<const uint8_t> records, size_t num_records);

// =============================================================================
// Statevector Expansion
// =============================================================================

/// Expand the factored state |psi> = gamma * U_C * P * (|phi>_A (x) |0>_D)
/// into a dense 2^n statevector for validation.
/// Capped at n <= 10 qubits (8 MB unitary matrix) to prevent OOM.
///
/// For unitary programs, compilation preserves the API-visible global phase
/// across optimization passes. Retained final-tableau expansion uses Stim's
/// complex<float> unitary path, so those amplitudes have float-scale
/// precision. After measurements or noise, only relative amplitudes and
/// probabilities are meaningful; overall phase may differ between
/// compilations.
std::vector<std::complex<double>> get_statevector(const CompiledModule& program,
                                                  const SchrodingerState& state);

}  // namespace clifft
