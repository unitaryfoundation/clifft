#pragma once

#include "ucc/backend/backend.h"

#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace ucc {

// =============================================================================
// Schrödinger Virtual Machine State
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

    // Reset to |0...0⟩ state for next shot (reuses allocation)
    void reset(uint64_t seed);

    // Access coefficient array
    [[nodiscard]] std::complex<double>* v() { return v_; }
    [[nodiscard]] const std::complex<double>* v() const { return v_; }
    [[nodiscard]] uint64_t array_size() const { return array_size_; }

    // Generate random double in [0, 1) using deterministic bit manipulation.
    // CRITICAL: Do NOT use std::uniform_real_distribution — its output is
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
    std::mt19937_64 rng_;
};

// =============================================================================
// SVM Execution
// =============================================================================

/// Execute a compiled program for one shot, populating state with results.
void execute(const CompiledModule& program, SchrodingerState& state);

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
/// picture back to the Schrödinger picture.
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
