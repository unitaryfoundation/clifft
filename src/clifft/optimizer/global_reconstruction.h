#pragma once

// Closed-output global reconstruction for CPU sampling.
//
// Given a noisy Clifford+T circuit and a requested output set (all measurement
// records, detectors, and observables, or only detectors and observables),
// reconstruction builds an equivalent smaller sampling problem in three steps:
//   1. reconstruct_global_sampling: dilate measurements into ancillas and
//      resets into fresh wires, dress T/T_DAG gadgets with input stabilizers,
//      fold the gadget projections into the input state, reduce to the
//      X-support basis, and schedule commuting measurements and rotations.
//   2. compress_reconstruction_noise: project categorical noise channels onto
//      the basis of operation sign masks and convolve channels with identical
//      image support.
//   3. lower_global_reconstruction: emit an HirModule with compressed noise
//      channels, fault-controlled rotations, and public outputs in source order.

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clifft {

// Records includes detector and observable outputs as well: readout noise can
// flip a record after an earlier detector has already consumed its value.
enum class ReconstructionOutputs { Records, Detectors };

struct ReconstructionLimits {
    uint32_t max_dilation_qubits = 2048;
    uint32_t max_noise_bits = 65536;
    uint32_t max_outputs = 4096;
    size_t max_operations = 50000;
    uint32_t max_schedule_operations = 2048;
    size_t max_planning_steps = 10000000;
    size_t max_channel_products = 1000000;
    size_t max_channel_support = 4096;
    size_t max_lowering_bytes = 128 * 1024 * 1024;
};

// Affine GF(2) sign mask over categorical fault variables plus a deterministic
// constant bit. Bits produced by one multi-outcome channel are mutually
// exclusive rather than independent Bernoulli draws.
struct ReconstructionSign {
    std::vector<uint64_t> words;
    bool constant = false;
};

// A Pauli rotation (half_turns != 0) or projective measurement (half_turns == 0)
// in the reduced X-support coordinates. When `output` is engaged, the
// measurement outcome is exported at that public output index.
struct ReconstructionOperation {
    PauliString axis;
    ReconstructionSign sign;
    double half_turns = 0;
    std::optional<uint32_t> output;
};

struct ReconstructionNoiseOutcome {
    double probability;
    std::vector<uint32_t> bits;
};

// Public output indices address visible records, then detectors, then
// observables.
struct ReconstructionOutputLayout {
    uint32_t num_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;

    uint32_t num_outputs() const { return num_records + num_detectors + num_observables; }
};

struct GlobalReconstruction {
    uint32_t num_qubits = 0;
    uint32_t dilation_qubits = 0;
    uint32_t num_noise_bits = 0;
    uint32_t peak_active_width = 0;
    ReconstructionOutputLayout outputs;
    std::vector<ReconstructionOperation> operations;
    std::vector<std::vector<ReconstructionNoiseOutcome>> noise_channels;
};

struct ReconstructionResult {
    std::optional<GlobalReconstruction> reconstruction;
    std::string fallback_reason;
};

// Compressed categorical fault channels and remapped operation signs in the
// span of noise parities actually observed by operations.
struct ReconstructionNoise {
    uint32_t num_bits = 0;
    std::vector<ReconstructionSign> signs;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels;
};

struct ReconstructionNoiseResult {
    std::optional<ReconstructionNoise> noise;
    std::string fallback_reason;
};

struct ReconstructionLoweringResult {
    std::optional<HirModule> hir;
    std::string fallback_reason;
};

// Reconstructs a closed sampling representation for the requested outputs of a
// parsed circuit. Unsupported gates or exceeded limits populate fallback_reason.
[[nodiscard]] ReconstructionResult reconstruct_global_sampling(
    const Circuit& circuit, ReconstructionOutputs outputs, const ReconstructionLimits& limits = {});

// Projects categorical noise channels onto the operation-sign span and
// convolves independent channels that share the same image support.
[[nodiscard]] ReconstructionNoiseResult compress_reconstruction_noise(
    const GlobalReconstruction& reconstruction, const ReconstructionLimits& limits = {});

// Emits an HirModule for the reconstructed problem, preserving the reference
// parities and post-selection mask computed on the original circuit.
[[nodiscard]] ReconstructionLoweringResult lower_global_reconstruction(
    const GlobalReconstruction& reconstruction, const PostselectionMetadata& postselection = {},
    const ReconstructionLimits& limits = {});

}  // namespace clifft
