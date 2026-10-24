#pragma once

#include "clifft/circuit/circuit.h"
#include "clifft/tableau/pauli_string.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clifft {

// Records includes the detector/observable outputs as well: readout noise can
// change a record after a detector has already consumed its previous value.
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

// The constant is separate from the categorical noise bits. In particular,
// terms from one channel are not independent Bernoulli variables.
struct ReconstructionSign {
    std::vector<uint64_t> words;
    bool constant = false;
};

struct ReconstructionOperation {
    PauliString axis;
    ReconstructionSign sign;
    double half_turns = 0;
    int32_t output = -1;
};

struct ReconstructionNoiseOutcome {
    double probability;
    std::vector<uint32_t> bits;
};

// Operation output indices address records, detectors, then observables. Keep
// snapshots separate: later readout noise can change already-consumed records.
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

// A closed joint-output transformation. The caller must retain the source for
// state, continuation, and original fault-site queries. Input is a parsed AST;
// unsupported optimization cases return a reason, not an invalid-input error.
[[nodiscard]] ReconstructionResult reconstruct_global_sampling(
    const Circuit& circuit, ReconstructionOutputs outputs, const ReconstructionLimits& limits = {});

}  // namespace clifft
