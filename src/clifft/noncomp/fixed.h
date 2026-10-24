#pragma once

// Internal fixed execution for requests whose hidden leaked-level label can
// be marginalized. Preparation certifies the whole request and compiles one
// quantum plan; shots supply classical status-conditioned inputs to that plan.
// Unsupported requests return a reason for the caller's trajectory fallback.

#include "clifft/noncomp/preparation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

namespace clifft {

enum class FixedNonComputationalReason {
    InitialState,
    NamedTransition,
    Spreading,
    LevelSensitiveReadout,
    PartnerChannel,
    UnsupportedOperation,
    DetailedLeakedStatus,
    ActiveWidth,
};

struct FixedNonComputationalRejection {
    FixedNonComputationalReason reason;
    // Index in the hook-expanded resolved circuit, including generated nodes.
    // The resolved operation's generated_by identifies its originating gate.
    std::optional<uint32_t> operation;
    uint32_t source_line = 0;
};

struct FixedNonComputationalMetrics {
    uint64_t preparation_compilations = 0;
    uint32_t peak_active_width = 0;
    uint32_t quantum_noise_inputs = 0;
    uint32_t readout_inputs = 0;
    uint32_t packed_noise_carriers = 0;
    // Scalar driver scratch; packed input storage depends on the resolved batch.
    size_t classical_workspace_bytes = 0;
};

// Per-call evidence for validation and benchmarks, separate from immutable
// preparation metrics. Shot counters are collected from executed worker ranges.
struct FixedNonComputationalSamplingMetrics {
    uint32_t batch_size = 1;
    uint32_t workers = 0;
    uint64_t scalar_shots = 0;
    uint64_t packed_shots = 0;
    uint64_t packed_batches = 0;
    // Retained buffer bytes for packed execution, external inputs, and driver scratch;
    // excludes immutable plan data, result rows, and allocator bookkeeping.
    uint64_t packed_worker_bytes = 0;
};

struct FixedNonComputationalResult {
    NonComputationalSample output;
    FixedNonComputationalSamplingMetrics metrics;
};

namespace detail {
struct FixedNonComputationalData;
}

class FixedNonComputationalProgram {
  public:
    // Unset batch_size uses the shared automatic policy; one forces scalar.
    FixedNonComputationalResult sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                       uint32_t threads = 1,
                                       std::optional<uint32_t> batch_size = std::nullopt) const;
    const FixedNonComputationalMetrics& metrics() const;
    bool returns_final_status() const;

  private:
    explicit FixedNonComputationalProgram(
        std::shared_ptr<const detail::FixedNonComputationalData> data);
    std::shared_ptr<const detail::FixedNonComputationalData> data_;

    friend std::variant<FixedNonComputationalProgram, FixedNonComputationalRejection>
    prepare_fixed_noncomputational(const PreparedNonComputationalRequest& request);
};

std::variant<FixedNonComputationalProgram, FixedNonComputationalRejection>
prepare_fixed_noncomputational(const PreparedNonComputationalRequest& request);

}  // namespace clifft
