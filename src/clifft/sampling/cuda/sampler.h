#pragma once

#include "clifft/sampling/cuda/executable.h"
#include "clifft/sampling/sampler.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace clifft::sampling::cuda {

enum class CoefficientPrecision : uint8_t {
    FP64,
    FP32,
};

// Execution tiers. Auto picks per plan: ThreadPerShot for peak widths a
// single thread should own, BlockShared when one shot's coefficients and
// scratch fit the device's opt-in shared-memory budget, BlockGlobal
// otherwise. Explicit values force one tier for benchmarking and are
// rejected when the plan does not fit that tier.
enum class ExecutionTier : uint8_t {
    Auto,
    ThreadPerShot,
    BlockShared,
    BlockGlobal,
};

struct SamplingOptions {
    std::optional<uint64_t> seed = std::nullopt;
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64;
    ExecutionTier tier = ExecutionTier::Auto;
    // Threads per block for the cooperative tiers, and shots per launch block
    // for the thread-per-shot tier.
    uint32_t block_size = 256;
    // Cap on concurrently resident shots for the cooperative tiers; shots
    // beyond it loop inside the kernel. Zero derives the cap from device
    // memory (global tier) or a multiple of the SM count (shared tier).
    uint32_t max_concurrent_shots = 0;
};

struct ReplayResult {
    bool reachable = true;
    bool survived = true;
    double log_probability = 0.0;
    SamplingResult outputs;
};

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string backend_info();

// Reports the tier Auto would select for this executable on the current
// device, without launching anything.
[[nodiscard]] ExecutionTier selected_tier(const Executable& executable,
                                          const SamplingOptions& options = {});

[[nodiscard]] SamplingResult sample(const Executable& executable, uint32_t shots,
                                    const SamplingOptions& options = {});
[[nodiscard]] SamplingSurvivorResult sample_survivors(const Executable& executable, uint32_t shots,
                                                      bool keep_records = false,
                                                      const SamplingOptions& options = {});
[[nodiscard]] ReplayResult replay_shot(
    const Executable& executable, std::span<const uint8_t> forced_records,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64);

}  // namespace clifft::sampling::cuda
