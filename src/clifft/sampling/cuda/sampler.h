#pragma once

#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/results.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace clifft::sampling::cuda {

enum class CoefficientPrecision : uint8_t {
    FP64,
    FP32,
};

// Auto selects per plan and device: ThreadPerShot at widths one thread should
// own, BlockShared when one shot's coefficients fit the device's opt-in shared
// memory, and BlockGlobal otherwise. Explicit values force one tier and are
// rejected when the plan does not fit it.
enum class ExecutionTier : uint8_t {
    Auto,
    ThreadPerShot,
    BlockShared,
    BlockGlobal,
};

inline constexpr uint32_t kDefaultBlockSize = 256;
inline constexpr uint32_t kDefaultMaxBatchShots = 65536;

struct SamplingOptions {
    std::optional<uint64_t> seed = std::nullopt;
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64;
    ExecutionTier tier = ExecutionTier::Auto;
    // Threads per block. The cooperative tiers split one shot's coefficients
    // across the block; the thread-per-shot tier packs shots into blocks.
    uint32_t block_size = kDefaultBlockSize;
    uint32_t max_batch_shots = kDefaultMaxBatchShots;
    // Cap on shots resident at once in the cooperative tiers; later shots
    // loop inside the kernel. Zero derives the cap from the multiprocessor
    // count and, for BlockGlobal, from free device memory.
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
// device without uploading anything.
[[nodiscard]] ExecutionTier selected_tier(
    const ExecutablePlan& executable,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64);

// Owns one uploaded executable and a precision-specific reusable workspace.
// The object is synchronous and bound to the device current at construction.
// Overlapping calls are rejected; use a separate Sampler per caller.
class Sampler {
  public:
    explicit Sampler(const ExecutablePlan& executable,
                     CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64,
                     uint32_t max_batch_shots = kDefaultMaxBatchShots,
                     ExecutionTier tier = ExecutionTier::Auto, uint32_t max_concurrent_shots = 0);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) noexcept;
    Sampler& operator=(Sampler&&) noexcept;

    [[nodiscard]] SamplingResult sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                        uint32_t block_size = kDefaultBlockSize);
    [[nodiscard]] SamplingSurvivorResult sample_survivors(
        uint32_t shots, bool keep_records = false, std::optional<uint64_t> seed = std::nullopt,
        uint32_t block_size = kDefaultBlockSize);
    [[nodiscard]] ReplayResult replay_shot(std::span<const uint8_t> forced_records);

    [[nodiscard]] CoefficientPrecision coefficient_precision() const;
    [[nodiscard]] ExecutionTier execution_tier() const;
    [[nodiscard]] uint32_t max_batch_shots() const;
    [[nodiscard]] uint32_t max_concurrent_shots() const;
    [[nodiscard]] size_t allocated_device_bytes() const;
    [[nodiscard]] uint32_t num_visible_records() const;
    [[nodiscard]] uint32_t num_records() const;
    [[nodiscard]] uint32_t num_detectors() const;
    [[nodiscard]] uint32_t num_observables() const;
    [[nodiscard]] uint32_t num_exp_vals() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] SamplingResult sample(const ExecutablePlan& executable, uint32_t shots,
                                    const SamplingOptions& options = {});
[[nodiscard]] SamplingSurvivorResult sample_survivors(const ExecutablePlan& executable,
                                                      uint32_t shots, bool keep_records = false,
                                                      const SamplingOptions& options = {});
[[nodiscard]] ReplayResult replay_shot(
    const ExecutablePlan& executable, std::span<const uint8_t> forced_records,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64);

}  // namespace clifft::sampling::cuda
