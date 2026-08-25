#pragma once

#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/results.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace clifft::sampling::hip {

enum class CoefficientPrecision : uint8_t {
    FP64,
    FP32,
};

inline constexpr uint32_t kDefaultBlockSize = 256;
inline constexpr uint32_t kDefaultMaxBatchShots = 65536;

struct SamplingOptions {
    std::optional<uint64_t> seed = std::nullopt;
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64;
    uint32_t block_size = kDefaultBlockSize;
    uint32_t max_batch_shots = kDefaultMaxBatchShots;
};

struct ReplayResult {
    bool reachable = true;
    bool survived = true;
    double log_probability = 0.0;
    SamplingResult outputs;
};

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string backend_info();

// Owns one uploaded executable and a precision-specific reusable workspace.
// The object is synchronous and bound to the device current at construction.
// Overlapping calls are rejected; use a separate Sampler per caller.
class Sampler {
  public:
    explicit Sampler(const ExecutablePlan& executable,
                     CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64,
                     uint32_t max_batch_shots = kDefaultMaxBatchShots);
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
    [[nodiscard]] uint32_t max_batch_shots() const;
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

}  // namespace clifft::sampling::hip
