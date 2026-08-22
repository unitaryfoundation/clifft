#pragma once

#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/sampler.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace clifft::sampling::hip {

enum class CoefficientPrecision : uint8_t {
    FP64,
    FP32,
};

struct SamplingOptions {
    std::optional<uint64_t> seed = std::nullopt;
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64;
    uint32_t block_size = 256;
};

struct ReplayResult {
    bool reachable = true;
    bool survived = true;
    double log_probability = 0.0;
    SamplingResult outputs;
};

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string backend_info();

[[nodiscard]] SamplingResult sample(const Executable& executable, uint32_t shots,
                                    const SamplingOptions& options = {});
[[nodiscard]] SamplingSurvivorResult sample_survivors(const Executable& executable, uint32_t shots,
                                                      bool keep_records = false,
                                                      const SamplingOptions& options = {});
[[nodiscard]] ReplayResult replay_shot(
    const Executable& executable, std::span<const uint8_t> forced_records,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64);

}  // namespace clifft::sampling::hip
