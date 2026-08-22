#pragma once

#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/sampler.h"

#include <cstdint>
#include <optional>
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

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string backend_info();

[[nodiscard]] SamplingResult sample(const Executable& executable, uint32_t shots,
                                    const SamplingOptions& options = {});
[[nodiscard]] SamplingSurvivorResult sample_survivors(const Executable& executable, uint32_t shots,
                                                      bool keep_records = false,
                                                      const SamplingOptions& options = {});

}  // namespace clifft::sampling::hip
