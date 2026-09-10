#pragma once

// Domain labels keep deterministic shot streams independent when the same
// user seed and shot index are used by different execution roles.

#include <cstdint>

namespace clifft {

inline constexpr uint64_t kSamplingExecutorDomain = 0x01;
inline constexpr uint64_t kHipSamplingExecutorDomain = 0x02;
inline constexpr uint64_t kBatchSamplingExecutorDomain = 0x03;
inline constexpr uint64_t kCudaSamplingExecutorDomain = 0x04;
inline constexpr uint64_t kTrajectoryDriverDomain = 0x11;
inline constexpr uint64_t kTrajectoryExecutorDomain = 0x12;

}  // namespace clifft
