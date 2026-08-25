#pragma once

#include <cstdint>
#include <vector>

namespace clifft::sampling {

// Backend-neutral row-major outputs from ordinary sampling.
struct SamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

// Backend-neutral outputs from postselected survivor sampling.
struct SamplingSurvivorResult {
    uint32_t total_shots = 0;
    uint32_t passed_shots = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

}  // namespace clifft::sampling
