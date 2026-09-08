#pragma once

#include <cstdint>
#include <vector>

namespace clifft::sampling {

// Selects which row-major output matrices a sampling request materializes.
// Executors may still retain internal records needed to evaluate requested
// detector or observable parities.
struct SamplingOutputSelection {
    bool measurements = false;
    bool detectors = false;
    bool observables = false;
    bool exp_vals = false;

    [[nodiscard]] constexpr bool any() const noexcept {
        return measurements || detectors || observables || exp_vals;
    }

    [[nodiscard]] static constexpr SamplingOutputSelection all() noexcept {
        return {.measurements = true, .detectors = true, .observables = true, .exp_vals = true};
    }
};

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
