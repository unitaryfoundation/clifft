#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clifft::sampling {

struct BatchTuningTrial {
    uint32_t batch_size = 1;
    uint32_t shot_workers = 1;
    uint64_t warmup_shots = 0;
    uint64_t shots = 0;
    double setup_seconds = 0;
    double elapsed_seconds = 0;
};

struct BatchTuningReport {
    uint32_t batch_size = 1;
    uint32_t baseline_batch_size = 1;
    uint32_t shot_workers = 1;
    uint32_t intra_shot_workers = 1;
    uint64_t trial_shots = 0;
    double elapsed_seconds = 0;
    std::string stop_reason;
    std::vector<BatchTuningTrial> trials;
};

// Backend-neutral row-major outputs from ordinary sampling.
struct SamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
    std::optional<BatchTuningReport> batch_tuning;
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
    std::optional<BatchTuningReport> batch_tuning;
};

}  // namespace clifft::sampling
