#pragma once

#include "clifft/sampling/executable_plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

// Sampling pipeline:
//   optimized HirModule -> SamplingPlan -> ExecutablePlan -> Executor -> results
// Planning produces semantic actions, lowering prepares fixed CPU descriptors,
// Executor owns mutable state for one shot, and the functions below drive
// repeated shots and collect their outputs.

// Collected row-major outputs from ordinary or fixed-fault-count sampling.
struct SamplingResult {
    std::vector<uint8_t> measurements;
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
    std::vector<double> exp_vals;
};

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

// Samples a fixed number of shots into row-major visible-record storage. The
// plan and executor are prepared once, and all output is allocated before the
// first shot enters hot execution. Plans with presampled symbols are rejected
// until their sampling distribution is part of the executable contract.
[[nodiscard]] std::vector<uint8_t> sample_records(const ExecutablePlan& plan, uint32_t shots,
                                                  std::optional<uint64_t> seed = std::nullopt);

// Replays each row-major visible record and returns its joint log probability.
// Unreachable records map to the lowest finite double because release builds
// assume finite arithmetic. Plans with presampled symbols or hidden records
// are rejected because this API does not yet marginalize over either source
// of hidden stochastic state.
[[nodiscard]] std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                                           std::span<const uint8_t> forced_records,
                                                           size_t num_records);

[[nodiscard]] SamplingResult sample(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_survivors(const ExecutablePlan& plan, uint32_t shots,
                                                      std::optional<uint64_t> seed = std::nullopt,
                                                      bool keep_records = false);

[[nodiscard]] SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                      std::optional<uint64_t> seed = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_k_survivors(const ExecutablePlan& plan, uint32_t shots,
                                                        uint32_t k,
                                                        std::optional<uint64_t> seed = std::nullopt,
                                                        bool keep_records = false);

}  // namespace clifft::sampling
