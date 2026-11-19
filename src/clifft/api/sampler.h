#pragma once

// Native sampling selects a prepared executor from the program and this call.
// Program owns immutable preparation; workers and collected rows belong to the
// call. The plan-only sampler remains available to WebAssembly.

#include "clifft/api/program.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/sharing_executor.h"

namespace clifft {

// Unset expert overrides select deterministic policies for this call.
struct SamplingOptions {
    uint32_t threads = 1;
    std::optional<sampling::ThreadLayout> thread_layout = std::nullopt;
    std::optional<uint32_t> batch_size = std::nullopt;
    std::optional<double> tuning_budget_seconds = std::nullopt;
    std::optional<bool> sharing = std::nullopt;
    std::optional<bool> group_by_noise = std::nullopt;
    std::optional<sampling::BatchNoiseMode> noise_preparation = std::nullopt;
    uint64_t max_worker_state_bytes = sampling::kDefaultSharedWorkerStateBudget;
};

[[nodiscard]] sampling::SamplingResult sample(const Program& program, uint32_t shots,
                                              std::optional<uint64_t> seed = std::nullopt,
                                              sampling::SamplingOutputs outputs = {},
                                              const SamplingOptions& options = {});

[[nodiscard]] sampling::SamplingSurvivorResult sample_survivors(
    const Program& program, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    sampling::SamplingOutputs outputs = false, const SamplingOptions& options = {});

}  // namespace clifft
