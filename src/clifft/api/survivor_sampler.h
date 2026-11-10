#pragma once

#include "clifft/api/program.h"
#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/results.h"
#include "clifft/sampling/sharing_executor.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace clifft {

// Retains workers and their classical batches and shared coefficient storage
// across count-only sampling calls. The compiled plans and sampling preparation
// are shared with the Program. An unset batch_size selects the capacity and
// worker count from each call's shots, rebuilding workers when they change.
class SurvivorSampler {
  public:
    SurvivorSampler(const Program& program, std::optional<uint32_t> batch_size = 8192,
                    uint32_t threads = 1, uint32_t initial_states = 128,
                    sampling::BatchNoiseMode noise_mode = sampling::BatchNoiseMode::SiteSparse,
                    bool sharing = true, bool early_postselection = true,
                    bool group_by_noise = false,
                    uint64_t max_worker_state_bytes = sampling::kDefaultSharedWorkerStateBudget,
                    std::optional<double> tuning_budget_seconds = std::nullopt);
    ~SurvivorSampler();

    [[nodiscard]] sampling::SamplingSurvivorResult sample(
        uint32_t shots, std::optional<uint64_t> seed = std::nullopt);
    [[nodiscard]] const Program& program() const noexcept { return program_; }

  private:
    struct Worker;
    void prepare_workers(sampling::BatchExecutionPolicy policy);
    [[nodiscard]] sampling::SamplingSurvivorResult sample_batches(uint32_t shots,
                                                                  std::optional<uint64_t> seed);
    void tune(uint32_t shots, std::optional<uint64_t> seed);
    Program program_;
    // Capacity of the retained workers; zero before the first automatic call.
    uint32_t batch_size_ = 0;
    // Set while each call selects its own capacity.
    bool call_sized_;
    uint32_t worker_count_;
    uint32_t initial_states_;
    uint64_t worker_state_bytes_;
    // Largest automatic capacity within the per-worker budgets.
    uint32_t safe_capacity_ = 0;
    bool sharing_;
    std::optional<double> tuning_budget_seconds_;
    std::optional<sampling::BatchTuningReport> tuning_report_;
    const sampling::ExecutablePlan* plan_ = nullptr;
    const sampling::PreparedBatchNoise* noise_ = nullptr;
    // Width-zero plans run packed classical execution without early checks or grouping.
    const sampling::EarlyPostselection* checks_ = nullptr;
    const sampling::SharingKey* key_ = nullptr;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex mutex_;
};

}  // namespace clifft
