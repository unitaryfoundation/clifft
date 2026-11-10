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
// are shared with the Program.
class SurvivorSampler {
  public:
    SurvivorSampler(const Program& program, std::optional<uint32_t> batch_size = 8192,
                    uint32_t threads = 1, uint32_t initial_states = 128,
                    sampling::BatchNoiseMode noise_mode = sampling::BatchNoiseMode::SiteSparse,
                    bool sharing = true, bool early_postselection = true,
                    bool group_by_noise = false,
                    uint64_t max_state_bytes = sampling::kDefaultBatchTotalWorkerBudget,
                    std::optional<double> tuning_budget_seconds = std::nullopt);
    ~SurvivorSampler();

    [[nodiscard]] sampling::SamplingSurvivorResult sample(
        uint32_t shots, std::optional<uint64_t> seed = std::nullopt);
    [[nodiscard]] const Program& program() const noexcept { return program_; }

  private:
    struct Worker;
    void prepare_workers(uint32_t capacity);
    [[nodiscard]] sampling::SamplingSurvivorResult sample_batches(uint32_t shots,
                                                                  std::optional<uint64_t> seed);
    void tune(uint32_t shots, std::optional<uint64_t> seed);
    Program program_;
    uint32_t batch_size_;
    uint32_t worker_count_;
    uint32_t initial_states_;
    uint64_t worker_state_bytes_;
    bool sharing_;
    std::optional<double> tuning_budget_seconds_;
    std::optional<sampling::BatchTuningReport> tuning_report_;
    const SamplingPreparation& preparation_;
    const sampling::ExecutablePlan& plan_;
    const sampling::PreparedBatchNoise& noise_;
    // Width-zero plans run packed classical execution without early checks or grouping.
    const sampling::EarlyPostselection* checks_;
    const sampling::SharingKey* key_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex mutex_;
};

}  // namespace clifft
