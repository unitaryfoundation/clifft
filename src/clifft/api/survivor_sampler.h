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

// Retains classical batches and shared coefficient storage across count-only
// sampling calls. Program and prepared plans must outlive this sampler.
class SurvivorSampler {
  public:
    SurvivorSampler(const Program& program, uint32_t batch_size = 8192, uint32_t threads = 1,
                    uint32_t initial_states = 128,
                    sampling::BatchNoiseMode noise_mode = sampling::BatchNoiseMode::SiteSparse,
                    bool sharing = true, bool early_postselection = true,
                    bool compact_outputs = false, bool group_by_noise = false);
    ~SurvivorSampler();

    [[nodiscard]] sampling::SamplingSurvivorResult sample(
        uint32_t shots, std::optional<uint64_t> seed = std::nullopt);
    [[nodiscard]] const Program& program() const noexcept { return *program_; }

  private:
    struct Worker;
    const Program* program_;
    uint32_t batch_size_;
    std::optional<sampling::ExecutablePlan> compact_plan_;
    const sampling::ExecutablePlan& plan_;
    sampling::PreparedBatchNoise noise_;
    sampling::EarlyPostselection checks_;
    std::optional<sampling::SharingKey> key_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex mutex_;
};

}  // namespace clifft
