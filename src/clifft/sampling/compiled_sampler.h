#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/output_writer.h"
#include "clifft/sampling/results.h"
#include "clifft/util/shot_seed.h"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace clifft::sampling {

// One file emitted by CompiledSampler::sample_write. Sources are concatenated
// in order and may repeat, matching detector samplers that prepend or append
// observables while also writing them to a separate file.
struct SamplingFileOutput {
    std::ostream* output = nullptr;
    SamplingFileFormat format = SamplingFileFormat::Format01;
    std::span<const SamplingBitSource> sources;
};

// Stateful sampler used by Stim-compatible facades. The executable
// plan, executor contexts, worker scratch, and sampler RNG root outlive every
// sample call. Calls are serialized so one sampler advances one reproducible
// stream even when Python releases the GIL.
class CompiledSampler {
  public:
    CompiledSampler(std::shared_ptr<const ExecutablePlan> plan,
                    SamplingOutputSelection available_outputs,
                    std::optional<uint64_t> seed = std::nullopt, uint32_t threads = 1,
                    std::optional<uint32_t> batch_size = std::nullopt);
    ~CompiledSampler();

    CompiledSampler(const CompiledSampler&) = delete;
    CompiledSampler& operator=(const CompiledSampler&) = delete;
    CompiledSampler(CompiledSampler&&) = delete;
    CompiledSampler& operator=(CompiledSampler&&) = delete;

    void sample(uint32_t shots, SamplingOutputBuffer output);
    [[nodiscard]] uint32_t sample_survivors(uint32_t shots, SamplingOutputBuffer output);
    void sample_write(uint32_t shots, std::span<const SamplingFileOutput> outputs);

    [[nodiscard]] const ExecutablePlan& plan() const noexcept { return *plan_; }
    [[nodiscard]] SamplingOutputSelection available_outputs() const noexcept {
        return available_outputs_;
    }
    [[nodiscard]] uint32_t lane_capacity() const noexcept { return lane_capacity_; }
    [[nodiscard]] uint32_t worker_count() const noexcept {
        return static_cast<uint32_t>(workers_.size());
    }
    [[nodiscard]] uint64_t calls_completed() const;

  private:
    struct Worker;

    void execute_rows(const SeedRoot& root, uint32_t first_root_shot, uint32_t shots,
                      SamplingOutputBuffer output);
    [[nodiscard]] uint32_t execute_survivors(const SeedRoot& root, uint32_t shots,
                                             SamplingOutputBuffer output);

    std::shared_ptr<const ExecutablePlan> plan_;
    SamplingOutputSelection available_outputs_;
    SeedRoot seed_root_{};
    uint32_t lane_capacity_ = 1;
    std::vector<std::unique_ptr<Worker>> workers_;
    mutable std::mutex mutex_;
    uint64_t calls_completed_ = 0;
};

}  // namespace clifft::sampling
