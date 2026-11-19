#pragma once

// Immutable preparation for one requested-output choice of a compiled plan.
// Record snapshots expand into affine expressions when their rows are omitted.
// Early checks and noise metadata are built on first request and shared by
// concurrent readers. The compiled plan outlives this object.

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/sharing_executor.h"

#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace clifft {

namespace detail {

// Builds a value on first request and shares it afterwards. A failed build
// leaves the slot empty, so a later request tries again.
template <typename T>
class LazyPrepared {
  public:
    template <typename... Args>
    [[nodiscard]] const T& get(Args&&... args) const {
        const std::lock_guard lock(mutex_);
        if (!value_) {
            value_ = std::make_unique<const T>(std::forward<Args>(args)...);
        }
        return *value_;
    }

  private:
    mutable std::mutex mutex_;
    mutable std::unique_ptr<const T> value_;
};

}  // namespace detail

class SamplingPreparation {
  public:
    SamplingPreparation(const sampling::ExecutablePlan& compiled, sampling::SamplingOutputs outputs,
                        bool early_postselection, bool survivor_counts);

    // The plan executors run, with unused output actions removed.
    [[nodiscard]] const sampling::ExecutablePlan& plan() const noexcept { return *plan_; }
    [[nodiscard]] const sampling::EarlyPostselection& checks() const;
    [[nodiscard]] const sampling::PreparedBatchNoise& noise(sampling::BatchNoiseMode mode) const;
    [[nodiscard]] const sampling::SharingKey& sharing_key() const;

    // Profile a fixed, independent batch. Both competing plans use the same
    // lane count; sampling calls cache the profile for their prepared outputs.
    static constexpr uint64_t kProbeStateBytes = 32 * 1024 * 1024;
    [[nodiscard]] uint32_t probe_capacity() const;
    [[nodiscard]] sampling::TrajectoryWork measure_work(uint32_t lanes) const;
    [[nodiscard]] const sampling::TrajectoryWork& trajectory_work() const;

  private:
    mutable std::once_flag profile_once_;
    mutable sampling::TrajectoryWork profile_;
    bool early_postselection_;
    std::optional<sampling::ExecutablePlan> compact_;
    const sampling::ExecutablePlan* plan_;
    detail::LazyPrepared<sampling::EarlyPostselection> checks_;
    detail::LazyPrepared<sampling::PreparedBatchNoise> dense_noise_;
    detail::LazyPrepared<sampling::PreparedBatchNoise> sparse_noise_;
    detail::LazyPrepared<sampling::PreparedBatchNoise> site_sparse_noise_;
    detail::LazyPrepared<sampling::PreparedBatchNoise> site_rows_noise_;
    detail::LazyPrepared<sampling::SharingKey> sharing_key_;
};

}  // namespace clifft
