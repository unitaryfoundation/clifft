#pragma once

// Immutable sampling preparation for one requested-output choice of a compiled
// plan. Aggregate survivor counts run a count-only plan that expands record
// snapshots and omits records, unselected detectors, and expectation probes.
// Retained rows keep the compiled plan and all of its outputs. Early checks,
// the noise metadata of each mode, and the noise grouping key are built on
// first request and then shared by concurrent readers. The compiled plan
// outlives this object.

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
    SamplingPreparation(const sampling::ExecutablePlan& compiled, bool keep_records,
                        bool early_postselection);

    // The plan executors run: count-only for aggregate output, otherwise the compiled plan.
    [[nodiscard]] const sampling::ExecutablePlan& plan() const noexcept { return *plan_; }
    [[nodiscard]] const sampling::EarlyPostselection& checks() const;
    [[nodiscard]] const sampling::PreparedBatchNoise& noise(sampling::BatchNoiseMode mode) const;
    [[nodiscard]] const sampling::SharingKey& sharing_key() const;

  private:
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
