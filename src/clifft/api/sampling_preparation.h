#pragma once

// Preparation for one requested-output choice of a compiled plan.
// Record snapshots expand into affine expressions when their rows are omitted.
// Early checks, noise metadata, and grouping keys are built on first request
// and shared by concurrent readers. Idle shared workers are pooled per active
// execution configuration up to a byte cap, checked out exclusively by each
// call, and released with this object. The compiled plan outlives this object.

#include "clifft/sampling/batch/classical.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/early_postselection.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/sharing_executor.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace clifft {

// Upper bound on idle shared-worker storage retained per SamplingPreparation.
inline constexpr uint64_t kMaxRetainedSharedWorkerBytes = 512 * 1024 * 1024;

// Configuration that determines whether idle shared workers can serve a call.
struct SharedWorkerKey {
    sampling::BatchOutputMode output_mode = sampling::BatchOutputMode::AggregateSurvivors;
    sampling::BatchNoiseMode noise_mode = sampling::BatchNoiseMode::SiteSparse;
    uint32_t lane_capacity = 0;
    uint64_t max_worker_state_bytes = 0;
    bool group_by_noise = false;

    bool operator==(const SharedWorkerKey&) const = default;
};

// Reusable per-thread execution state for shared sampling calls. A call borrows
// exclusive ownership for its workers before dispatch and returns them on exit.
struct SharedWorker {
    using Executor = std::variant<sampling::BatchExecutor, sampling::SharingExecutor>;

    Executor executor;
    uint64_t classical_bytes = 0;
    uint32_t passed = 0;
    uint32_t logical_errors = 0;
    std::vector<uint64_t> observable_ones;

    SharedWorker(const sampling::ExecutablePlan& plan, const sampling::PreparedBatchNoise& noise,
                 const sampling::EarlyPostselection* checks, const sampling::SharingKey* key,
                 const SharedWorkerKey& config);

    [[nodiscard]] uint64_t retained_bytes() const noexcept;
};

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

    // Checks out up to count matching idle workers (all returned entries are non-null).
    [[nodiscard]] std::vector<std::unique_ptr<SharedWorker>> borrow_shared_workers(
        const SharedWorkerKey& key, uint32_t count) const;
    // Returns completed workers for reuse up to kMaxRetainedSharedWorkerBytes.
    // Returning a different configuration replaces any previously cached pool.
    void return_shared_workers(const SharedWorkerKey& key,
                               std::vector<std::unique_ptr<SharedWorker>> workers) const;
    [[nodiscard]] uint64_t retained_shared_worker_bytes() const noexcept;
    [[nodiscard]] size_t retained_shared_worker_count() const noexcept;

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

    // Declared last so idle workers destruct before the plans, checks, noise,
    // and grouping keys they reference.
    mutable std::mutex worker_pool_mutex_;
    mutable std::optional<SharedWorkerKey> worker_pool_key_;
    mutable std::vector<std::unique_ptr<SharedWorker>> idle_workers_;
    mutable uint64_t retained_worker_bytes_ = 0;
};

}  // namespace clifft
