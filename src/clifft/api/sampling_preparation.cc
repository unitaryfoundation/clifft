#include "clifft/api/sampling_preparation.h"

#include "clifft/sampling/batch/policy.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <stdexcept>

namespace clifft {

namespace {

constexpr uint32_t kMaxInitialSharedStates = 128;
// Bound upfront state preallocation per worker on wide plans; additional states
// grow at explicit suspension boundaries when trajectories split.
constexpr uint64_t kInitialSharedStateBytes = 512 * 1024;

SharedWorker::Executor make_shared_executor(const sampling::ExecutablePlan& plan,
                                            const sampling::PreparedBatchNoise& noise,
                                            const sampling::EarlyPostselection* checks,
                                            const sampling::SharingKey* key,
                                            const SharedWorkerKey& config) {
    // Width-zero plans have no coefficient evolution to share. Their packed
    // executor applies classical checks without per-shot group bookkeeping.
    if (plan.peak_active_width() == 0) {
        return SharedWorker::Executor(std::in_place_type<sampling::BatchExecutor>, plan,
                                      config.lane_capacity, config.output_mode,
                                      sampling::BatchSamplingMode::Ordinary, &noise);
    }
    const uint32_t initial_states = static_cast<uint32_t>(std::min<uint64_t>(
        {config.lane_capacity, kMaxInitialSharedStates,
         std::max<uint64_t>(1, kInitialSharedStateBytes /
                                   sampling::State::storage_bytes(plan.peak_active_width()))}));
    return SharedWorker::Executor(std::in_place_type<sampling::SharingExecutor>, plan, noise,
                                  *checks, config.lane_capacity, initial_states, true, key,
                                  config.max_worker_state_bytes, config.output_mode);
}

uint64_t shared_worker_classical_bytes(const sampling::ExecutablePlan& plan,
                                       const sampling::PreparedBatchNoise& noise,
                                       const sampling::SharingKey* key,
                                       const SharedWorkerKey& config) {
    const uint64_t counter_bytes =
        saturating_multiply_u64(plan.num_observables(), sizeof(uint64_t));
    const uint64_t executor_bytes =
        plan.peak_active_width() == 0
            ? sampling::batch_detail::batch_worker_storage_bytes(
                  plan, config.lane_capacity, config.output_mode,
                  sampling::BatchSamplingMode::Ordinary, &noise)
            : sampling::SharingExecutor::classical_storage_bytes(
                  plan, noise, key, config.lane_capacity, config.output_mode);
    return saturating_add_u64(executor_bytes, counter_bytes);
}

}  // namespace

SharedWorker::SharedWorker(const sampling::ExecutablePlan& plan,
                           const sampling::PreparedBatchNoise& noise,
                           const sampling::EarlyPostselection* checks,
                           const sampling::SharingKey* key, const SharedWorkerKey& config)
    : executor(make_shared_executor(plan, noise, checks, key, config)),
      classical_bytes(shared_worker_classical_bytes(plan, noise, key, config)),
      observable_ones(plan.num_observables()) {}

uint64_t SharedWorker::retained_bytes() const noexcept {
    const auto* sharing = std::get_if<sampling::SharingExecutor>(&executor);
    return saturating_add_u64(classical_bytes, sharing ? sharing->state_pool_bytes() : 0);
}

SamplingPreparation::SamplingPreparation(const sampling::ExecutablePlan& compiled,
                                         sampling::SamplingOutputs outputs,
                                         bool early_postselection, bool survivor_counts)
    : early_postselection_(early_postselection),
      compact_(outputs.records == sampling::RecordOutputs::All
                   ? std::nullopt
                   : std::optional(
                         compiled.compact_outputs(outputs, early_postselection, survivor_counts))),
      plan_(compact_ ? &*compact_ : &compiled) {}

const sampling::EarlyPostselection& SamplingPreparation::checks() const {
    return checks_.get(*plan_, early_postselection_);
}

const sampling::PreparedBatchNoise& SamplingPreparation::noise(
    sampling::BatchNoiseMode mode) const {
    switch (mode) {
        case sampling::BatchNoiseMode::Dense:
            return dense_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::Sparse:
            return sparse_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::SiteSparse:
            return site_sparse_noise_.get(*plan_, mode);
        case sampling::BatchNoiseMode::SiteRows:
            return site_rows_noise_.get(*plan_, mode);
    }
    throw std::invalid_argument("unknown batch noise mode");
}

const sampling::SharingKey& SamplingPreparation::sharing_key() const {
    return sharing_key_.get(*plan_);
}

std::vector<std::unique_ptr<SharedWorker>> SamplingPreparation::borrow_shared_workers(
    const SharedWorkerKey& key, uint32_t count) const {
    std::vector<std::unique_ptr<SharedWorker>> out;
    if (count == 0) {
        return out;
    }
    const std::lock_guard lock(worker_pool_mutex_);
    if (worker_pool_key_ != key || idle_workers_.empty()) {
        return out;
    }
    const size_t take = std::min<size_t>(count, idle_workers_.size());
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        auto worker = std::move(idle_workers_.back());
        idle_workers_.pop_back();
        const uint64_t bytes = worker->retained_bytes();
        retained_worker_bytes_ =
            bytes <= retained_worker_bytes_ ? retained_worker_bytes_ - bytes : 0;
        out.push_back(std::move(worker));
    }
    return out;
}

void SamplingPreparation::return_shared_workers(
    const SharedWorkerKey& key, std::vector<std::unique_ptr<SharedWorker>> workers) const {
    std::vector<std::unique_ptr<SharedWorker>> replaced;
    {
        const std::lock_guard lock(worker_pool_mutex_);
        if (worker_pool_key_ != key) {
            replaced = std::move(idle_workers_);
            idle_workers_.clear();
            retained_worker_bytes_ = 0;
            worker_pool_key_ = key;
        }
        for (auto& worker : workers) {
            if (!worker) {
                continue;
            }
            const uint64_t bytes = worker->retained_bytes();
            if (bytes <= kMaxRetainedSharedWorkerBytes &&
                retained_worker_bytes_ <= kMaxRetainedSharedWorkerBytes - bytes) {
                retained_worker_bytes_ += bytes;
                idle_workers_.push_back(std::move(worker));
            }
        }
    }
    // Replaced and over-budget workers (remaining in workers) destruct outside the mutex.
}

uint64_t SamplingPreparation::retained_shared_worker_bytes() const noexcept {
    const std::lock_guard lock(worker_pool_mutex_);
    return retained_worker_bytes_;
}

size_t SamplingPreparation::retained_shared_worker_count() const noexcept {
    const std::lock_guard lock(worker_pool_mutex_);
    return idle_workers_.size();
}

}  // namespace clifft
