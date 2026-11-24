#include "clifft/api/sampling_preparation.h"

#include "clifft/sampling/batch/policy.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <cassert>
#include <list>
#include <new>
#include <stdexcept>
#include <unordered_map>

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
    // Plans without coefficient evolution or a runtime Clifford frame have
    // nothing to share. Their packed executor applies classical checks without
    // per-shot group bookkeeping.
    if (!sampling::shares_groups(plan)) {
        return SharedWorker::Executor(std::in_place_type<sampling::BatchExecutor>, plan,
                                      config.lane_capacity, config.output_mode,
                                      sampling::BatchSamplingMode::Ordinary, &noise);
    }
    const uint32_t initial_states = static_cast<uint32_t>(std::min<uint64_t>(
        {config.lane_capacity, kMaxInitialSharedStates,
         std::max<uint64_t>(
             1, kInitialSharedStateBytes / sampling::SharingExecutor::group_storage_bytes(
                                               plan, plan.initial_capacity_width()))}));
    return SharedWorker::Executor(std::in_place_type<sampling::SharingExecutor>, plan, noise,
                                  *checks, config.lane_capacity, initial_states, true, key,
                                  config.max_worker_state_bytes, config.output_mode);
}

uint64_t shared_worker_classical_bytes(const sampling::ExecutablePlan& plan,
                                       const sampling::PreparedBatchNoise& noise,
                                       const sampling::SharingKey* key,
                                       const SharedWorkerKey& config, size_t counter_capacity) {
    const uint64_t counter_bytes = saturating_multiply_u64(counter_capacity, sizeof(uint64_t));
    const uint64_t executor_bytes =
        sampling::shares_groups(plan)
            ? sampling::SharingExecutor::classical_storage_bytes(
                  plan, noise, key, config.lane_capacity, config.output_mode)
            : sampling::batch_detail::batch_worker_storage_bytes(
                  plan, config.lane_capacity, config.output_mode,
                  sampling::BatchSamplingMode::Ordinary, &noise);
    return saturating_add_u64(sizeof(SharedWorker),
                              saturating_add_u64(executor_bytes, counter_bytes));
}

// Process-wide LRU cache of idle shared workers keyed by owning
// SamplingPreparation. Every node in lru_ holds at least one worker; front is
// most recently used. Evicted and replaced workers destruct outside mutex_.
class SharedWorkerCache {
  public:
    std::vector<std::unique_ptr<SharedWorker>> borrow(const SamplingPreparation* owner,
                                                      const SharedWorkerKey& key, uint32_t count) {
        std::vector<std::unique_ptr<SharedWorker>> out;
        if (count == 0) {
            return out;
        }
        std::vector<std::unique_ptr<SharedWorker>> discarded;
        {
            const std::lock_guard lock(mutex_);
            const auto found = index_.find(owner);
            if (found == index_.end()) {
                return out;
            }
            auto it = found->second;
            if (it->key != key) {
                discarded = detach_entry(found);
                return out;
            }
            const size_t take = std::min<size_t>(count, it->workers.size());
            out.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                auto worker = std::move(it->workers.back());
                it->workers.pop_back();
                const uint64_t bytes = worker->retained_bytes();
                assert(bytes <= it->bytes && bytes <= total_bytes_);
                it->bytes -= bytes;
                total_bytes_ -= bytes;
                out.push_back(std::move(worker));
            }
            if (it->workers.empty()) {
                assert(it->bytes == 0);
                lru_.erase(it);
                index_.erase(found);
            } else {
                lru_.splice(lru_.begin(), lru_, it);
            }
        }
        return out;
    }

    void return_workers(const SamplingPreparation* owner, const SharedWorkerKey& key,
                        std::vector<std::unique_ptr<SharedWorker>> workers) noexcept {
        std::vector<std::unique_ptr<SharedWorker>> evicted;
        {
            const std::lock_guard lock(mutex_);
            auto found = index_.find(owner);
            if (found != index_.end()) {
                if (found->second->key != key) {
                    evicted = detach_entry(found);
                    found = index_.end();
                } else {
                    lru_.splice(lru_.begin(), lru_, found->second);
                }
            }
            for (auto& worker : workers) {
                if (!worker) {
                    continue;
                }
                const uint64_t bytes = worker->retained_bytes();
                const uint64_t owner_bytes = found == index_.end() ? 0 : found->second->bytes;
                if (bytes > budget_bytes_ || owner_bytes > budget_bytes_ - bytes) {
                    continue;
                }
                try {
                    while (total_bytes_ > budget_bytes_ - bytes && !lru_.empty() &&
                           lru_.back().owner != owner) {
                        evicted.reserve(evicted.size() + 1);
                        evicted.push_back(pop_tail_worker());
                    }
                    assert(total_bytes_ <= budget_bytes_ - bytes);
                    if (found == index_.end()) {
                        CacheEntry entry;
                        entry.owner = owner;
                        entry.key = key;
                        entry.workers.reserve(1);
                        lru_.push_front(std::move(entry));
                        try {
                            found = index_.emplace(owner, lru_.begin()).first;
                        } catch (...) {
                            lru_.pop_front();
                            throw;
                        }
                        auto& inserted = *found->second;
                        inserted.workers.push_back(std::move(worker));
                        inserted.bytes = bytes;
                        total_bytes_ += bytes;
                    } else {
                        auto& entry = *found->second;
                        entry.workers.reserve(entry.workers.size() + 1);
                        entry.workers.push_back(std::move(worker));
                        entry.bytes += bytes;
                        total_bytes_ += bytes;
                    }
                } catch (const std::bad_alloc&) {
                    // Best-effort retention: drop remaining workers on OOM
                    // without discarding the completed sampling result.
                    break;
                }
            }
        }
        // Evicted and unretained workers (remaining in workers) destruct outside mutex_.
    }

    void release(const SamplingPreparation* owner) noexcept {
        std::vector<std::unique_ptr<SharedWorker>> discarded;
        {
            const std::lock_guard lock(mutex_);
            const auto found = index_.find(owner);
            if (found == index_.end()) {
                return;
            }
            discarded = detach_entry(found);
        }
    }

    uint64_t retained_bytes(const SamplingPreparation* owner) const noexcept {
        const std::lock_guard lock(mutex_);
        const auto found = index_.find(owner);
        return found == index_.end() ? 0 : found->second->bytes;
    }

    size_t retained_count(const SamplingPreparation* owner) const noexcept {
        const std::lock_guard lock(mutex_);
        const auto found = index_.find(owner);
        return found == index_.end() ? 0 : found->second->workers.size();
    }

    uint64_t total_bytes() const noexcept {
        const std::lock_guard lock(mutex_);
        return total_bytes_;
    }

    uint64_t set_budget(uint64_t new_budget) noexcept {
        std::vector<std::unique_ptr<SharedWorker>> evicted;
        uint64_t previous = 0;
        {
            const std::lock_guard lock(mutex_);
            previous = budget_bytes_;
            budget_bytes_ = new_budget;
            while (total_bytes_ > budget_bytes_ && !lru_.empty()) {
                auto old = pop_tail_worker();
                try {
                    evicted.push_back(std::move(old));
                } catch (const std::bad_alloc&) {
                    // Best-effort deferred destruction when bookkeeping hits OOM.
                }
            }
        }
        return previous;
    }

  private:
    struct CacheEntry {
        const SamplingPreparation* owner = nullptr;
        SharedWorkerKey key;
        std::vector<std::unique_ptr<SharedWorker>> workers;
        uint64_t bytes = 0;
    };

    using Index = std::unordered_map<const SamplingPreparation*, std::list<CacheEntry>::iterator>;

    std::vector<std::unique_ptr<SharedWorker>> detach_entry(Index::iterator found) noexcept {
        auto it = found->second;
        assert(it->bytes <= total_bytes_);
        total_bytes_ -= it->bytes;
        auto workers = std::move(it->workers);
        lru_.erase(it);
        index_.erase(found);
        return workers;
    }

    std::unique_ptr<SharedWorker> pop_tail_worker() noexcept {
        auto& victim = lru_.back();
        assert(!victim.workers.empty());
        auto old = std::move(victim.workers.back());
        victim.workers.pop_back();
        const uint64_t old_bytes = old->retained_bytes();
        assert(old_bytes <= victim.bytes && old_bytes <= total_bytes_);
        victim.bytes -= old_bytes;
        total_bytes_ -= old_bytes;
        if (victim.workers.empty()) {
            assert(victim.bytes == 0);
            index_.erase(victim.owner);
            lru_.pop_back();
        }
        return old;
    }

    mutable std::mutex mutex_;
    std::list<CacheEntry> lru_;
    Index index_;
    uint64_t total_bytes_ = 0;
    uint64_t budget_bytes_ = kMaxRetainedSharedWorkerBytes;
};

SharedWorkerCache& shared_worker_cache() {
    // Immortal singleton avoids static-destruction-order hazards when Programs
    // outlive function-local statics during process teardown.
    static auto* cache = new SharedWorkerCache();
    return *cache;
}

}  // namespace

SharedWorker::SharedWorker(const sampling::ExecutablePlan& plan,
                           const sampling::PreparedBatchNoise& noise,
                           const sampling::EarlyPostselection* checks,
                           const sampling::SharingKey* key, const SharedWorkerKey& config)
    : executor(make_shared_executor(plan, noise, checks, key, config)),
      observable_ones(plan.num_observables()) {
    classical_bytes =
        shared_worker_classical_bytes(plan, noise, key, config, observable_ones.capacity());
}

uint64_t SharedWorker::retained_bytes() const noexcept {
    const auto* sharing = std::get_if<sampling::SharingExecutor>(&executor);
    return saturating_add_u64(classical_bytes, sharing ? sharing->state_pool_bytes() : 0);
}

uint64_t total_retained_shared_worker_bytes() noexcept {
    return shared_worker_cache().total_bytes();
}

namespace detail {

ScopedSharedWorkerCacheBudget::ScopedSharedWorkerCacheBudget(uint64_t budget_bytes)
    : previous_budget_(shared_worker_cache().set_budget(budget_bytes)) {}

ScopedSharedWorkerCacheBudget::~ScopedSharedWorkerCacheBudget() {
    (void)shared_worker_cache().set_budget(previous_budget_);
}

}  // namespace detail

SamplingPreparation::SamplingPreparation(const sampling::ExecutablePlan& compiled,
                                         sampling::SamplingOutputs outputs,
                                         bool early_postselection, bool survivor_counts)
    : early_postselection_(early_postselection),
      compact_(outputs.records == sampling::RecordOutputs::All
                   ? std::nullopt
                   : std::optional(
                         compiled.compact_outputs(outputs, early_postselection, survivor_counts))),
      plan_(compact_ ? &*compact_ : &compiled) {}

SamplingPreparation::~SamplingPreparation() {
    // Release any cached idle workers before compact_, checks_, noise, and
    // sharing_key_ members are destroyed.
    shared_worker_cache().release(this);
}

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
    return shared_worker_cache().borrow(this, key, count);
}

void SamplingPreparation::return_shared_workers(
    const SharedWorkerKey& key, std::vector<std::unique_ptr<SharedWorker>> workers) const noexcept {
    shared_worker_cache().return_workers(this, key, std::move(workers));
}

uint64_t SamplingPreparation::retained_shared_worker_bytes() const noexcept {
    return shared_worker_cache().retained_bytes(this);
}

size_t SamplingPreparation::retained_shared_worker_count() const noexcept {
    return shared_worker_cache().retained_count(this);
}

}  // namespace clifft
