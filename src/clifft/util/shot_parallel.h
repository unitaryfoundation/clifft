#pragma once

// Shared cross-shot scheduling for sampling frontends. The worker factory
// creates every base context before this helper starts dispatch, then workers
// claim disjoint global shot ranges dynamically. Fixed-plan samplers fully
// allocate those contexts up front; trajectory samplers may grow them only at
// explicit continuation boundaries.

#include "clifft/util/intra_shot_parallel.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace clifft {

struct ShotRange {
    uint32_t begin = 0;
    uint32_t end = 0;
};

// requested_threads == 0 selects the implementation-reported hardware
// concurrency. Environments without native threads use a serial budget.
inline uint32_t resolve_thread_budget(uint32_t requested_threads) noexcept {
#if defined(__EMSCRIPTEN__)
    (void)requested_threads;
    return 1;
#else
    if (requested_threads == 0) {
        const uint32_t concurrency = std::thread::hardware_concurrency();
        if (concurrency != 0) {
            return concurrency;
        }
        return 1;
    }
    return requested_threads;
#endif
}

// A shot batch never creates more cross-shot workers than shots.
inline uint32_t resolve_shot_worker_count(uint32_t shots, uint32_t requested_threads) noexcept {
    return std::min(resolve_thread_budget(requested_threads), shots);
}

inline uint32_t shot_chunk_size(uint32_t shots, uint32_t workers) noexcept {
    assert(workers != 0 && "shot chunking requires at least one worker");
    constexpr uint64_t kTargetChunksPerWorker = 8;
    const uint64_t target_chunks = static_cast<uint64_t>(workers) * kTargetChunksPerWorker;
    return static_cast<uint32_t>(
        std::max<uint64_t>(1, (static_cast<uint64_t>(shots) + target_chunks - 1) / target_chunks));
}

// make_worker(index) runs for every worker before any range is dispatched.
// run_range(worker_handle, range) may execute concurrently for different
// handles. If it throws, remaining work is cancelled, all threads are joined,
// and the first exception is rethrown on the calling thread.
// Internal linkage keeps the target-local OpenMP configuration from changing a
// shared template definition in consumers that do not inherit the core define.
template <typename MakeWorker, typename RunRange>
static auto run_shot_ranges(uint32_t shots, uint32_t requested_threads, MakeWorker&& make_worker,
                            RunRange&& run_range) {
    using WorkerHandle = std::remove_cvref_t<std::invoke_result_t<MakeWorker, uint32_t>>;
    const uint32_t worker_count = resolve_shot_worker_count(shots, requested_threads);
    std::vector<WorkerHandle> workers;
    workers.reserve(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(std::invoke(make_worker, worker));
    }
    if (worker_count == 0) {
        return workers;
    }
    if (worker_count == 1) {
        std::invoke(run_range, workers[0], ShotRange{0, shots});
        return workers;
    }

#if defined(__EMSCRIPTEN__)
    std::invoke(run_range, workers[0], ShotRange{0, shots});
#else
    const uint32_t chunk_size = shot_chunk_size(shots, worker_count);
    std::atomic<uint64_t> next_shot{0};
    std::atomic<bool> cancelled{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker_loop = [&](uint32_t worker) noexcept {
        try {
            while (!cancelled.load(std::memory_order_relaxed)) {
                const uint64_t begin = next_shot.fetch_add(chunk_size, std::memory_order_relaxed);
                if (begin >= shots) {
                    return;
                }
                const uint64_t end = std::min<uint64_t>(begin + chunk_size, shots);
                std::invoke(run_range, workers[worker],
                            ShotRange{static_cast<uint32_t>(begin), static_cast<uint32_t>(end)});
            }
        } catch (...) {
            {
                std::lock_guard lock(error_mutex);
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }
            cancelled.store(true, std::memory_order_relaxed);
        }
    };

    if (openmp_process_binding_active()) {
#if defined(CLIFFT_USE_OPENMP)
        const int omp_worker_count = static_cast<int>(std::min<uint32_t>(
            worker_count, static_cast<uint32_t>(std::numeric_limits<int>::max())));
#pragma omp parallel num_threads(omp_worker_count)
        { worker_loop(static_cast<uint32_t>(omp_get_thread_num())); }
#endif
    } else {
        // std::jthread is unavailable on some of the minimum compiler and standard
        // library versions we support. This lifetime helper ensures every std::thread
        // is joined, including when constructing a later thread throws after earlier
        // workers have started.
        struct JoiningThreads {
            std::vector<std::thread>& threads;

            ~JoiningThreads() {
                for (std::thread& thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(worker_count - 1);
        {
            JoiningThreads joining{threads};
            try {
                for (uint32_t worker = 1; worker < worker_count; ++worker) {
                    threads.emplace_back(worker_loop, worker);
                }
            } catch (...) {
                cancelled.store(true, std::memory_order_relaxed);
                throw;
            }
            worker_loop(0);
        }
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
#endif
    return workers;
}

}  // namespace clifft
