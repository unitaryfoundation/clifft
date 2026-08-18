#pragma once

// Shared cross-shot scheduling for sampling frontends. The worker factory
// creates every fully allocated context before this helper starts dispatch,
// then workers claim disjoint global shot ranges dynamically.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <exception>
#include <functional>
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
// concurrency. A shot batch never creates more workers than shots, and
// environments without native threads use the serial implementation.
inline uint32_t resolve_shot_worker_count(uint32_t shots, uint32_t requested_threads) noexcept {
    if (shots == 0) {
        return 0;
    }
#if defined(__EMSCRIPTEN__)
    (void)requested_threads;
    return 1;
#else
    uint32_t workers = requested_threads;
    if (workers == 0) {
        workers = std::thread::hardware_concurrency();
        if (workers == 0) {
            workers = 1;
        }
    }
    return std::min(workers, shots);
#endif
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
template <typename MakeWorker, typename RunRange>
auto run_shot_ranges(uint32_t shots, uint32_t requested_threads, MakeWorker&& make_worker,
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

    // std::thread keeps the documented compiler-library floor. This guard is
    // needed because constructing a later thread can throw after earlier ones
    // have started; every exit must join those threads before destroying them.
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
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
#endif
    return workers;
}

}  // namespace clifft
