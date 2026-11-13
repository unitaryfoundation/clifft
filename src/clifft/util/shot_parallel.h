#pragma once

// Shared cross-shot scheduling for sampling frontends. Workers claim disjoint
// global ranges dynamically. Their contexts are initialized before dispatch,
// either on the calling thread or on each worker's execution thread. Trajectory
// samplers may grow storage only at explicit continuation boundaries.

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

enum class WorkerInitialization { BeforeDispatch, OnWorkerThread };

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

inline uint32_t shot_chunk_size(uint32_t shots, uint32_t workers,
                                uint32_t minimum_chunk_size = 1) noexcept {
    assert(workers != 0 && minimum_chunk_size != 0 &&
           "shot chunking requires workers and a positive minimum chunk");
    // Several claims per worker redistribute uneven shot costs without paying
    // for one atomic scheduler operation per shot.
    constexpr uint64_t kTargetChunksPerWorker = 8;
    const uint64_t target_chunks = static_cast<uint64_t>(workers) * kTargetChunksPerWorker;
    const uint64_t target_size = std::max<uint64_t>(
        minimum_chunk_size,
        std::max<uint64_t>(1, (static_cast<uint64_t>(shots) + target_chunks - 1) / target_chunks));
    if (target_size >= shots) {
        return shots;
    }
    const uint64_t aligned_size =
        ((target_size + minimum_chunk_size - 1) / minimum_chunk_size) * minimum_chunk_size;
    return static_cast<uint32_t>(std::min<uint64_t>(aligned_size, shots));
}

// By default make_worker(index) runs for every worker before any dispatch.
// OnWorkerThread constructs each context on the thread that uses it, before
// that thread claims any ranges. Factories then must allow concurrent calls.
// run_range(worker_handle, range) may execute concurrently for different
// handles. If it throws, remaining work is cancelled, all threads are joined,
// and the first exception is rethrown on the calling thread.
// Internal linkage keeps the target-local OpenMP configuration from changing a
// shared template definition in consumers that do not inherit the core define.
template <typename MakeWorker, typename RunRange>
static auto run_shot_ranges(
    uint32_t shots, uint32_t requested_threads, MakeWorker&& make_worker, RunRange&& run_range,
    uint32_t minimum_chunk_size = 1,
    WorkerInitialization initialization = WorkerInitialization::BeforeDispatch) {
    using WorkerHandle = std::remove_cvref_t<std::invoke_result_t<MakeWorker, uint32_t>>;
    const uint32_t worker_count = resolve_shot_worker_count(shots, requested_threads);
    std::vector<WorkerHandle> workers(worker_count);
    if (initialization == WorkerInitialization::BeforeDispatch || worker_count == 1) {
        for (uint32_t worker = 0; worker < worker_count; ++worker) {
            workers[worker] = std::invoke(make_worker, worker);
        }
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
    const uint32_t chunk_size = shot_chunk_size(shots, worker_count, minimum_chunk_size);
    std::atomic<uint64_t> next_shot{0};
    std::atomic<bool> cancelled{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker_loop = [&](uint32_t worker) noexcept {
        try {
            if (initialization == WorkerInitialization::OnWorkerThread) {
                workers[worker] = std::invoke(make_worker, worker);
            }
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
