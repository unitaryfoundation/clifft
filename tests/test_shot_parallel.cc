#include "clifft/util/shot_parallel.h"

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>

TEST_CASE("Shot scheduler resolves bounded worker counts") {
    REQUIRE(clifft::resolve_shot_worker_count(0, 0) == 0);
    REQUIRE(clifft::resolve_shot_worker_count(1, 0) == 1);
    REQUIRE(clifft::resolve_shot_worker_count(3, 1) == 1);
    REQUIRE(clifft::resolve_shot_worker_count(3, 99) == 3);
    REQUIRE(clifft::shot_chunk_size(1000, 4) == 32);
    REQUIRE(clifft::shot_chunk_size(1000, 4, 64) == 64);
}

TEST_CASE("Shot scheduler constructs workers before dispatch and visits each shot") {
    constexpr uint32_t shots = 64;
    constexpr uint32_t worker_count = 4;
    std::atomic<uint32_t> constructed{0};
    std::atomic<bool> dispatched_early{false};
    std::array<std::atomic<uint32_t>, shots> visits{};

    auto workers = clifft::run_shot_ranges(
        shots, worker_count,
        [&](uint32_t worker) {
            constructed.fetch_add(1, std::memory_order_relaxed);
            return std::make_unique<uint32_t>(worker);
        },
        [&](const auto&, clifft::ShotRange range) {
            if (constructed.load(std::memory_order_relaxed) != worker_count) {
                dispatched_early.store(true, std::memory_order_relaxed);
            }
            for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                visits[shot].fetch_add(1, std::memory_order_relaxed);
            }
        });

    REQUIRE(workers.size() == worker_count);
    REQUIRE_FALSE(dispatched_early.load(std::memory_order_relaxed));
    for (const auto& count : visits) {
        REQUIRE(count.load(std::memory_order_relaxed) == 1);
    }
}

TEST_CASE("Shot scheduler joins workers before rethrowing a range error") {
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto wait_until_deadline = [&](const auto& predicate) {
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= wait_deadline) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    };

    std::atomic<bool> background_started{false};
    std::atomic<bool> caller_worker_started{false};
    std::atomic<bool> allow_background_finish{false};
    std::atomic<bool> background_finished{false};
    std::atomic<bool> caller_returned{false};
    std::exception_ptr caller_error;

    std::thread caller([&] {
        try {
            clifft::run_shot_ranges(
                32, 2, [](uint32_t worker) { return std::make_unique<uint32_t>(worker); },
                [&](const auto& worker, clifft::ShotRange) {
                    if (*worker == 1) {
                        background_started.store(true, std::memory_order_release);
                        if (!wait_until_deadline([&] {
                                return allow_background_finish.load(std::memory_order_acquire);
                            })) {
                            throw std::runtime_error("timed out waiting for background release");
                        }
                        background_finished.store(true, std::memory_order_release);
                        return;
                    }
                    caller_worker_started.store(true, std::memory_order_release);
                    if (!wait_until_deadline(
                            [&] { return background_started.load(std::memory_order_acquire); })) {
                        throw std::runtime_error("timed out waiting for background worker");
                    }
                    throw std::runtime_error("range failed");
                });
        } catch (...) {
            caller_error = std::current_exception();
        }
        caller_returned.store(true, std::memory_order_release);
    });

    const bool workers_started = wait_until_deadline([&] {
        return background_started.load(std::memory_order_acquire) &&
               caller_worker_started.load(std::memory_order_acquire);
    });
    const bool returned_before_release = caller_returned.load(std::memory_order_acquire);
    // Release and join before asserting so a timeout cannot strand a joinable thread.
    allow_background_finish.store(true, std::memory_order_release);
    caller.join();

    REQUIRE(workers_started);
    REQUIRE_FALSE(returned_before_release);
    REQUIRE(background_finished.load(std::memory_order_acquire));
    REQUIRE(caller_error != nullptr);
    REQUIRE_THROWS_WITH(std::rethrow_exception(caller_error), "range failed");
}
