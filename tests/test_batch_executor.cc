#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/fault_sampling.h"
#include "clifft/util/shot_seed.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using clifft::KFaultSampler;
using clifft::make_seed_root;
using clifft::SeedRoot;
using clifft::sampling::BatchExecutionPolicy;
using clifft::sampling::BatchExecutor;
using clifft::sampling::BatchOutputMode;
using clifft::sampling::BatchSamplingMode;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::resolve_batch_execution_policy;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

ExecutablePlan compile_batch_test_plan(
    std::optional<std::span<const uint8_t>> postselection = std::nullopt) {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        X_ERROR(0.125) 0
        H 0
        T 0
        M 0
        H 1
        M(0.25) 1
        EXP_VAL Z1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        H 2
        T 2
        EXP_VAL X2
        M 2
    )"));
    clifft::sampling::SamplingPlanOptions options;
    if (postselection.has_value()) {
        options.postselection_mask = *postselection;
    }
    return ExecutablePlan(clifft::sampling::plan_sampling(hir, options));
}

ExecutablePlan compile_batch_fixture(const char* name) {
    clifft::HirModule hir =
        clifft::trace(clifft::parse_file(std::string(CLIFFT_FIXTURES_DIR) + "/" + name));
    auto pass_manager = clifft::default_hir_pass_manager();
    pass_manager.run(hir);
    return ExecutablePlan(clifft::sampling::plan_sampling(hir));
}

void compare_lane_outputs(const BatchExecutor& actual, const BatchExecutor& replay, uint32_t lane,
                          const ExecutablePlan& plan) {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        CAPTURE(lane, record);
        REQUIRE(actual.measurement(lane, record) == replay.measurement(lane, record));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        CAPTURE(lane, detector);
        REQUIRE(actual.detector(lane, detector) == replay.detector(lane, detector));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        CAPTURE(lane, observable);
        REQUIRE(actual.observable(lane, observable) == replay.observable(lane, observable));
    }
    for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
        CAPTURE(lane, exp_val);
        REQUIRE(actual.exp_val(lane, exp_val) == replay.exp_val(lane, exp_val));
    }
}

}  // namespace

TEST_CASE("Packed executor replays seeded fixed-plan rows") {
    constexpr uint32_t shots = 65;
    const ExecutablePlan plan = compile_batch_test_plan();
    const SeedRoot root = make_seed_root(shots, uint64_t{9183});
    BatchExecutor batch(plan, shots);
    batch.run_batch(root, 0, shots);
    BatchExecutor replay(plan, shots);
    replay.run_batch(root, 0, shots);
    REQUIRE(batch.surviving_shots() == shots);
    REQUIRE(replay.surviving_shots() == shots);

    for (uint32_t shot = 0; shot < shots; ++shot) {
        REQUIRE(batch.shot_index(shot) == shot);
        REQUIRE(replay.shot_index(shot) == shot);
        compare_lane_outputs(batch, replay, shot, plan);
    }
}

TEST_CASE("Packed executor replays compacted survivor sidecars") {
    constexpr uint32_t shots = 129;
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan plan = compile_batch_test_plan(postselection);
    const SeedRoot root = make_seed_root(shots, uint64_t{9184});
    BatchExecutor batch(plan, shots);
    batch.run_batch(root, 0, shots);
    BatchExecutor replay(plan, shots);
    replay.run_batch(root, 0, shots);

    REQUIRE(batch.surviving_shots() == replay.surviving_shots());
    for (uint32_t lane = 0; lane < batch.surviving_shots(); ++lane) {
        REQUIRE(batch.shot_index(lane) == replay.shot_index(lane));
        compare_lane_outputs(batch, replay, lane, plan);
    }
}

TEST_CASE("Packed executor replays fixed-fault rows") {
    constexpr uint32_t shots = 67;
    const ExecutablePlan plan = compile_batch_test_plan();
    const SeedRoot root = make_seed_root(shots, uint64_t{9185});
    const std::vector<double> probabilities = plan.noise_site_probabilities();
    KFaultSampler batch_faults(probabilities, 1);
    BatchExecutor batch(plan, shots, BatchOutputMode::Rows, BatchSamplingMode::FixedFaults);
    batch.run_batch(root, 0, shots, batch_faults);

    KFaultSampler replay_faults(probabilities, 1);
    BatchExecutor replay(plan, shots, BatchOutputMode::Rows, BatchSamplingMode::FixedFaults);
    replay.run_batch(root, 0, shots, replay_faults);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        REQUIRE(batch.shot_index(shot) == shot);
        REQUIRE(replay.shot_index(shot) == shot);
        compare_lane_outputs(batch, replay, shot, plan);
    }
}

TEST_CASE("Packed capacity policy bounds worker state footprint") {
    const ExecutablePlan narrow(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse("H 0 1\nM 0 1\n"))));
    REQUIRE(resolve_batch_execution_policy(narrow, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 2048);
    REQUIRE(resolve_batch_execution_policy(narrow, 1024, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 1024);
    REQUIRE(resolve_batch_execution_policy(narrow, 63, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 1);
    REQUIRE(resolve_batch_execution_policy(narrow, 63, 1, 1, BatchOutputMode::Rows, uint32_t{65})
                .lane_capacity == 63);
    REQUIRE(
        resolve_batch_execution_policy(narrow, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{4096})
            .lane_capacity == 2048);
    REQUIRE(resolve_batch_execution_policy(narrow, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{1})
                .lane_capacity == 1);
    REQUIRE_THROWS_WITH(
        resolve_batch_execution_policy(narrow, 4096, 1, 2, BatchOutputMode::Rows, uint32_t{2}),
        "packed batch_size is incompatible with intra-shot workers");

    std::string circuit;
    for (uint32_t qubit = 0; qubit < 18; ++qubit) {
        circuit.append("H ")
            .append(std::to_string(qubit))
            .append("\nT ")
            .append(std::to_string(qubit))
            .append("\n");
    }
    const ExecutablePlan wide(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit))));
    REQUIRE(wide.peak_active_width() == 18);
    REQUIRE(resolve_batch_execution_policy(wide, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 1);
    REQUIRE(resolve_batch_execution_policy(wide, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{2})
                .lane_capacity == 2);
    REQUIRE_THROWS_WITH(
        resolve_batch_execution_policy(wide, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{2048}),
        "explicit batch_size exceeds the 64 MiB packed-state limit; request a smaller batch_size");

    const ExecutablePlan aligned(clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse(circuit + "H 18\nT 18\nH 18\nM 18\n"))));
    REQUIRE(aligned.peak_active_width() == 19);
    REQUIRE_THROWS_WITH(
        resolve_batch_execution_policy(aligned, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{2}),
        "explicit batch_size exceeds the 64 MiB packed-state limit; request a smaller batch_size");

    const ExecutablePlan interleaved(clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse("H 0 1 2 3 4\nT 0 1 2 3 4\nH 0 1 2 3 4\nM 0 1 2 3 4\n"))));
    REQUIRE(interleaved.peak_active_width() == 5);
    REQUIRE(
        resolve_batch_execution_policy(interleaved, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt)
            .lane_capacity == 1024);

    const ExecutablePlan noisy = compile_batch_test_plan();
    REQUIRE(noisy.num_batch_noise_carriers() == 0);
    REQUIRE(resolve_batch_execution_policy(noisy, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 2048);
    REQUIRE(resolve_batch_execution_policy(noisy, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{65})
                .lane_capacity == 65);
    REQUIRE(resolve_batch_execution_policy(noisy, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt,
                                           BatchSamplingMode::Ordinary,
                                           clifft::sampling::kDefaultBatchWorkerBudget)
                .lane_capacity == 1);
}

TEST_CASE("Packed capacity policy accounts for lane-scaled sidecars") {
    constexpr uint32_t shots = 100'000;
    const ExecutablePlan d7 = compile_batch_fixture("surface_d7_r7_p001.stim");
    const ExecutablePlan d11 = compile_batch_fixture("surface_d11_r11_p001.stim");

    REQUIRE(d7.num_batch_noise_carriers() > 0);
    REQUIRE(d7.num_batch_noise_carriers() < d7.num_symbols());
    REQUIRE(d11.num_batch_noise_carriers() > 0);
    REQUIRE(d11.num_batch_noise_carriers() < d11.num_symbols());
    const BatchExecutionPolicy d7_serial =
        resolve_batch_execution_policy(d7, shots, 1, 1, BatchOutputMode::Rows, std::nullopt);
    const BatchExecutionPolicy d7_threaded =
        resolve_batch_execution_policy(d7, shots, 16, 1, BatchOutputMode::Rows, std::nullopt);
    REQUIRE(d7_serial.lane_capacity == 2048);
    REQUIRE(d7_serial.worker_count == 1);
    REQUIRE(d7_threaded.lane_capacity == d7_serial.lane_capacity);
    REQUIRE(d7_threaded.worker_count == 16);

    REQUIRE(resolve_batch_execution_policy(d11, shots, 1, 1, BatchOutputMode::Rows, std::nullopt)
                .lane_capacity == 2048);
    REQUIRE(resolve_batch_execution_policy(d11, shots, 1, 1, BatchOutputMode::AggregateSurvivors,
                                           std::nullopt)
                .lane_capacity == 2048);
    REQUIRE(resolve_batch_execution_policy(d11, shots, 1, 1, BatchOutputMode::Rows, std::nullopt,
                                           BatchSamplingMode::FixedFaults)
                .lane_capacity == 2048);
}
