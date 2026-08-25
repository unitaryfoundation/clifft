#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/batch_executor.h"
#include "clifft/sampling/executor.h"
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

using clifft::derive_state;
using clifft::KFaultSampler;
using clifft::make_seed_root;
using clifft::SeedRoot;
using clifft::sampling::BatchExecutor;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::resolve_batch_capacity;

namespace {

void seed_executor(Executor& executor, const SeedRoot& root, uint32_t shot) {
    const std::array<uint64_t, 4> words = derive_state(root, shot, clifft::kSamplingExecutorDomain);
    executor.reseed_full(words[0], words[1], words[2], words[3]);
}

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

void compare_lane_outputs(const BatchExecutor& batch, uint32_t lane, const Executor& scalar,
                          const ExecutablePlan& plan) {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        CAPTURE(lane, record);
        REQUIRE(batch.measurement(lane, record) == (scalar.visible_records()[record] != 0));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        CAPTURE(lane, detector);
        REQUIRE(batch.detector(lane, detector) == (scalar.detectors()[detector] != 0));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        CAPTURE(lane, observable);
        REQUIRE(batch.observable(lane, observable) == (scalar.observables()[observable] != 0));
    }
    for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
        CAPTURE(lane, exp_val);
        REQUIRE(batch.exp_val(lane, exp_val) == scalar.exp_vals()[exp_val]);
    }
}

}  // namespace

TEST_CASE("Packed executor preserves seeded fixed-plan rows") {
    constexpr uint32_t shots = 65;
    const ExecutablePlan plan = compile_batch_test_plan();
    const SeedRoot root = make_seed_root(shots, uint64_t{9183});
    BatchExecutor batch(plan, shots);
    batch.run_batch(root, 0, shots);
    REQUIRE(batch.surviving_shots() == shots);

    Executor scalar(plan);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        seed_executor(scalar, root, shot);
        scalar.run_shot();
        REQUIRE(batch.shot_index(shot) == shot);
        compare_lane_outputs(batch, shot, scalar, plan);
    }
}

TEST_CASE("Packed executor compacts seeded survivor sidecars") {
    constexpr uint32_t shots = 129;
    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan plan = compile_batch_test_plan(postselection);
    const SeedRoot root = make_seed_root(shots, uint64_t{9184});
    BatchExecutor batch(plan, shots);
    batch.run_batch(root, 0, shots);

    Executor scalar(plan);
    uint32_t lane = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        seed_executor(scalar, root, shot);
        scalar.run_shot();
        if (scalar.discarded()) {
            continue;
        }
        REQUIRE(lane < batch.surviving_shots());
        REQUIRE(batch.shot_index(lane) == shot);
        compare_lane_outputs(batch, lane, scalar, plan);
        ++lane;
    }
    REQUIRE(lane == batch.surviving_shots());
}

TEST_CASE("Packed executor preserves fixed-fault rows") {
    constexpr uint32_t shots = 67;
    const ExecutablePlan plan = compile_batch_test_plan();
    const SeedRoot root = make_seed_root(shots, uint64_t{9185});
    const std::vector<double> probabilities = plan.noise_site_probabilities();
    KFaultSampler batch_faults(probabilities, 1);
    BatchExecutor batch(plan, shots);
    batch.run_batch(root, 0, shots, batch_faults);

    KFaultSampler scalar_faults(probabilities, 1);
    Executor scalar(plan);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        seed_executor(scalar, root, shot);
        scalar.run_shot(scalar_faults);
        REQUIRE(batch.shot_index(shot) == shot);
        compare_lane_outputs(batch, shot, scalar, plan);
    }
}

TEST_CASE("Packed capacity policy bounds worker state footprint") {
    const ExecutablePlan narrow(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse("H 0 1\nM 0 1\n"))));
    REQUIRE(resolve_batch_capacity(narrow, 4096, 1, 1, std::nullopt) == 512);
    REQUIRE(resolve_batch_capacity(narrow, 1024, 4, 1, std::nullopt) == 256);
    REQUIRE(resolve_batch_capacity(narrow, 63, 1, 1, std::nullopt) == 1);
    REQUIRE(resolve_batch_capacity(narrow, 63, 1, 1, uint32_t{65}) == 63);
    REQUIRE(resolve_batch_capacity(narrow, 4096, 1, 1, uint32_t{4096}) == 2048);
    REQUIRE(resolve_batch_capacity(narrow, 4096, 1, 1, uint32_t{1}) == 1);
    REQUIRE_THROWS_WITH(resolve_batch_capacity(narrow, 4096, 1, 2, uint32_t{2}),
                        "packed batch_size is incompatible with intra-shot workers");

    std::string circuit;
    for (uint32_t qubit = 0; qubit < 12; ++qubit) {
        circuit.append("H ")
            .append(std::to_string(qubit))
            .append("\nT ")
            .append(std::to_string(qubit))
            .append("\n");
    }
    const ExecutablePlan wide(
        clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit))));
    REQUIRE(wide.peak_active_width() == 12);
    REQUIRE(resolve_batch_capacity(wide, 4096, 1, 1, std::nullopt) == 1);

    const ExecutablePlan noisy = compile_batch_test_plan();
    REQUIRE(resolve_batch_capacity(noisy, 4096, 1, 1, std::nullopt) == 1);
    REQUIRE(resolve_batch_capacity(noisy, 4096, 1, 1, uint32_t{65}) == 65);
}
