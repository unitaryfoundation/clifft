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
using clifft::sampling::ActiveExpectation;
using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::BatchExecutionPolicy;
using clifft::sampling::BatchExecutor;
using clifft::sampling::BatchOutputMode;
using clifft::sampling::BatchSamplingMode;
using clifft::sampling::DetectorSlot;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::ExpValSlot;
using clifft::sampling::kDefaultMaxWidthFiveBatchLaneWork;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::PlannedAction;
using clifft::sampling::RecordClassical;
using clifft::sampling::RecordParity;
using clifft::sampling::RecordSlot;
using clifft::sampling::resolve_batch_execution_policy;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;
using clifft::sampling::WriteDetector;
using clifft::sampling::WriteExpectationValue;
using clifft::sampling::batch_detail::BatchCompactionPolicyInput;
using clifft::sampling::batch_detail::should_compact_batch_lanes;

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
                          uint32_t replay_lane, const ExecutablePlan& plan) {
    for (uint32_t record = 0; record < plan.num_visible_records(); ++record) {
        CAPTURE(lane, record);
        REQUIRE(actual.measurement(lane, record) == replay.measurement(replay_lane, record));
    }
    for (uint32_t detector = 0; detector < plan.num_detectors(); ++detector) {
        CAPTURE(lane, detector);
        REQUIRE(actual.detector(lane, detector) == replay.detector(replay_lane, detector));
    }
    for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
        CAPTURE(lane, observable);
        REQUIRE(actual.observable(lane, observable) == replay.observable(replay_lane, observable));
    }
    for (uint32_t exp_val = 0; exp_val < plan.num_exp_vals(); ++exp_val) {
        CAPTURE(lane, exp_val);
        REQUIRE(actual.exp_val(lane, exp_val) == replay.exp_val(replay_lane, exp_val));
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
        compare_lane_outputs(batch, replay, shot, shot, plan);
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
        compare_lane_outputs(batch, replay, lane, lane, plan);
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
        compare_lane_outputs(batch, replay, shot, shot, plan);
    }
}

TEST_CASE("Final survivor compaction preserves rows and shot identities across resets") {
    for (const bool non_clifford : {false, true}) {
        std::string circuit = "H 1\n";
        if (non_clifford) {
            circuit += "T 1\n";
        }
        circuit += R"(
            EXP_VAL X1
            CX 1 2
            X_ERROR(0.125) 0 2
            M(0.25) 0
            M 1
            M(0.25) 2
            EXP_VAL Z1
            EXP_VAL Z2
            X 3
            M 3
            OBSERVABLE_INCLUDE(0) rec[-2]
            OBSERVABLE_INCLUDE(2) rec[-3] rec[-2]
            DETECTOR rec[-4]
            DETECTOR
            DETECTOR rec[-1]
        )";
        const clifft::HirModule hir = clifft::trace(clifft::parse(circuit));
        const ExecutablePlan unselected(clifft::sampling::plan_sampling(hir));
        REQUIRE((unselected.peak_active_width() > 0) == non_clifford);
        REQUIRE(unselected.num_expression_registers() > 0);
        REQUIRE(unselected.noise_site_probabilities().size() == 4);
        // All rejection happens after the final random draw, so filtering an
        // unselected run is an exact oracle for compacted rows and shot mapping.
        for (const std::array<uint8_t, 3> mask :
             {std::array<uint8_t, 3>{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}) {
            const ExecutablePlan selected(
                clifft::sampling::plan_sampling(hir, {.postselection_mask = mask,
                                                      .expected_detectors = {},
                                                      .expected_observables = {}}));
            for (const BatchSamplingMode mode :
                 {BatchSamplingMode::Ordinary, BatchSamplingMode::FixedFaults}) {
                for (const uint32_t capacity : {63, 64, 65, 129}) {
                    CAPTURE(non_clifford, mask, mode, capacity);
                    BatchExecutor batch(selected, capacity, BatchOutputMode::Rows, mode);
                    BatchExecutor reference(unselected, capacity, BatchOutputMode::Rows, mode);
                    KFaultSampler batch_faults(selected.noise_site_probabilities(), 1);
                    KFaultSampler reference_faults(unselected.noise_site_probabilities(), 1);
                    const SeedRoot root = make_seed_root(8 * capacity, uint64_t{9186});
                    uint32_t first_shot = 0;
                    for (const uint32_t shots : {capacity, capacity - 1, 0U, 1U, capacity}) {
                        CAPTURE(shots, first_shot);
                        if (mode == BatchSamplingMode::FixedFaults) {
                            batch.run_batch(root, first_shot, shots, batch_faults);
                            reference.run_batch(root, first_shot, shots, reference_faults);
                        } else {
                            batch.run_batch(root, first_shot, shots);
                            reference.run_batch(root, first_shot, shots);
                        }
                        uint32_t destination = 0;
                        for (uint32_t source = 0; source < shots; ++source) {
                            bool rejected = false;
                            for (uint32_t detector = 0; detector < mask.size(); ++detector) {
                                rejected |= mask[detector] && reference.detector(source, detector);
                            }
                            if (!rejected) {
                                REQUIRE(destination < batch.surviving_shots());
                                REQUIRE(batch.shot_index(destination) == first_shot + source);
                                compare_lane_outputs(batch, reference, destination, source,
                                                     selected);
                                ++destination;
                            }
                        }
                        REQUIRE(batch.surviving_shots() == destination);
                        if (mask[0] && shots >= 63) {
                            REQUIRE(destination > 0);
                            REQUIRE(destination < shots);
                        }
                        first_shot += capacity;
                    }
                }
            }
        }
    }
}

TEST_CASE("Intermediate compaction retains pending expressions and forced readout faults") {
    std::string circuit = "X_ERROR(0.8) 0\nM 0\nDETECTOR rec[-1]\nH 1\n";
    for (uint32_t probe = 0; probe < 96; ++probe) {
        circuit += "T 1\nEXP_VAL X1\nH 1\n";
    }
    circuit += "X_ERROR(0.1) 2\nM(0.1) 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    const clifft::HirModule hir = clifft::trace(clifft::parse(circuit));
    const ExecutablePlan unselected(clifft::sampling::plan_sampling(hir));
    const std::array<uint8_t, 1> mask{1};
    const ExecutablePlan selected(clifft::sampling::plan_sampling(
        hir, {.postselection_mask = mask, .expected_detectors = {}, .expected_observables = {}}));
    REQUIRE(selected.peak_active_width() > 0);
    REQUIRE(selected.num_readout_noise_sites() == 1);
    constexpr uint32_t shots = 129;
    BatchExecutor batch(selected, shots, BatchOutputMode::Rows, BatchSamplingMode::FixedFaults);
    BatchExecutor reference(unselected, shots, BatchOutputMode::Rows,
                            BatchSamplingMode::FixedFaults);
    KFaultSampler batch_faults(selected.noise_site_probabilities(), 1);
    KFaultSampler reference_faults(unselected.noise_site_probabilities(), 1);
    const SeedRoot root = make_seed_root(shots, uint64_t{9187});
    batch.run_batch(root, 0, shots, batch_faults);
    reference.run_batch(root, 0, shots, reference_faults);
    REQUIRE(batch.surviving_shots() > 0);
    REQUIRE(batch.surviving_shots() < shots);
    uint32_t destination = 0;
    for (uint32_t source = 0; source < shots; ++source) {
        if (!reference.detector(source, 0)) {
            REQUIRE(destination < batch.surviving_shots());
            REQUIRE(batch.shot_index(destination) == source);
            // With one fault, surviving shots must flip the final readout by
            // either its presampled X error or its forced measurement error.
            REQUIRE(batch.measurement(destination, 1));
            compare_lane_outputs(batch, reference, destination, source, selected);
            ++destination;
        }
    }
    REQUIRE(destination == batch.surviving_shots());
}

TEST_CASE("Packed compaction policy distinguishes compact and defer decisions") {
    BatchCompactionPolicyInput input{
        .remaining_lane_work = {.common = 1},
        .state_size = 2,
        .bit_columns = 1,
        .word_capacity = 2,
        .peak_active_width = 1,
        .active_lanes = 128,
        .live_lanes = 64,
    };

    CHECK_FALSE(should_compact_batch_lanes(input, BatchOutputMode::Rows));
    input.remaining_lane_work.common = 1'000'000;
    CHECK(should_compact_batch_lanes(input, BatchOutputMode::Rows));

    input.live_lanes = input.active_lanes;
    CHECK_FALSE(should_compact_batch_lanes(input, BatchOutputMode::Rows));
}

TEST_CASE("Packed compaction policy excludes aggregate expectation tails") {
    const BatchCompactionPolicyInput input{
        .remaining_lane_work = {.row_output = 24'000},
        .state_size = 2,
        .bit_columns = 1,
        .row_output_entries = 3'000,
        .word_capacity = 2,
        .peak_active_width = 1,
        .active_lanes = 128,
        .live_lanes = 64,
    };

    CHECK(should_compact_batch_lanes(input, BatchOutputMode::Rows));
    CHECK_FALSE(should_compact_batch_lanes(input, BatchOutputMode::AggregateSurvivors));
}

TEST_CASE("Executable postselection metadata tracks a fused rotation tail") {
    SamplingPlan plan;
    plan.num_qubits = 2;
    plan.initial_active_width = 2;
    plan.peak_active_width = 2;
    plan.num_detectors = 1;
    plan.actions = {
        PlannedAction{2, 2, WriteDetector{RecordParity{}, DetectorSlot{0}, true}},
        PlannedAction{2, 2, RotateActivePauli{ActivePauli{0b01, 0b00}, 0.25, AffineBool{}}},
        PlannedAction{2, 2, RotateActivePauli{ActivePauli{0b10, 0b01}, -0.3, AffineBool{true}}},
        PlannedAction{2, 2, RotateActivePauli{ActivePauli{0b11, 0b11}, 0.4, AffineBool{}}},
    };

    const ExecutablePlan executable(plan);
    REQUIRE(executable.num_actions() == 2);
    CHECK(executable.inspect_action(0) ==
          "WRITE_DETECTOR detector=d0 outcome=0 postselect "
          "remaining_batch_lane_work_common=32 remaining_batch_lane_work_row_output=0");
    CHECK(executable.inspect_action(1) == "FUSED_ROTATION descriptor=0");
    CHECK(executable.estimated_batch_lane_work(BatchOutputMode::AggregateSurvivors) == 32);
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

    const std::array<uint8_t, 1> postselection{1};
    const ExecutablePlan postselected = compile_batch_test_plan(postselection);
    REQUIRE(postselected.has_postselection());
    REQUIRE(resolve_batch_execution_policy(postselected, 4096, 1, 1, BatchOutputMode::Rows,
                                           std::nullopt)
                .lane_capacity == 1);
    REQUIRE(resolve_batch_execution_policy(postselected, 4096, 1, 1, BatchOutputMode::Rows,
                                           uint32_t{65})
                .lane_capacity == 65);

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

    SamplingPlan sustained_plan;
    sustained_plan.num_qubits = 5;
    sustained_plan.initial_active_width = 5;
    sustained_plan.peak_active_width = 5;
    sustained_plan.num_exp_vals = 200;
    for (uint32_t probe = 0; probe < sustained_plan.num_exp_vals; ++probe) {
        sustained_plan.actions.push_back(
            PlannedAction{5, 5,
                          WriteExpectationValue{ActiveExpectation{ActivePauli{1, 0}, AffineBool{}},
                                                ExpValSlot{probe}}});
    }
    const ExecutablePlan sustained(sustained_plan);
    REQUIRE(sustained.estimated_width_five_batch_lane_work(BatchOutputMode::Rows) >
            kDefaultMaxWidthFiveBatchLaneWork);
    REQUIRE(sustained.estimated_width_five_batch_lane_work(BatchOutputMode::AggregateSurvivors) ==
            0);
    REQUIRE(
        resolve_batch_execution_policy(sustained, 4096, 1, 1, BatchOutputMode::Rows, std::nullopt)
            .lane_capacity == 1);
    REQUIRE(resolve_batch_execution_policy(sustained, 4096, 1, 1,
                                           BatchOutputMode::AggregateSurvivors, std::nullopt)
                .lane_capacity == 1024);
    REQUIRE(
        resolve_batch_execution_policy(sustained, 4096, 1, 1, BatchOutputMode::Rows, uint32_t{64})
            .lane_capacity == 64);

    SamplingPlan brief_width_five_plan;
    brief_width_five_plan.num_qubits = 5;
    brief_width_five_plan.initial_active_width = 5;
    brief_width_five_plan.peak_active_width = 5;
    brief_width_five_plan.num_visible_records = 2104;
    brief_width_five_plan.symbols.assign(4, SymbolKind::Branch);
    for (uint32_t measurement = 0; measurement < 4; ++measurement) {
        const uint32_t active_before = 5 - measurement;
        const uint32_t pivot = active_before - 1;
        const SymbolId branch{measurement};
        brief_width_five_plan.actions.push_back(
            PlannedAction{active_before, active_before - 1,
                          MeasureActivePauli{ActivePauli{uint64_t{1} << pivot, 0}, pivot, branch,
                                             AffineBool::symbol(branch), RecordSlot{measurement}}});
    }
    for (uint32_t cycle = 0; cycle < 2100; ++cycle) {
        brief_width_five_plan.actions.push_back(
            PlannedAction{1, 1, RotateActivePauli{ActivePauli{1, 0}, 0.25, AffineBool{}}});
        brief_width_five_plan.actions.push_back(
            PlannedAction{1, 1, RecordClassical{AffineBool{}, RecordSlot{cycle + 4}}});
    }
    const ExecutablePlan brief_width_five(brief_width_five_plan);
    REQUIRE(brief_width_five.estimated_batch_lane_work(BatchOutputMode::Rows) >
            kDefaultMaxWidthFiveBatchLaneWork);
    REQUIRE(brief_width_five.estimated_width_five_batch_lane_work(BatchOutputMode::Rows) <
            kDefaultMaxWidthFiveBatchLaneWork);
    REQUIRE(resolve_batch_execution_policy(brief_width_five, 4096, 1, 1, BatchOutputMode::Rows,
                                           std::nullopt)
                .lane_capacity > 1);

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

TEST_CASE("Packed capacity policy limits workers to available batches") {
    const ExecutablePlan plan = compile_batch_test_plan();
    for (const auto output_mode : {BatchOutputMode::Rows, BatchOutputMode::AggregateSurvivors}) {
        for (const uint32_t shots : {65, 66, 130, 131, 257}) {
            CAPTURE(output_mode, shots);
            const auto policy =
                resolve_batch_execution_policy(plan, shots, 2, 1, output_mode, uint32_t{65});
            REQUIRE(policy.lane_capacity == 65);
            REQUIRE(policy.worker_count == (shots == 65 ? 1 : 2));
        }
        for (const uint32_t shots : {257, 4097}) {
            CAPTURE(output_mode, shots);
            const auto policy =
                resolve_batch_execution_policy(plan, shots, 2, 1, output_mode, std::nullopt);
            REQUIRE(policy.lane_capacity == (shots == 257 ? 257 : 2048));
            REQUIRE(policy.worker_count == (shots == 257 ? 1 : 2));
        }
    }
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
