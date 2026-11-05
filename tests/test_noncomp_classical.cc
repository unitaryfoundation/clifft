#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/fixed_classical.h"
#include "clifft/noncomp/sample.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

using namespace clifft;
using namespace clifft::detail;
using namespace clifft::sampling;

TEST_CASE("Packed classical status conditions survive resets and reused partial batches") {
    const ExecutablePlan plan(plan_sampling(trace(
        parse("X_ERROR(0.25) 0\nX_ERROR(0.25) 0\nX_ERROR(0.25) 1\n"
              "X_ERROR(0.25) 1\nX_ERROR(0.25) 0\nM 0 1\nREADOUT_NOISE(0.5) rec[-1] rec[-2]"))));
    REQUIRE(plan.num_presampled_symbols() == 5);
    REQUIRE(plan.num_readout_noise_sites() == 2);
    const auto program = [] {
        // Source descriptors die here. The compiled program owns every binding
        // and shares no mutable status with its source or another worker.
        const std::vector<InputStep> steps{
            NoiseInput{{0}, std::nullopt, 0, {{0, 1.0}}},
            TransitionInput{0, 1.0, false},
            NoiseInput{{0}, std::nullopt, 0, {{1, 1.0}}},
            NoiseInput{{1}, 0, kLeaked, {{2, 1.0}}},
            ProbeInput{0, 0, kLeaked},
            ResetInput{0, {kComputational, kComputational, kLost}},
            NoiseInput{{1}, 0, kLeaked, {{3, 1.0}}},
            NoiseInput{{0}, std::nullopt, 0, {{4, 1.0}}},
            TransitionInput{1, 1.0, true},
            ProbeInput{1, 1, kLost},
        };
        return FixedClassicalProgram(steps, 2, 5, 2, 2);
    }();
    const auto copy = program;
    FixedClassicalWorkspace workspace(2, 129);
    BatchExternalInputs inputs(plan, 129);
    const auto root = make_seed_root(136, uint64_t{11});
    for (uint32_t count : {129, 65, 1, 0}) {
        for (uint32_t input = 0; input < 5; ++input)
            std::ranges::fill(inputs.noise(input), ~uint64_t{0});
        for (uint32_t site = 0; site < 2; ++site) {
            std::ranges::fill(inputs.readout_zero(site), ~uint64_t{0});
            std::ranges::fill(inputs.readout_one(site), ~uint64_t{0});
        }
        NonComputationalSample output;
        output.shots = 136;
        output.num_qubits = 2;
        output.num_measurements = 2;
        output.heralds.resize(272);
        output.final_status.resize(272);
        copy.prepare(root, 7, count, workspace, inputs, output);
        for (size_t word = 0; word < packed_word_count(129); ++word) {
            const uint64_t live =
                word * 64 >= count ? 0
                                   : low_lane_mask(std::min(
                                         uint32_t{64}, count - static_cast<uint32_t>(word * 64)));
            for (uint32_t input = 0; input < 5; ++input)
                CHECK(inputs.noise(input)[word] == ((input == 1 || input == 3) ? 0 : live));
            for (uint32_t site = 0; site < 2; ++site) {
                CHECK(inputs.readout_zero(site)[word] == live);
                CHECK(inputs.readout_one(site)[word] == live);
            }
        }
        for (uint32_t shot = 0; shot < output.shots; ++shot) {
            CHECK(output.final_status[shot * 2] == QubitStatus::Computational);
            CHECK(output.final_status[shot * 2 + 1] ==
                  (shot >= 7 && shot < 7 + count ? QubitStatus::Lost : QubitStatus::Computational));
        }
        CHECK(std::ranges::all_of(output.heralds, [](uint8_t value) { return value == 0; }));
    }
}

TEST_CASE("Packed classical readout endpoints preserve conditions and herald placeholders") {
    const ExecutablePlan plan(
        plan_sampling(trace(parse("REPEAT 5 {\nX_ERROR(0.25) 0\n}\nM(0.5) 0 0 0 0 0 0"))));
    REQUIRE(plan.num_presampled_symbols() == 5);
    REQUIRE(plan.num_readout_noise_sites() == 6);
    const std::array<ReadoutChannel, 3> by_status{ReadoutChannel{0, 0, 0}, ReadoutChannel{1, 0, 0},
                                                  ReadoutChannel{0, 1, 0}};
    const std::vector<InputStep> steps{
        NoiseInput{{0}, std::nullopt, 0, {{0, 1.0}}},
        ReadoutInput{0, 0, 0, {ReadoutChannel{1, 0, 0}, {}, {}}},
        ProbeInput{1, 0, kLeaked},
        // Both intervening operations leave the computational condition intact.
        NoiseInput{{0}, std::nullopt, 0, {{1, 1.0}}},
        ReadoutInput{2, 2, 0, {ReadoutChannel{0, 1, 0}, {}, {}}},
        ReadoutInput{3, 3, 0, {ReadoutChannel{1, 0, 1}, {}, {}}},
        NoiseInput{{0}, std::nullopt, 0, {{2, 0.25}, {3, 0.25}, {4, 1.0}}},
        TransitionInput{0, 1.0, false},
        ReadoutInput{4, 4, 0, by_status},
        TransitionInput{0, 1.0, true},
        ReadoutInput{5, 5, 0, by_status},
    };
    const FixedClassicalProgram program(steps, 1, 5, 6, 6);
    FixedClassicalWorkspace workspace(1, 129);
    BatchExternalInputs inputs(plan, 129);
    const auto root = make_seed_root(136, uint64_t{21});
    for (uint32_t count : {129, 65, 1, 0}) {
        for (uint32_t input = 0; input < 5; ++input)
            std::ranges::fill(inputs.noise(input), ~uint64_t{0});
        for (uint32_t site = 0; site < 6; ++site) {
            std::ranges::fill(inputs.readout_zero(site), ~uint64_t{0});
            std::ranges::fill(inputs.readout_one(site), ~uint64_t{0});
        }
        NonComputationalSample output;
        output.shots = 136;
        output.num_qubits = 1;
        output.num_measurements = 6;
        output.heralds.resize(output.shots * output.num_measurements);
        program.prepare(root, 7, count, workspace, inputs, output);
        for (size_t word = 0; word < packed_word_count(129); ++word) {
            const uint64_t live =
                word * 64 >= count ? 0
                                   : low_lane_mask(std::min(
                                         uint32_t{64}, count - static_cast<uint32_t>(word * 64)));
            CHECK(inputs.noise(0)[word] == live);
            CHECK(inputs.noise(1)[word] == live);
            CHECK(inputs.noise(3)[word] == 0);
            CHECK((inputs.noise(2)[word] | inputs.noise(4)[word]) == live);
            CHECK((inputs.noise(2)[word] & inputs.noise(4)[word]) == 0);
            for (uint32_t site : {0, 4}) {
                CHECK(inputs.readout_zero(site)[word] == live);
                CHECK(inputs.readout_one(site)[word] == 0);
            }
            for (uint32_t site : {2, 5}) {
                CHECK(inputs.readout_zero(site)[word] == 0);
                CHECK(inputs.readout_one(site)[word] == live);
            }
            CHECK(inputs.readout_zero(1)[word] == 0);
            CHECK(inputs.readout_one(1)[word] == 0);
            CHECK(inputs.readout_zero(3)[word] == inputs.readout_one(3)[word]);
            CHECK((inputs.readout_zero(3)[word] & ~live) == 0);
        }
        for (uint32_t shot = 0; shot < output.shots; ++shot) {
            for (uint32_t record = 0; record < output.num_measurements; ++record) {
                CHECK(output.heralds[shot * output.num_measurements + record] ==
                      (shot >= 7 && shot < 7 + count && record == 3));
            }
        }
    }
}
