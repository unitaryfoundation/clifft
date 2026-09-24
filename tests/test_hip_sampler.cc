#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/hip/sampler.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include "gpu_replay_cases.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingResult;
using clifft::sampling::SamplingSurvivorResult;
using clifft::sampling::hip::CoefficientPrecision;
using clifft::sampling::hip::ExecutionTier;
using clifft::sampling::hip::kThreadPerShotMaxActiveWidth;
using clifft::sampling::hip::Sampler;
using clifft::sampling::hip::SamplingOptions;
using clifft::sampling::hip::tier_name;
using CpuExecutablePlan = clifft::sampling::ExecutablePlan;
using HipExecutablePlan = clifft::sampling::hip::ExecutablePlan;

namespace {

constexpr std::array kExplicitTiers{ExecutionTier::ThreadPerShot, ExecutionTier::BlockShared,
                                    ExecutionTier::BlockGlobal};

void require_hip_device() {
    if (!clifft::sampling::hip::is_available()) {
        SKIP("requires an AMD GPU visible to the HIP runtime");
    }
}

SamplingPlan plan_from(std::string_view circuit_text) {
    return clifft::sampling::plan_sampling(clifft::trace(clifft::parse(circuit_text)));
}

void require_same_rows(const SamplingResult& left, const SamplingResult& right) {
    REQUIRE(left.measurements == right.measurements);
    REQUIRE(left.detectors == right.detectors);
    REQUIRE(left.observables == right.observables);
    REQUIRE(left.exp_vals == right.exp_vals);
}

double standard_error(double probability, double samples) {
    return std::sqrt(probability * (1.0 - probability) / samples);
}

// The H/T prefix keeps one active coordinate per qubit.
std::string parallel_t_observable(uint32_t width) {
    std::string text;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "H " + std::to_string(qubit) + "\n";
    }
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "T " + std::to_string(qubit) + "\n";
    }
    text += "EXP_VAL X0";
    for (uint32_t qubit = 1; qubit < width; ++qubit) {
        text += "*X" + std::to_string(qubit);
    }
    return text + "\n";
}

std::string parallel_t_detector(uint32_t width) {
    const uint32_t ancilla = width + 1;
    std::string text;
    for (uint32_t qubit = 0; qubit <= width; ++qubit) {
        text += "H " + std::to_string(qubit) + "\n";
    }
    for (uint32_t qubit = 0; qubit <= width; ++qubit) {
        text += "T " + std::to_string(qubit) + "\n";
    }
    for (uint32_t qubit = 0; qubit <= width; ++qubit) {
        text += "CX " + std::to_string(qubit) + " " + std::to_string(ancilla) + "\n";
    }
    return text + "M " + std::to_string(ancilla) + "\nDETECTOR rec[-1]\n";
}

// Measuring an unused ancilla gives zero before the readout flip.
std::string parallel_t_readout_noise(uint32_t width) {
    std::string text;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "H " + std::to_string(qubit) + "\n";
    }
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "T " + std::to_string(qubit) + "\n";
    }
    text += "M " + std::to_string(width) + "\n";
    text += "READOUT_NOISE(1) rec[-1]\nDETECTOR rec[-1]\n";
    return text;
}

// Qubit 0 measures one after four T gates. The lowered measurement
// expression includes its own branch symbol, which tests replay read ordering.
std::string parallel_t_replay(uint32_t width) {
    std::string text;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "H " + std::to_string(qubit) + "\n";
    }
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text += "T " + std::to_string(qubit) + "\n";
    }
    return text + "T 0\nT 0\nT 0\nH 0\nM 0\nEXP_VAL Z0\n";
}

// Branch probabilities are 3/8, 3/8, 1/8, 1/8. Qubit 40 stays outside the
// prefix to exercise dormant measurement at every tested width.
std::string biased_active_and_dormant(uint32_t width) {
    std::string prefix_h = "H";
    std::string prefix_t = "T";
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        prefix_h += " " + std::to_string(qubit);
        prefix_t += " " + std::to_string(qubit);
    }
    return prefix_h + "\n" + prefix_t +
           "\nCX 0 1\nH 0\nM 0\nEXP_VAL X1*X2\nH 40\nM 40\nEXP_VAL Z40\n";
}

// The shared-memory limit depends on the device and coefficient precision.
uint32_t first_block_global_width(CoefficientPrecision precision) {
    for (uint32_t width = kThreadPerShotMaxActiveWidth + 1; width <= 24; ++width) {
        const HipExecutablePlan executable(plan_from(parallel_t_observable(width)));
        if (clifft::sampling::hip::selected_tier(executable, precision) ==
            ExecutionTier::BlockGlobal) {
            return width;
        }
    }
    return 0;
}

}  // namespace

TEST_CASE("HIP sampler zero shots does not require a device") {
    const HipExecutablePlan executable(SamplingPlan{});

    REQUIRE(SamplingOptions{}.coefficient_precision == CoefficientPrecision::FP64);
    const SamplingResult rows = clifft::sampling::hip::sample(executable, 0);
    const SamplingSurvivorResult survivors = clifft::sampling::hip::sample_survivors(executable, 0);

    REQUIRE(rows.measurements.empty());
    REQUIRE(rows.detectors.empty());
    REQUIRE(rows.observables.empty());
    REQUIRE(rows.exp_vals.empty());
    REQUIRE(survivors.total_shots == 0);
    REQUIRE(survivors.passed_shots == 0);
    REQUIRE(survivors.observable_ones.empty());

    const SamplingOptions automatic{.block_size = clifft::sampling::hip::kAutoBlockSize};
    REQUIRE(SamplingOptions{}.block_size == clifft::sampling::hip::kAutoBlockSize);
    REQUIRE(clifft::sampling::hip::sample(executable, 0, automatic).measurements.empty());

    const SamplingOptions invalid_high{.block_size = 1025};
    const SamplingOptions invalid_batch{.max_batch_shots = 0};
    REQUIRE_THROWS_AS(clifft::sampling::hip::sample_survivors(executable, 0, false, invalid_high),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::hip::sample(executable, 0, invalid_batch),
                      std::invalid_argument);
}

TEST_CASE("HIP replay validates unsupported inputs before device access") {
    const HipExecutablePlan empty(SamplingPlan{});
    REQUIRE_THROWS_AS(clifft::sampling::hip::replay_shot(empty, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);

    const HipExecutablePlan noisy(plan_from("X_ERROR(0.1) 0\nM 0\n"));
    REQUIRE_THROWS_AS(clifft::sampling::hip::replay_shot(noisy, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);
}

TEST_CASE("HIP replay matches CPU readout noise unreachability") {
    const SamplingPlan plan = plan_from("M 0\nREADOUT_NOISE(0.1) rec[-1]\n");
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const std::array<uint8_t, 1> forced{0};
        clifft::sampling::Executor cpu(cpu_executable);
        const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
        const clifft::sampling::hip::ReplayResult actual =
            clifft::sampling::hip::replay_shot(hip_executable, forced, precision);
        CAPTURE(precision);
        REQUIRE_FALSE(expected.reachable);
        REQUIRE(actual.reachable == expected.reachable);
        REQUIRE_FALSE(actual.survived);
        REQUIRE(actual.outputs.measurements.empty());
        REQUIRE(actual.outputs.detectors.empty());
        REQUIRE(actual.outputs.observables.empty());
        REQUIRE(actual.outputs.exp_vals.empty());
    }
}

TEST_CASE("HIP replay omits incomplete outputs from discarded paths") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection;
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir, plan_options);
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const std::array<uint8_t, 1> forced{1};
        clifft::sampling::Executor cpu(cpu_executable);
        const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
        const clifft::sampling::hip::ReplayResult actual =
            clifft::sampling::hip::replay_shot(hip_executable, forced, precision);
        CAPTURE(precision);
        REQUIRE(expected.reachable);
        REQUIRE(cpu.discarded());
        REQUIRE(actual.reachable == expected.reachable);
        REQUIRE_FALSE(actual.survived);
        REQUIRE(actual.outputs.measurements.empty());
        REQUIRE(actual.outputs.detectors.empty());
        REQUIRE(actual.outputs.observables.empty());
        REQUIRE(actual.outputs.exp_vals.empty());
    }
}

TEST_CASE("HIP sampler is repeatable within each coefficient precision") {
    require_hip_device();
    const HipExecutablePlan executable(plan_from(R"(
        H 0
        T 0
        H 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingOptions options{.seed = uint64_t{1234}, .coefficient_precision = precision};
        const SamplingResult first = clifft::sampling::hip::sample(executable, 4096, options);
        const SamplingResult second = clifft::sampling::hip::sample(executable, 4096, options);
        require_same_rows(first, second);
    }
}

TEST_CASE("HIP retained sampler preserves seeded rows across bounded batches") {
    require_hip_device();
    const HipExecutablePlan executable(plan_from(R"(
        H 0
        T 0
        H 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));

    constexpr uint32_t kShots = 257;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        Sampler sampler(executable, precision, 7);
        REQUIRE(sampler.coefficient_precision() == precision);
        REQUIRE(sampler.max_batch_shots() == 7);
        REQUIRE(sampler.allocated_device_bytes() > 0);
        const size_t allocated_bytes = sampler.allocated_device_bytes();

        const SamplingResult batched = sampler.sample(kShots, uint64_t{1234}, 64);
        const SamplingResult repeated = sampler.sample(kShots, uint64_t{1234}, 64);
        const SamplingResult single_batch =
            clifft::sampling::hip::sample(executable, kShots,
                                          {.seed = uint64_t{1234},
                                           .coefficient_precision = precision,
                                           .block_size = 64,
                                           .max_batch_shots = kShots});

        require_same_rows(batched, repeated);
        require_same_rows(batched, single_batch);
        REQUIRE(sampler.allocated_device_bytes() == allocated_bytes);
    }
}

TEST_CASE("HIP retained sampler preserves survivor order across bounded batches") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        M 0
        DETECTOR rec[-1]
        H 1
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection;
    const HipExecutablePlan executable(clifft::sampling::plan_sampling(hir, plan_options));
    require_hip_device();

    constexpr uint32_t kShots = 263;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        Sampler sampler(executable, precision, 5);
        const SamplingSurvivorResult batched =
            sampler.sample_survivors(kShots, true, uint64_t{91}, 32);
        const SamplingSurvivorResult single_batch =
            clifft::sampling::hip::sample_survivors(executable, kShots, true,
                                                    {.seed = uint64_t{91},
                                                     .coefficient_precision = precision,
                                                     .block_size = 32,
                                                     .max_batch_shots = kShots});

        REQUIRE(batched.total_shots == single_batch.total_shots);
        REQUIRE(batched.passed_shots == single_batch.passed_shots);
        REQUIRE(batched.logical_errors == single_batch.logical_errors);
        REQUIRE(batched.observable_ones == single_batch.observable_ones);
        REQUIRE(batched.measurements == single_batch.measurements);
        REQUIRE(batched.detectors == single_batch.detectors);
        REQUIRE(batched.observables == single_batch.observables);
        REQUIRE(batched.exp_vals == single_batch.exp_vals);
    }
}

TEST_CASE("HIP sampler computes expectation values with FP64 accumulation") {
    const SamplingPlan plan = plan_from(R"(
        R_X(0.13) 0
        R_Y(-0.27) 1
        R_ZZ(0.19) 0 1
        R_PAULI(0.31) X0*Y1
        EXP_VAL X0
        EXP_VAL Y0
        EXP_VAL Z0
        EXP_VAL X1
        EXP_VAL Y1
        EXP_VAL Z1
        EXP_VAL X0*X1
        EXP_VAL X0*Y1
        EXP_VAL Z0*Z1
    )");
    const HipExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 4, uint64_t{17});

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 5e-6}}) {
        const SamplingResult actual = clifft::sampling::hip::sample(
            executable, 4, {.seed = uint64_t{17}, .coefficient_precision = precision});
        REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
        for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
            CAPTURE(precision, index);
            REQUIRE_THAT(actual.exp_vals[index],
                         Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
        }
    }
}

TEST_CASE("HIP replay matches every CPU measurement branch in every tier") {
    const auto test_case = GENERATE(Catch::Generators::from_range(clifft::test::kGpuReplayCases));
    const auto tier = GENERATE(Catch::Generators::from_range(kExplicitTiers));
    CAPTURE(test_case.name, tier_name(tier));
    const SamplingPlan plan = plan_from(test_case.circuit);
    const HipExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    const uint32_t records = test_case.visible + test_case.hidden;
    REQUIRE(executable.num_visible_records() == test_case.visible);
    REQUIRE(executable.num_records() == records);
    REQUIRE(executable.peak_active_width() >= test_case.min_active_width);
    REQUIRE(cpu_executable.num_visible_records() == test_case.visible);
    REQUIRE(cpu_executable.num_hidden_records() == test_case.hidden);
    require_hip_device();

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 2e-5}}) {
        Sampler sampler(executable, precision, 1, tier);
        REQUIRE(sampler.execution_tier() == tier);
        for (uint32_t bits = 0; bits < (uint32_t{1} << records); ++bits) {
            std::vector<uint8_t> forced(records);
            for (uint32_t index = 0; index < records; ++index) {
                forced[index] = (bits >> (records - index - 1)) & 1U;
            }
            clifft::sampling::Executor cpu(cpu_executable);
            const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
            const auto actual = sampler.replay_shot(forced);
            CAPTURE(precision, forced);
            REQUIRE(actual.reachable == expected.reachable);
            if (!expected.reachable) {
                continue;
            }
            REQUIRE(actual.survived);
            REQUIRE_THAT(actual.log_probability,
                         Catch::Matchers::WithinAbs(expected.log_probability, tolerance));
            REQUIRE(actual.outputs.measurements ==
                    std::vector<uint8_t>(forced.begin(), forced.begin() + test_case.visible));
            REQUIRE(actual.outputs.detectors ==
                    std::vector<uint8_t>(cpu.detectors().begin(), cpu.detectors().end()));
            REQUIRE(actual.outputs.observables ==
                    std::vector<uint8_t>(cpu.observables().begin(), cpu.observables().end()));
            REQUIRE(actual.outputs.exp_vals.size() == cpu.exp_vals().size());
            for (size_t index = 0; index < cpu.exp_vals().size(); ++index) {
                REQUIRE_THAT(actual.outputs.exp_vals[index],
                             Catch::Matchers::WithinAbs(cpu.exp_vals()[index], tolerance));
            }
        }
    }
}

TEST_CASE("HIP reset sampling preserves visible rows across batch boundaries") {
    const SamplingPlan plan = plan_from(clifft::test::kResetBatchCircuit);
    const HipExecutablePlan executable(plan);
    REQUIRE(executable.num_visible_records() == 2);
    REQUIRE(executable.num_records() == 3);
    REQUIRE(executable.peak_active_width() >= 1);

    constexpr uint32_t kShots = 257;
    const CpuExecutablePlan cpu_executable(plan);
    clifft::test::require_reset_batch_rows(
        clifft::sampling::sample(cpu_executable, kShots, uint64_t{1234}), kShots);
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        CAPTURE(precision);
        Sampler batched(executable, precision, 7);
        Sampler single_batch(executable, precision, kShots);
        const SamplingResult rows = batched.sample(kShots, uint64_t{1234}, 64);
        require_same_rows(rows, single_batch.sample(kShots, uint64_t{1234}, 64));
        clifft::test::require_reset_batch_rows(rows, kShots);
    }
}

TEST_CASE("HIP sampler applies both asymmetric readout endpoints exactly") {
    const HipExecutablePlan executable(plan_from(R"(
        M 0
        READOUT_NOISE(1, 0) rec[-1]
        X 1
        M 1
        READOUT_NOISE(0, 1) rec[-1]
    )"));
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingResult result = clifft::sampling::hip::sample(
            executable, 64, {.seed = uint64_t{3}, .coefficient_precision = precision});
        REQUIRE(result.measurements.size() == 128);
        for (size_t shot = 0; shot < 64; ++shot) {
            CAPTURE(precision, shot);
            REQUIRE(result.measurements[2 * shot] == 1);
            REQUIRE(result.measurements[2 * shot + 1] == 0);
        }
    }
}

TEST_CASE("HIP sampler evaluates both observable value domains") {
    const SamplingPlan plan = plan_from(R"(
        X_ERROR(1) 1
        X_ERROR(1) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        READOUT_NOISE(1) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
    )");
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    constexpr uint32_t kShots = 16;
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, kShots, uint64_t{11});
    REQUIRE(std::ranges::all_of(expected.measurements, [](uint8_t value) { return value == 0; }));
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        REQUIRE(expected.observables[2 * shot] == 1);
        REQUIRE(expected.observables[2 * shot + 1] == 0);
    }
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingResult actual = clifft::sampling::hip::sample(
            hip_executable, kShots, {.seed = uint64_t{11}, .coefficient_precision = precision});
        CAPTURE(precision);
        require_same_rows(actual, expected);
    }
}

TEST_CASE("HIP sampler matches the full categorical Pauli channel distribution") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0
        CX 0 1
        H 0
        M 0 1
    )");
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    REQUIRE(cpu_executable.num_presampled_symbols() == 3);

    std::array<double, 4> expected{};
    for (uint32_t outcome = 0; outcome < 4; ++outcome) {
        std::array<uint8_t, 3> presampled{};
        if (outcome != 0) {
            presampled[outcome - 1] = 1;
        }
        clifft::sampling::Executor cpu(cpu_executable);
        cpu.run_shot(presampled);
        const uint32_t key = cpu.visible_records()[0] | (cpu.visible_records()[1] << 1U);
        expected[key] += std::array{0.4, 0.1, 0.2, 0.3}[outcome];
    }
    REQUIRE(std::count_if(expected.begin(), expected.end(),
                          [](double probability) { return probability > 0.0; }) == 4);
    require_hip_device();

    constexpr uint32_t kShots = 50000;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingResult result = clifft::sampling::hip::sample(
            hip_executable, kShots, {.seed = uint64_t{29}, .coefficient_precision = precision});
        std::array<uint32_t, 4> counts{};
        for (uint32_t shot = 0; shot < kShots; ++shot) {
            const uint32_t key =
                result.measurements[2 * shot] | (result.measurements[2 * shot + 1] << 1U);
            ++counts[key];
        }
        for (uint32_t key = 0; key < counts.size(); ++key) {
            const double actual = static_cast<double>(counts[key]) / kShots;
            const double tolerance =
                6.0 * standard_error(expected[key], static_cast<double>(kShots)) + 1e-3;
            CAPTURE(precision, key, expected[key], actual);
            REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected[key], tolerance));
        }
    }
}

TEST_CASE("HIP survivor compaction retains complete rows") {
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        M 0
        DETECTOR rec[-1]
        H 1
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection;
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir, plan_options);
    const HipExecutablePlan executable(plan);
    require_hip_device();

    constexpr uint32_t kShots = 8192;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingSurvivorResult result = clifft::sampling::hip::sample_survivors(
            executable, kShots, true, {.seed = uint64_t{43}, .coefficient_precision = precision});
        REQUIRE(result.passed_shots > 0);
        REQUIRE(result.passed_shots < kShots);
        REQUIRE(result.measurements.size() == static_cast<size_t>(result.passed_shots) * 2);
        REQUIRE(result.detectors.size() == result.passed_shots);
        REQUIRE(result.observables.size() == result.passed_shots);
        uint64_t observable_ones = 0;
        for (uint32_t shot = 0; shot < result.passed_shots; ++shot) {
            CAPTURE(precision, shot);
            REQUIRE(result.detectors[shot] == 0);
            REQUIRE(result.measurements[2 * shot] == 0);
            REQUIRE(result.observables[shot] == result.measurements[2 * shot + 1]);
            observable_ones += result.observables[shot];
        }
        REQUIRE(result.observable_ones[0] == observable_ones);
        REQUIRE(result.logical_errors == observable_ones);
    }
}

TEST_CASE("HIP sampler matches CPU survivor statistics with noise") {
    require_hip_device();
    const clifft::HirModule hir = clifft::trace(clifft::parse(R"(
        H 0
        T 0
        EXP_VAL X0
        X_ERROR(0.1) 1
        M(0.05) 1
        DETECTOR rec[-1]
        H 2
        M 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));
    const std::array<uint8_t, 1> postselection{1};
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection;
    const SamplingPlan plan = clifft::sampling::plan_sampling(hir, plan_options);
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    constexpr uint32_t kShots = 40000;

    const SamplingSurvivorResult gpu = clifft::sampling::hip::sample_survivors(
        hip_executable, kShots, true, {.seed = uint64_t{91}});
    const SamplingSurvivorResult cpu =
        clifft::sampling::sample_survivors(cpu_executable, kShots, uint64_t{91}, true);

    const double cpu_survival = static_cast<double>(cpu.passed_shots) / kShots;
    const double gpu_survival = static_cast<double>(gpu.passed_shots) / kShots;
    const double survival_tolerance =
        6.0 * standard_error(cpu_survival, static_cast<double>(kShots)) + 1e-3;
    REQUIRE_THAT(gpu_survival, Catch::Matchers::WithinAbs(cpu_survival, survival_tolerance));
    REQUIRE(gpu.passed_shots > 0);
    REQUIRE(cpu.passed_shots > 0);

    const double cpu_observable = static_cast<double>(cpu.observable_ones[0]) / cpu.passed_shots;
    const double gpu_observable = static_cast<double>(gpu.observable_ones[0]) / gpu.passed_shots;
    const double observable_tolerance =
        6.0 * standard_error(cpu_observable, static_cast<double>(cpu.passed_shots)) + 1e-3;
    REQUIRE_THAT(gpu_observable, Catch::Matchers::WithinAbs(cpu_observable, observable_tolerance));
    REQUIRE(gpu.exp_vals.size() == gpu.passed_shots);
    REQUIRE(cpu.exp_vals.size() == cpu.passed_shots);
    for (double value : gpu.exp_vals) {
        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(cpu.exp_vals[0], 1e-12));
    }
}

TEST_CASE("HIP replay agrees with the CPU on impossible branches at both precisions") {
    // FP32 rounding leaves a small probability on this impossible branch.
    // The precision-specific threshold must still reject it.
    const SamplingPlan plan = plan_from(R"(
        H 0
        H 1
        H 2
        H 3
        T 0
        T 1
        T 2
        T 3
        EXP_VAL X0
        T 0
        T 0
        T 0
        H 0
        M 0
    )");
    const HipExecutablePlan hip_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    REQUIRE(hip_executable.num_visible_records() == 1);
    require_hip_device();

    for (const uint8_t forced_value : {uint8_t{0}, uint8_t{1}}) {
        const std::array<uint8_t, 1> forced{forced_value};
        clifft::sampling::Executor cpu(cpu_executable);
        const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            CAPTURE(forced_value, precision);
            const clifft::sampling::hip::ReplayResult actual =
                clifft::sampling::hip::replay_shot(hip_executable, forced, precision);
            REQUIRE(actual.reachable == expected.reachable);
        }
    }
}
TEST_CASE("HIP sampler matches CPU expectation values on the cooperative LDS tier") {
    constexpr uint32_t kWidth = 8;
    const SamplingPlan plan = plan_from(parallel_t_observable(kWidth));
    REQUIRE(plan.peak_active_width == kWidth);
    const HipExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();
    REQUIRE(clifft::sampling::hip::selected_tier(executable) == ExecutionTier::BlockShared);

    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 4, uint64_t{29});
    REQUIRE(expected.exp_vals.size() == 4);

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 1e-5}}) {
        const SamplingResult actual = clifft::sampling::hip::sample(
            executable, 4, {.seed = uint64_t{29}, .coefficient_precision = precision});
        REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
        for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
            CAPTURE(precision, index);
            REQUIRE_THAT(actual.exp_vals[index],
                         Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
        }
    }
}

TEST_CASE("HIP sampler matches CPU expectation values when the state exceeds workgroup memory") {
    constexpr uint32_t kWidth = 14;
    const SamplingPlan plan = plan_from(parallel_t_observable(kWidth));
    REQUIRE(plan.peak_active_width == kWidth);
    const HipExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();
    REQUIRE(clifft::sampling::hip::selected_tier(executable) == ExecutionTier::BlockGlobal);

    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 2, uint64_t{31});

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 1e-4}}) {
        const SamplingResult actual = clifft::sampling::hip::sample(
            executable, 2, {.seed = uint64_t{31}, .coefficient_precision = precision});
        REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
        for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
            CAPTURE(precision, index);
            REQUIRE_THAT(actual.exp_vals[index],
                         Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
        }
    }
}

TEST_CASE("HIP sampler matches CPU detector statistics at cooperative widths") {
    constexpr uint32_t kWidth = 8;
    const SamplingPlan plan = plan_from(parallel_t_detector(kWidth));
    REQUIRE(plan.peak_active_width > kThreadPerShotMaxActiveWidth);
    const HipExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_hip_device();
    constexpr uint32_t kShots = 20000;

    const SamplingResult cpu = clifft::sampling::sample(cpu_executable, kShots, uint64_t{47});
    const SamplingResult gpu =
        clifft::sampling::hip::sample(executable, kShots, {.seed = uint64_t{47}});
    REQUIRE(cpu.detectors.size() == gpu.detectors.size());

    const auto rate = [](const std::vector<uint8_t>& values) {
        return static_cast<double>(std::count(values.begin(), values.end(), uint8_t{1})) /
               static_cast<double>(values.size());
    };
    const double cpu_rate = rate(cpu.detectors);
    const double gpu_rate = rate(gpu.detectors);
    const double tolerance = 6.0 * standard_error(cpu_rate, static_cast<double>(kShots)) + 1e-3;
    REQUIRE_THAT(gpu_rate, Catch::Matchers::WithinAbs(cpu_rate, tolerance));
}

TEST_CASE("HIP sampler is repeatable at cooperative widths") {
    const SamplingPlan plan = plan_from(parallel_t_detector(8));
    REQUIRE(plan.peak_active_width > kThreadPerShotMaxActiveWidth);
    const HipExecutablePlan executable(plan);
    require_hip_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const SamplingOptions options{.seed = uint64_t{53}, .coefficient_precision = precision};
        const SamplingResult first = clifft::sampling::hip::sample(executable, 2048, options);
        const SamplingResult second = clifft::sampling::hip::sample(executable, 2048, options);
        require_same_rows(first, second);
    }
}

TEST_CASE("HIP cooperative readout noise flips each record exactly once") {
    require_hip_device();
    constexpr uint32_t kShots = 64;

    for (const uint32_t width : {uint32_t{8}, uint32_t{14}}) {
        const SamplingPlan plan = plan_from(parallel_t_readout_noise(width));
        REQUIRE(plan.peak_active_width == width);
        const HipExecutablePlan executable(plan);

        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            const ExecutionTier tier = clifft::sampling::hip::selected_tier(executable, precision);
            CAPTURE(width, precision, tier_name(tier));
            REQUIRE(tier != ExecutionTier::ThreadPerShot);

            const SamplingResult rows = clifft::sampling::hip::sample(
                executable, kShots, {.seed = uint64_t{5}, .coefficient_precision = precision});
            REQUIRE(rows.measurements.size() == kShots);
            REQUIRE(rows.detectors.size() == kShots);
            REQUIRE(std::count(rows.measurements.begin(), rows.measurements.end(), uint8_t{1}) ==
                    static_cast<long>(kShots));
            REQUIRE(std::count(rows.detectors.begin(), rows.detectors.end(), uint8_t{1}) ==
                    static_cast<long>(kShots));
        }
    }
}

TEST_CASE("HIP cooperative replay reads the branch symbol before publishing it") {
    require_hip_device();

    for (const uint32_t width : {uint32_t{8}, uint32_t{14}}) {
        const SamplingPlan plan = plan_from(parallel_t_replay(width));
        REQUIRE(plan.peak_active_width == width);
        const HipExecutablePlan hip_executable(plan);
        const CpuExecutablePlan cpu_executable(plan);
        REQUIRE(hip_executable.num_visible_records() == 1);

        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            const ExecutionTier tier =
                clifft::sampling::hip::selected_tier(hip_executable, precision);
            const uint32_t boundary = first_block_global_width(precision);
            CAPTURE(width, precision, tier_name(tier), boundary);
            REQUIRE(tier ==
                    (width < boundary ? ExecutionTier::BlockShared : ExecutionTier::BlockGlobal));

            for (const uint8_t forced_value : {uint8_t{0}, uint8_t{1}}) {
                const std::array<uint8_t, 1> forced{forced_value};
                clifft::sampling::Executor cpu(cpu_executable);
                const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
                const clifft::sampling::hip::ReplayResult actual =
                    clifft::sampling::hip::replay_shot(hip_executable, forced, precision);
                CAPTURE(forced_value);
                REQUIRE(actual.reachable == expected.reachable);
                if (!expected.reachable) {
                    continue;
                }
                REQUIRE_THAT(actual.log_probability,
                             Catch::Matchers::WithinAbs(expected.log_probability, 1e-12));
                REQUIRE(actual.outputs.measurements ==
                        std::vector<uint8_t>(forced.begin(), forced.end()));
                REQUIRE(actual.outputs.exp_vals.size() == cpu.exp_vals().size());
                REQUIRE(actual.outputs.exp_vals.size() == 1);
                for (size_t index = 0; index < actual.outputs.exp_vals.size(); ++index) {
                    CAPTURE(index);
                    REQUIRE_THAT(actual.outputs.exp_vals[index],
                                 Catch::Matchers::WithinAbs(cpu.exp_vals()[index], 1e-9));
                }
            }
        }
    }
}

TEST_CASE("HIP block tiers honour an explicit block size and reject unusable ones") {
    const SamplingPlan plan = plan_from(parallel_t_observable(8));
    REQUIRE(plan.peak_active_width == 8);
    const HipExecutablePlan executable(plan);
    require_hip_device();
    REQUIRE(clifft::sampling::hip::selected_tier(executable) != ExecutionTier::ThreadPerShot);

    Sampler sampler(executable, CoefficientPrecision::FP64, 256);
    const SamplingResult automatic =
        sampler.sample(64, uint64_t{11}, clifft::sampling::hip::kAutoBlockSize);
    for (const uint32_t lanes : {uint32_t{64}, uint32_t{128}, uint32_t{256}}) {
        CAPTURE(lanes);
        const SamplingResult forced = sampler.sample(64, uint64_t{11}, lanes);
        REQUIRE(forced.exp_vals.size() == automatic.exp_vals.size());
        for (size_t index = 0; index < forced.exp_vals.size(); ++index) {
            REQUIRE_THAT(forced.exp_vals[index],
                         Catch::Matchers::WithinAbs(automatic.exp_vals[index], 1e-12));
        }
    }
    // Not a power of two, below a wavefront, and past the reduction capacity.
    for (const uint32_t lanes : {uint32_t{96}, uint32_t{32}, uint32_t{512}}) {
        CAPTURE(lanes);
        REQUIRE_THROWS_AS(sampler.sample(8, uint64_t{11}, lanes), std::invalid_argument);
    }
}

TEST_CASE("HIP tier selection changes at the shared memory boundary") {
    require_hip_device();
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const uint32_t boundary = first_block_global_width(precision);
        CAPTURE(precision, boundary);
        REQUIRE(boundary > kThreadPerShotMaxActiveWidth + 1);

        const HipExecutablePlan below(plan_from(parallel_t_observable(boundary - 1)));
        const HipExecutablePlan at(plan_from(parallel_t_observable(boundary)));
        REQUIRE(clifft::sampling::hip::selected_tier(below, precision) ==
                ExecutionTier::BlockShared);
        REQUIRE(clifft::sampling::hip::selected_tier(at, precision) == ExecutionTier::BlockGlobal);

        // The thread-per-shot width cutoff only applies to automatic selection.
        REQUIRE(Sampler(at, precision, 1, ExecutionTier::ThreadPerShot).execution_tier() ==
                ExecutionTier::ThreadPerShot);
        REQUIRE(Sampler(below, precision, 1, ExecutionTier::BlockGlobal).execution_tier() ==
                ExecutionTier::BlockGlobal);
        REQUIRE_THROWS_AS(Sampler(at, precision, 1, ExecutionTier::BlockShared),
                          std::invalid_argument);
    }
}

TEST_CASE("HIP replay matches the CPU on biased branches at both block tiers") {
    require_hip_device();
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const uint32_t boundary = first_block_global_width(precision);
        REQUIRE(boundary > kThreadPerShotMaxActiveWidth + 1);
        for (const auto& [prefix, expected_tier] :
             {std::pair{boundary - 1, ExecutionTier::BlockShared},
              std::pair{boundary, ExecutionTier::BlockGlobal}}) {
            const SamplingPlan plan = plan_from(biased_active_and_dormant(prefix));
            const HipExecutablePlan hip_executable(plan);
            const CpuExecutablePlan cpu_executable(plan);
            REQUIRE(hip_executable.num_visible_records() == 2);
            REQUIRE(hip_executable.num_records() == 2);
            CAPTURE(precision, prefix, tier_name(expected_tier), plan.peak_active_width);
            REQUIRE(plan.peak_active_width == prefix);
            REQUIRE(clifft::sampling::hip::selected_tier(hip_executable, precision) ==
                    expected_tier);

            Sampler sampler(hip_executable, precision, 1);
            for (uint32_t bits = 0; bits < 4U; ++bits) {
                const std::array<uint8_t, 2> forced{static_cast<uint8_t>((bits >> 1) & 1U),
                                                    static_cast<uint8_t>(bits & 1U)};
                clifft::sampling::Executor cpu(cpu_executable);
                const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
                const clifft::sampling::hip::ReplayResult actual = sampler.replay_shot(forced);
                CAPTURE(forced[0], forced[1]);
                REQUIRE(actual.reachable == expected.reachable);
                if (!expected.reachable) {
                    continue;
                }
                const double tolerance = precision == CoefficientPrecision::FP64 ? 1e-12 : 2e-5;
                REQUIRE_THAT(actual.log_probability,
                             Catch::Matchers::WithinAbs(expected.log_probability, tolerance));
                REQUIRE(actual.outputs.measurements ==
                        std::vector<uint8_t>(forced.begin(), forced.end()));
                REQUIRE(actual.outputs.exp_vals.size() == cpu.exp_vals().size());
                REQUIRE(actual.outputs.exp_vals.size() == 2);
                for (size_t index = 0; index < actual.outputs.exp_vals.size(); ++index) {
                    CAPTURE(index);
                    REQUIRE_THAT(actual.outputs.exp_vals[index],
                                 Catch::Matchers::WithinAbs(cpu.exp_vals()[index], tolerance));
                }
            }
        }
    }
}

TEST_CASE("HIP sampling and batching agree with the CPU at block tiers") {
    require_hip_device();
    constexpr uint32_t kShots = 20000;
    const CoefficientPrecision precision = CoefficientPrecision::FP64;
    const uint32_t boundary = first_block_global_width(precision);
    REQUIRE(boundary > kThreadPerShotMaxActiveWidth + 1);

    for (const auto& [prefix, expected_tier] : {std::pair{boundary - 1, ExecutionTier::BlockShared},
                                                std::pair{boundary, ExecutionTier::BlockGlobal}}) {
        const SamplingPlan plan = plan_from(biased_active_and_dormant(prefix));
        const HipExecutablePlan executable(plan);
        const CpuExecutablePlan cpu_executable(plan);
        CAPTURE(prefix, tier_name(expected_tier), plan.peak_active_width);
        REQUIRE(plan.peak_active_width == prefix);
        REQUIRE(clifft::sampling::hip::selected_tier(executable, precision) == expected_tier);

        // Changing batch size must preserve each shot's random stream.
        Sampler whole(executable, precision, kShots);
        Sampler split(executable, precision, 3000);
        REQUIRE(split.max_batch_shots() <= 3000);
        const SamplingResult unsplit = whole.sample(kShots, uint64_t{77});
        const SamplingResult batched = split.sample(kShots, uint64_t{77});
        require_same_rows(unsplit, batched);

        const auto rate = [&](const std::vector<uint8_t>& values, size_t stride, size_t offset) {
            uint32_t ones = 0;
            for (size_t index = offset; index < values.size(); index += stride) {
                ones += values[index];
            }
            return static_cast<double>(ones) / static_cast<double>(values.size() / stride);
        };
        const SamplingResult cpu = clifft::sampling::sample(cpu_executable, kShots, uint64_t{77});
        const double cpu_first = rate(cpu.measurements, 2, 0);
        const double gpu_first = rate(unsplit.measurements, 2, 0);
        REQUIRE_THAT(cpu_first, Catch::Matchers::WithinAbs(0.25, 0.02));
        const double tolerance = 6.0 * standard_error(cpu_first, kShots) + 1e-3;
        REQUIRE_THAT(gpu_first, Catch::Matchers::WithinAbs(cpu_first, tolerance));
    }
}

TEST_CASE("HIP block tiers preserve postselected rows across batches") {
    const auto tier = GENERATE(ExecutionTier::BlockShared, ExecutionTier::BlockGlobal);
    const auto precision = GENERATE(CoefficientPrecision::FP64, CoefficientPrecision::FP32);
    CAPTURE(tier_name(tier), precision);

    // Reject before the second measurement so discarded shots leave incomplete rows.
    // The final readout flip distinguishes the reported record from the measured state.
    const clifft::HirModule hir = clifft::trace(clifft::parse(parallel_t_observable(8) + R"(
        CX 0 1
        H 0
        M 0
        DETECTOR rec[-1]
        EXP_VAL X1*X2
        H 40
        M 40
        READOUT_NOISE(1) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z40
    )"));
    const std::array<uint8_t, 1> postselection{1};
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection;
    const HipExecutablePlan executable(clifft::sampling::plan_sampling(hir, plan_options));
    const HipExecutablePlan unselected(clifft::sampling::plan_sampling(hir));
    REQUIRE(executable.peak_active_width() == 8);
    REQUIRE(executable.has_postselection());
    REQUIRE(executable.num_visible_records() == 2);
    REQUIRE(executable.num_detectors() == 1);
    REQUIRE(executable.num_observables() == 1);
    REQUIRE(executable.num_exp_vals() == 3);
    require_hip_device();

    constexpr uint32_t kShots = 4099;
    constexpr uint64_t kSeed = 77;
    const SamplingResult rows = clifft::sampling::hip::sample(
        unselected, kShots, {.seed = kSeed, .coefficient_precision = precision, .tier = tier});
    REQUIRE(rows.measurements.size() == 2 * kShots);
    REQUIRE(rows.detectors.size() == kShots);
    REQUIRE(rows.observables.size() == kShots);
    REQUIRE(rows.exp_vals.size() == 3 * kShots);

    // Filter full GPU rows to check survival and row order without requiring
    // the CPU and GPU random streams to match.
    SamplingSurvivorResult expected;
    expected.total_shots = kShots;
    expected.observable_ones.resize(1, 0);
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        CAPTURE(shot);
        REQUIRE(rows.detectors[shot] == rows.measurements[2 * shot]);
        REQUIRE(rows.observables[shot] == rows.measurements[2 * shot + 1]);
        REQUIRE_THAT(rows.exp_vals[3 * shot + 2],
                     Catch::Matchers::WithinAbs(2.0 * rows.measurements[2 * shot + 1] - 1.0, 2e-5));
        if (rows.detectors[shot] != 0) {
            continue;
        }
        ++expected.passed_shots;
        expected.logical_errors += rows.observables[shot];
        expected.observable_ones[0] += rows.observables[shot];
        expected.measurements.insert(expected.measurements.end(),
                                     rows.measurements.begin() + 2 * shot,
                                     rows.measurements.begin() + 2 * (shot + 1));
        expected.detectors.push_back(rows.detectors[shot]);
        expected.observables.push_back(rows.observables[shot]);
        expected.exp_vals.insert(expected.exp_vals.end(), rows.exp_vals.begin() + 3 * shot,
                                 rows.exp_vals.begin() + 3 * (shot + 1));
    }
    REQUIRE(expected.passed_shots > 0);
    REQUIRE(expected.passed_shots < kShots);
    REQUIRE(expected.logical_errors > 0);
    REQUIRE(expected.logical_errors < expected.passed_shots);

    for (const uint32_t batch_size : {kShots, uint32_t{127}}) {
        Sampler sampler(executable, precision, batch_size, tier);
        REQUIRE(sampler.execution_tier() == tier);
        REQUIRE(sampler.max_batch_shots() <= batch_size);
        const size_t allocated_bytes = sampler.allocated_device_bytes();
        for (const bool keep_records : {false, true}) {
            CAPTURE(batch_size, keep_records);
            const SamplingSurvivorResult actual =
                sampler.sample_survivors(kShots, keep_records, kSeed);
            REQUIRE(actual.total_shots == expected.total_shots);
            REQUIRE(actual.passed_shots == expected.passed_shots);
            REQUIRE(actual.logical_errors == expected.logical_errors);
            REQUIRE(actual.observable_ones == expected.observable_ones);
            if (keep_records) {
                REQUIRE(actual.measurements == expected.measurements);
                REQUIRE(actual.detectors == expected.detectors);
                REQUIRE(actual.observables == expected.observables);
                REQUIRE(actual.exp_vals == expected.exp_vals);
            } else {
                REQUIRE(actual.measurements.empty());
                REQUIRE(actual.detectors.empty());
                REQUIRE(actual.observables.empty());
                REQUIRE(actual.exp_vals.empty());
            }
            REQUIRE(sampler.allocated_device_bytes() == allocated_bytes);
        }
    }
}
