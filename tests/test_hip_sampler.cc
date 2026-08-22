#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/hip/sampler.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

using clifft::sampling::ExecutablePlan;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingResult;
using clifft::sampling::SamplingSurvivorResult;
using clifft::sampling::hip::CoefficientPrecision;
using clifft::sampling::hip::Executable;
using clifft::sampling::hip::SamplingOptions;

namespace {

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

}  // namespace

TEST_CASE("HIP sampler zero shots does not require a device") {
    const Executable executable(SamplingPlan{});

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

    const SamplingOptions invalid_low{.block_size = 0};
    const SamplingOptions invalid_high{.block_size = 1025};
    REQUIRE_THROWS_AS(clifft::sampling::hip::sample(executable, 0, invalid_low),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::hip::sample_survivors(executable, 0, false, invalid_high),
                      std::invalid_argument);
}

TEST_CASE("HIP replay validates unsupported inputs before device access") {
    const Executable empty(SamplingPlan{});
    REQUIRE_THROWS_AS(clifft::sampling::hip::replay_shot(empty, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);

    const Executable noisy(plan_from("X_ERROR(0.1) 0\nM 0\n"));
    REQUIRE_THROWS_AS(clifft::sampling::hip::replay_shot(noisy, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);
}

TEST_CASE("HIP replay matches CPU readout noise unreachability") {
    const SamplingPlan plan = plan_from("M 0\nREADOUT_NOISE(0.1) rec[-1]\n");
    const Executable hip_executable(plan);
    const ExecutablePlan cpu_executable(plan);
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
    const Executable hip_executable(plan);
    const ExecutablePlan cpu_executable(plan);
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
    const Executable executable(plan_from(R"(
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
    const Executable executable(plan);
    const ExecutablePlan cpu_executable(plan);
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

TEST_CASE("HIP replay matches every CPU measurement branch") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        H 1
        T 0
        T 1
        CX 0 1
        MPP Y0*Z1
        R_PAULI(0.17) X0*Y1
        M 0
        CX rec[-1] 2
        EXP_VAL Z2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");
    const Executable hip_executable(plan);
    const ExecutablePlan cpu_executable(plan);
    REQUIRE(hip_executable.num_visible_records() == 2);
    require_hip_device();

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 2e-5}}) {
        for (uint8_t first : {uint8_t{0}, uint8_t{1}}) {
            for (uint8_t second : {uint8_t{0}, uint8_t{1}}) {
                const std::array<uint8_t, 2> forced{first, second};
                clifft::sampling::Executor cpu(cpu_executable);
                const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
                const clifft::sampling::hip::ReplayResult actual =
                    clifft::sampling::hip::replay_shot(hip_executable, forced, precision);
                CAPTURE(precision, first, second);
                REQUIRE(actual.reachable == expected.reachable);
                if (!expected.reachable) {
                    continue;
                }
                REQUIRE_THAT(actual.log_probability,
                             Catch::Matchers::WithinAbs(expected.log_probability, tolerance));
                REQUIRE(actual.outputs.measurements ==
                        std::vector<uint8_t>(forced.begin(), forced.end()));
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
}

TEST_CASE("HIP sampler applies both asymmetric readout endpoints exactly") {
    const Executable executable(plan_from(R"(
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

TEST_CASE("HIP sampler matches the full categorical Pauli channel distribution") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0
        CX 0 1
        H 0
        M 0 1
    )");
    const Executable hip_executable(plan);
    const ExecutablePlan cpu_executable(plan);
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
    const Executable executable(plan);
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
    const Executable hip_executable(plan);
    const ExecutablePlan cpu_executable(plan);
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
