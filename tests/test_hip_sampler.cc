#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/hip/executable.h"
#include "clifft/sampling/hip/sampler.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <string_view>

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
    require_hip_device();
    const SamplingPlan plan = plan_from(R"(
        H 0
        T 0
        EXP_VAL X0
        EXP_VAL Y0
        EXP_VAL Z0
    )");
    const Executable executable(plan);
    const ExecutablePlan cpu_executable(plan);
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 8, uint64_t{17});

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 2e-6}}) {
        const SamplingResult actual = clifft::sampling::hip::sample(
            executable, 8, {.seed = uint64_t{17}, .coefficient_precision = precision});
        REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
        for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
            CAPTURE(precision, index);
            REQUIRE_THAT(actual.exp_vals[index],
                         Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
        }
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
        hip_executable, kShots, false, {.seed = uint64_t{91}});
    const SamplingSurvivorResult cpu =
        clifft::sampling::sample_survivors(cpu_executable, kShots, uint64_t{91});

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
}
