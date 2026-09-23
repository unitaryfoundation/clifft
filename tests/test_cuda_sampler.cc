#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/cuda/device_program.h"
#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include "gpu_replay_cases.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <exception>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingResult;
using clifft::sampling::SamplingSurvivorResult;
using clifft::sampling::cuda::CoefficientPrecision;
using clifft::sampling::cuda::ExecutionTier;
using clifft::sampling::cuda::Sampler;
using clifft::sampling::cuda::SamplingOptions;
using CpuExecutablePlan = clifft::sampling::ExecutablePlan;
using CudaExecutablePlan = clifft::sampling::cuda::ExecutablePlan;

namespace {

constexpr std::array<ExecutionTier, 3> kExplicitTiers{
    ExecutionTier::ThreadPerShot, ExecutionTier::BlockShared, ExecutionTier::BlockGlobal};

void require_cuda_device() {
    if (!clifft::sampling::cuda::is_available()) {
        SKIP("requires an NVIDIA GPU visible to the CUDA runtime");
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

// Six promoted coordinates give the cooperative tiers 64 coefficients, so a
// default 256-thread block sweeps them with two warps and leaves the rest
// idle, while the state stays small enough for the thread-per-shot tier to
// cross-check.
constexpr std::string_view kWideCircuit = R"(
    H 0
    H 1
    H 2
    H 3
    H 4
    H 5
    T 0
    T 1
    T 2
    T 3
    T 4
    T 5
    CX 0 1
    CX 2 3
    CX 4 5
    CX 1 2
    CX 3 4
    R_PAULI(0.21) X0*Y3
    EXP_VAL X0
    EXP_VAL Z1*Z2
    EXP_VAL X0*Y3
    EXP_VAL Y4*Z5
    M 0 1 2 3 4 5
)";

// Two rotations about Y0 compose to a pi rotation, so qubit 0 ends in |1> and
// the measurement below is numerically deterministic -- but the pair is opaque
// to the planner's symbolic folding, so it survives as a real
// MeasureActivePauli instead of collapsing to a classical record. Six magic
// states widen the active block to seven coordinates, so the cooperative tiers
// sweep 128 coefficients across several warps.
constexpr std::string_view kDeterministicBranchCircuit = R"(
    R_PAULI(0.3) Y0
    R_PAULI(0.7) Y0
    H 1
    H 2
    H 3
    H 4
    H 5
    H 6
    T 1
    T 2
    T 3
    T 4
    T 5
    T 6
    CX 1 2
    CX 3 4
    CX 5 6
    CX 2 3
    CX 4 5
    M 0
)";

// A postselected detector reads record 0, and the very next action has lane 0
// flip that same record. Every shot must survive, because the detector is
// defined on the pre-flip value.
constexpr std::string_view kDetectorThenFlipCircuit = R"(
    H 1
    H 2
    H 3
    H 4
    H 5
    H 6
    T 1
    T 2
    T 3
    T 4
    T 5
    T 6
    CX 1 2
    CX 3 4
    CX 5 6
    M 0
    DETECTOR rec[-1]
    READOUT_NOISE(1) rec[-1]
)";

// Width-k magic-state family used to locate the shared-memory boundary at
// runtime. k Hadamard/T pairs promote k coordinates; the CX layers and the
// rotation mix them; the expectation values before the first measurement check
// the strided kernels exactly; and the measurement of qubit 0 collapses the
// state, so the rotation and expectation values after it observe the evolved,
// post-collapse state. The noisy variant adds a Pauli channel, a measurement
// with a postselected detector, and an observable, so cooperative noise,
// discard, and survivor compaction run at the same width.
std::string wide_circuit_text(uint32_t width, bool noisy) {
    std::ostringstream text;
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text << "H " << qubit << "\n";
    }
    for (uint32_t qubit = 0; qubit < width; ++qubit) {
        text << "T " << qubit << "\n";
    }
    for (uint32_t qubit = 0; qubit + 1 < width; qubit += 2) {
        text << "CX " << qubit << " " << qubit + 1 << "\n";
    }
    for (uint32_t qubit = 1; qubit + 1 < width; qubit += 2) {
        text << "CX " << qubit << " " << qubit + 1 << "\n";
    }
    text << "R_PAULI(0.21) X0*Y1\n"
            "EXP_VAL X0\n"
            "EXP_VAL Z1*Z2\n"
            "M 0\n"
            "R_PAULI(0.13) Z1*X2\n"
            "EXP_VAL X1\n"
            "EXP_VAL Y2\n";
    if (noisy) {
        text << "PAULI_CHANNEL_1(0.05, 0.05, 0.05) 1\n"
                "M 1\n"
                "DETECTOR rec[-1]\n";
    } else {
        text << "M 1\n";
    }
    text << "M";
    for (uint32_t qubit = 2; qubit < width; ++qubit) {
        text << " " << qubit;
    }
    text << "\n";
    if (noisy) {
        text << "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n";
    }
    return text.str();
}

SamplingPlan noisy_postselected_plan(uint32_t width) {
    static constexpr std::array<uint8_t, 1> kPostselectDetector{1};
    clifft::sampling::SamplingPlanOptions options;
    options.postselection_mask = kPostselectDetector;
    return clifft::sampling::plan_sampling(
        clifft::trace(clifft::parse(wide_circuit_text(width, true))), options);
}

// The shared-memory boundary depends on the device's opt-in limit, so locate
// it at runtime: the narrowest width whose per-shot coefficient storage no
// longer fits. Lowering is host-only, so probing several widths is cheap.
std::optional<uint32_t> first_block_global_width(CoefficientPrecision precision) {
    for (uint32_t width = 8; width <= 18; ++width) {
        const CudaExecutablePlan executable(plan_from(wide_circuit_text(width, false)));
        if (clifft::sampling::cuda::selected_tier(executable, precision) ==
            ExecutionTier::BlockGlobal) {
            return width;
        }
    }
    return std::nullopt;
}

size_t coefficient_slab_bytes(uint32_t width, CoefficientPrecision precision) {
    const size_t element = precision == CoefficientPrecision::FP32 ? sizeof(float) : sizeof(double);
    return clifft::sampling::cuda::detail::coefficient_elements_per_shot(width) * element;
}

std::vector<std::vector<uint8_t>> forced_record_paths(uint32_t num_records) {
    std::vector<std::vector<uint8_t>> paths;
    paths.emplace_back(num_records, uint8_t{0});
    paths.emplace_back(num_records, uint8_t{1});
    for (const uint8_t first : {uint8_t{0}, uint8_t{1}}) {
        std::vector<uint8_t> alternating(num_records);
        for (uint32_t index = 0; index < num_records; ++index) {
            alternating[index] = static_cast<uint8_t>((first + index) & 1U);
        }
        paths.push_back(std::move(alternating));
    }
    return paths;
}

double two_sample_tolerance(double probability, double left_samples, double right_samples) {
    return 6.0 * std::sqrt(probability * (1.0 - probability) *
                           (1.0 / left_samples + 1.0 / right_samples)) +
           1e-3;
}

}  // namespace

TEST_CASE("CUDA sampler zero shots does not require a device") {
    const CudaExecutablePlan executable(SamplingPlan{});

    REQUIRE(SamplingOptions{}.coefficient_precision == CoefficientPrecision::FP64);
    REQUIRE(SamplingOptions{}.tier == ExecutionTier::Auto);
    const SamplingResult rows = clifft::sampling::cuda::sample(executable, 0);
    const SamplingSurvivorResult survivors =
        clifft::sampling::cuda::sample_survivors(executable, 0);

    REQUIRE(rows.measurements.empty());
    REQUIRE(rows.detectors.empty());
    REQUIRE(rows.observables.empty());
    REQUIRE(rows.exp_vals.empty());
    REQUIRE(survivors.total_shots == 0);
    REQUIRE(survivors.passed_shots == 0);
    REQUIRE(survivors.observable_ones.empty());

    SamplingOptions invalid_low;
    invalid_low.block_size = 0;
    SamplingOptions invalid_high;
    invalid_high.block_size = 1025;
    SamplingOptions invalid_shape;
    invalid_shape.block_size = 96;
    SamplingOptions invalid_batch;
    invalid_batch.max_batch_shots = 0;
    REQUIRE_THROWS_AS(clifft::sampling::cuda::sample(executable, 0, invalid_low),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::cuda::sample_survivors(executable, 0, false, invalid_high),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::cuda::sample(executable, 0, invalid_shape),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::cuda::sample(executable, 0, invalid_batch),
                      std::invalid_argument);
}

TEST_CASE("CUDA replay validates unsupported inputs before device access") {
    const CudaExecutablePlan empty(SamplingPlan{});
    REQUIRE_THROWS_AS(clifft::sampling::cuda::replay_shot(empty, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);

    const CudaExecutablePlan noisy(plan_from("X_ERROR(0.1) 0\nM 0\n"));
    REQUIRE_THROWS_AS(clifft::sampling::cuda::replay_shot(noisy, std::array<uint8_t, 1>{0}),
                      std::invalid_argument);
}

TEST_CASE("CUDA tier selection follows width and shared memory") {
    const CudaExecutablePlan narrow(plan_from("H 0\nT 0\nH 0\nM 0\n"));
    const CudaExecutablePlan wide(plan_from(kWideCircuit));
    REQUIRE(narrow.peak_active_width() <= clifft::sampling::cuda::kThreadPerShotMaxActiveWidth);
    REQUIRE(wide.peak_active_width() == 6);
    require_cuda_device();

    REQUIRE(clifft::sampling::cuda::selected_tier(narrow) == ExecutionTier::ThreadPerShot);
    // Six coordinates need a few kilobytes, which every CUDA device offers.
    REQUIRE(clifft::sampling::cuda::selected_tier(wide) == ExecutionTier::BlockShared);
    REQUIRE(clifft::sampling::cuda::selected_tier(wide, CoefficientPrecision::FP32) ==
            ExecutionTier::BlockShared);

    for (const ExecutionTier tier : kExplicitTiers) {
        Sampler sampler(wide, CoefficientPrecision::FP64, 64, tier);
        CAPTURE(tier);
        REQUIRE(sampler.execution_tier() == tier);
        REQUIRE(sampler.max_batch_shots() == 64);
        REQUIRE(sampler.max_concurrent_shots() >= 1);
        REQUIRE(sampler.max_concurrent_shots() <= 64);
    }
    Sampler capped(wide, CoefficientPrecision::FP64, 64, ExecutionTier::BlockGlobal, 3);
    REQUIRE(capped.max_concurrent_shots() == 3);
}

TEST_CASE("CUDA replay matches CPU readout noise unreachability") {
    const SamplingPlan plan = plan_from("M 0\nREADOUT_NOISE(0.1) rec[-1]\n");
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_cuda_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const std::array<uint8_t, 1> forced{0};
        clifft::sampling::Executor cpu(cpu_executable);
        const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
        const clifft::sampling::cuda::ReplayResult actual =
            clifft::sampling::cuda::replay_shot(cuda_executable, forced, precision);
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

TEST_CASE("CUDA replay omits incomplete outputs from discarded paths") {
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
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_cuda_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const std::array<uint8_t, 1> forced{1};
        clifft::sampling::Executor cpu(cpu_executable);
        const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
        const clifft::sampling::cuda::ReplayResult actual =
            clifft::sampling::cuda::replay_shot(cuda_executable, forced, precision);
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

TEST_CASE("CUDA sampler is repeatable within each coefficient precision") {
    require_cuda_device();
    const CudaExecutablePlan executable(plan_from(R"(
        H 0
        T 0
        H 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )"));

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        SamplingOptions options;
        options.seed = uint64_t{1234};
        options.coefficient_precision = precision;
        const SamplingResult first = clifft::sampling::cuda::sample(executable, 4096, options);
        const SamplingResult second = clifft::sampling::cuda::sample(executable, 4096, options);
        require_same_rows(first, second);
    }
}

TEST_CASE("CUDA retained sampler preserves seeded rows across bounded batches") {
    require_cuda_device();
    const CudaExecutablePlan executable(plan_from(R"(
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
        SamplingOptions options;
        options.seed = uint64_t{1234};
        options.coefficient_precision = precision;
        options.block_size = 64;
        options.max_batch_shots = kShots;
        const SamplingResult single_batch =
            clifft::sampling::cuda::sample(executable, kShots, options);

        require_same_rows(batched, repeated);
        require_same_rows(batched, single_batch);
        REQUIRE(sampler.allocated_device_bytes() == allocated_bytes);
    }
}

TEST_CASE("CUDA cooperative concurrency cap preserves seeded rows") {
    const CudaExecutablePlan executable(plan_from(kWideCircuit));
    require_cuda_device();

    constexpr uint32_t kShots = 301;
    for (const ExecutionTier tier : {ExecutionTier::BlockShared, ExecutionTier::BlockGlobal}) {
        Sampler capped(executable, CoefficientPrecision::FP64, 50, tier, 3);
        Sampler open(executable, CoefficientPrecision::FP64, kShots, tier);
        CAPTURE(tier);
        REQUIRE(capped.max_concurrent_shots() == 3);
        const SamplingResult batched = capped.sample(kShots, uint64_t{77}, 32);
        const SamplingResult single = open.sample(kShots, uint64_t{77}, 32);
        require_same_rows(batched, single);
    }
}

TEST_CASE("CUDA shared samplers with different widths can run from concurrent callers") {
    require_cuda_device();
    int device = 0;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    constexpr uint32_t kShots = 8;
    constexpr size_t kAttempts = 64;

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const auto boundary = first_block_global_width(precision);
        REQUIRE(boundary.has_value());
        const std::array<CudaExecutablePlan, 2> programs{
            CudaExecutablePlan(plan_from(wide_circuit_text(*boundary - 1, false))),
            CudaExecutablePlan(plan_from(wide_circuit_text(*boundary - 2, false)))};
        std::array<SamplingResult, 2> expected;
        std::array<clifft::sampling::cuda::ReplayResult, 2> expected_replay;
        std::array<std::vector<uint8_t>, 2> forced;
        for (size_t caller = 0; caller < programs.size(); ++caller) {
            Sampler sampler(programs[caller], precision, kShots, ExecutionTier::BlockShared);
            expected[caller] = sampler.sample(kShots, uint64_t{73});
            forced[caller].assign(
                expected[caller].measurements.begin(),
                expected[caller].measurements.begin() + programs[caller].num_records());
            expected_replay[caller] = sampler.replay_shot(forced[caller]);
            REQUIRE(expected_replay[caller].reachable);
        }

        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors;
        std::array<std::vector<SamplingResult>, 2> actual;
        std::array<std::vector<clifft::sampling::cuda::ReplayResult>, 2> replay;
        auto run = [&](size_t caller) {
            try {
                // CUDA device selection belongs to the calling host thread.
                if (cudaSetDevice(device) != cudaSuccess) {
                    throw std::runtime_error("could not select the test device");
                }
                Sampler sampler(programs[caller], precision, kShots, ExecutionTier::BlockShared);
                for (size_t attempt = 0; attempt < kAttempts; ++attempt) {
                    rendezvous.arrive_and_wait();
                    actual[caller].push_back(sampler.sample(kShots, uint64_t{73}));
                    rendezvous.arrive_and_wait();
                    replay[caller].push_back(sampler.replay_shot(forced[caller]));
                }
            } catch (...) {
                errors[caller] = std::current_exception();
                rendezvous.arrive_and_drop();
            }
        };
        std::jthread first(run, 0);
        std::jthread second(run, 1);
        first.join();
        second.join();

        for (size_t caller = 0; caller < programs.size(); ++caller) {
            CAPTURE(precision, caller);
            if (errors[caller]) {
                std::rethrow_exception(errors[caller]);
            }
            REQUIRE(actual[caller].size() == kAttempts);
            REQUIRE(replay[caller].size() == kAttempts);
            for (size_t attempt = 0; attempt < kAttempts; ++attempt) {
                require_same_rows(actual[caller][attempt], expected[caller]);
                REQUIRE(replay[caller][attempt].reachable == expected_replay[caller].reachable);
                REQUIRE(replay[caller][attempt].survived == expected_replay[caller].survived);
                REQUIRE(replay[caller][attempt].log_probability ==
                        expected_replay[caller].log_probability);
                require_same_rows(replay[caller][attempt].outputs, expected_replay[caller].outputs);
            }
        }
    }
}

TEST_CASE("CUDA retained sampler preserves survivor order across bounded batches") {
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
    const CudaExecutablePlan executable(clifft::sampling::plan_sampling(hir, plan_options));
    require_cuda_device();

    constexpr uint32_t kShots = 263;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        Sampler sampler(executable, precision, 5);
        const SamplingSurvivorResult batched =
            sampler.sample_survivors(kShots, true, uint64_t{91}, 32);
        SamplingOptions options;
        options.seed = uint64_t{91};
        options.coefficient_precision = precision;
        options.block_size = 32;
        options.max_batch_shots = kShots;
        const SamplingSurvivorResult single_batch =
            clifft::sampling::cuda::sample_survivors(executable, kShots, true, options);

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

TEST_CASE("CUDA sampler computes expectation values with FP64 accumulation") {
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
    const CudaExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    require_cuda_device();
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 4, uint64_t{17});

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                               std::pair{CoefficientPrecision::FP32, 5e-6}}) {
        SamplingOptions options;
        options.seed = uint64_t{17};
        options.coefficient_precision = precision;
        const SamplingResult actual = clifft::sampling::cuda::sample(executable, 4, options);
        REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
        for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
            CAPTURE(precision, index);
            REQUIRE_THAT(actual.exp_vals[index],
                         Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
        }
    }
}

TEST_CASE("CUDA execution tiers agree with the CPU on a wide program") {
    const SamplingPlan plan = plan_from(kWideCircuit);
    const CudaExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    REQUIRE(executable.peak_active_width() == 6);
    require_cuda_device();

    // Expectation values precede the measurements, so they are deterministic
    // and check the strided rotation, promotion, and reduction kernels exactly.
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, 2, uint64_t{5});
    for (const ExecutionTier tier : kExplicitTiers) {
        for (const uint32_t block_size : {uint32_t{4}, uint32_t{256}}) {
            for (const auto& [precision, tolerance] :
                 {std::pair{CoefficientPrecision::FP64, 1e-12},
                  std::pair{CoefficientPrecision::FP32, 5e-6}}) {
                Sampler sampler(executable, precision, 2, tier);
                const SamplingResult actual = sampler.sample(2, uint64_t{5}, block_size);
                CAPTURE(tier, block_size, precision);
                REQUIRE(actual.exp_vals.size() == expected.exp_vals.size());
                for (size_t index = 0; index < actual.exp_vals.size(); ++index) {
                    CAPTURE(index);
                    REQUIRE_THAT(actual.exp_vals[index],
                                 Catch::Matchers::WithinAbs(expected.exp_vals[index], tolerance));
                }
            }
        }
    }

    constexpr uint32_t kShots = 20000;
    const SamplingResult cpu_rows = clifft::sampling::sample(cpu_executable, kShots, uint64_t{23});
    std::array<double, 6> cpu_marginals{};
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        for (size_t bit = 0; bit < 6; ++bit) {
            cpu_marginals[bit] += cpu_rows.measurements[6 * shot + bit];
        }
    }
    for (double& marginal : cpu_marginals) {
        marginal /= kShots;
    }
    for (const ExecutionTier tier : kExplicitTiers) {
        Sampler sampler(executable, CoefficientPrecision::FP64, kShots, tier);
        const SamplingResult rows = sampler.sample(kShots, uint64_t{23}, 64);
        REQUIRE(rows.measurements.size() == 6 * kShots);
        for (size_t bit = 0; bit < 6; ++bit) {
            double marginal = 0.0;
            for (uint32_t shot = 0; shot < kShots; ++shot) {
                marginal += rows.measurements[6 * shot + bit];
            }
            marginal /= kShots;
            const double tolerance =
                6.0 * standard_error(cpu_marginals[bit], static_cast<double>(kShots)) + 1e-3;
            CAPTURE(tier, bit, cpu_marginals[bit], marginal);
            REQUIRE_THAT(marginal, Catch::Matchers::WithinAbs(cpu_marginals[bit], tolerance));
        }
    }
}

TEST_CASE("CUDA replay matches every CPU measurement branch in every tier") {
    const auto test_case = GENERATE(Catch::Generators::from_range(clifft::test::kGpuReplayCases));
    CAPTURE(test_case.name);
    const SamplingPlan plan = plan_from(test_case.circuit);
    const CudaExecutablePlan executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    const uint32_t records = test_case.visible + test_case.hidden;
    REQUIRE(executable.num_visible_records() == test_case.visible);
    REQUIRE(executable.num_records() == records);
    REQUIRE(executable.peak_active_width() >= test_case.min_active_width);
    REQUIRE(cpu_executable.num_visible_records() == test_case.visible);
    REQUIRE(cpu_executable.num_hidden_records() == test_case.hidden);
    require_cuda_device();

    for (const ExecutionTier tier : kExplicitTiers) {
        CAPTURE(tier);
        for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-12},
                                                   std::pair{CoefficientPrecision::FP32, 2e-5}}) {
            Sampler sampler(executable, precision, 1, tier);
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
}

TEST_CASE("CUDA reset sampling preserves visible rows across batch boundaries") {
    const SamplingPlan plan = plan_from(clifft::test::kResetBatchCircuit);
    const CudaExecutablePlan executable(plan);
    REQUIRE(executable.num_visible_records() == 2);
    REQUIRE(executable.num_records() == 3);
    REQUIRE(executable.peak_active_width() >= 1);

    constexpr uint32_t kShots = 257;
    const CpuExecutablePlan cpu_executable(plan);
    clifft::test::require_reset_batch_rows(
        clifft::sampling::sample(cpu_executable, kShots, uint64_t{1234}), kShots);
    require_cuda_device();

    for (const ExecutionTier tier : kExplicitTiers) {
        CAPTURE(tier);
        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            CAPTURE(precision);
            Sampler batched(executable, precision, 7, tier);
            Sampler single_batch(executable, precision, kShots, tier);
            const SamplingResult rows = batched.sample(kShots, uint64_t{1234}, 64);
            require_same_rows(rows, single_batch.sample(kShots, uint64_t{1234}, 64));
            clifft::test::require_reset_batch_rows(rows, kShots);
        }
    }
}

// Regression coverage for the two cooperative publication hazards fixed
// alongside these cases. Both are races between lanes of one block, so no test
// can force the interleaving that exposes them; these circuits instead put the
// exact action sequences on hardware in every tier, across several block sizes,
// and assert the sequentially correct answer. Before the barriers went in, a
// lane that observed the published value reached a different control-flow
// decision from lane 0 and left the block at a different barrier.
TEST_CASE("CUDA replay resolves a deterministic branch identically in every tier") {
    const SamplingPlan plan = plan_from(kDeterministicBranchCircuit);
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    REQUIRE(cuda_executable.peak_active_width() == 7);
    REQUIRE(cuda_executable.num_visible_records() == 1);

    // The forced-zero path is unreachable and the forced-one path is not, so a
    // lane that mis-resolves the branch symbol disagrees with lane 0 about
    // whether the shot exists at all.
    const std::array<uint8_t, 1> forced_zero{0};
    const std::array<uint8_t, 1> forced_one{1};
    {
        clifft::sampling::Executor cpu(cpu_executable);
        REQUIRE_FALSE(cpu.replay_shot(forced_zero).reachable);
    }
    {
        clifft::sampling::Executor cpu(cpu_executable);
        REQUIRE(cpu.replay_shot(forced_one).reachable);
    }
    require_cuda_device();

    for (const ExecutionTier tier : kExplicitTiers) {
        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            Sampler sampler(cuda_executable, precision, 1, tier);
            CAPTURE(tier, precision);
            // Repeat so an unlucky warp interleaving has many chances to show.
            for (int attempt = 0; attempt < 32; ++attempt) {
                CAPTURE(attempt);
                REQUIRE_FALSE(sampler.replay_shot(forced_zero).reachable);
                const clifft::sampling::cuda::ReplayResult reached =
                    sampler.replay_shot(forced_one);
                REQUIRE(reached.reachable);
                REQUIRE(reached.outputs.measurements == std::vector<uint8_t>{1});
            }
        }
    }
}

TEST_CASE("CUDA postselected detectors read records before a later flip") {
    const SamplingPlan plan = plan_from(kDetectorThenFlipCircuit);
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    REQUIRE(cuda_executable.num_detectors() == 1);

    constexpr uint32_t kShots = 512;
    const SamplingSurvivorResult cpu_survivors =
        clifft::sampling::sample_survivors(cpu_executable, kShots, 11u);
    REQUIRE(cpu_survivors.total_shots == kShots);
    REQUIRE(cpu_survivors.passed_shots == kShots);
    require_cuda_device();

    for (const ExecutionTier tier : kExplicitTiers) {
        for (const CoefficientPrecision precision :
             {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
            // A block far wider than the 64-coefficient sweep leaves most warps
            // with no coefficient work, which is the shape most likely to let
            // one warp run ahead of another.
            for (const uint32_t block_size : {uint32_t{64}, uint32_t{256}, uint32_t{1024}}) {
                Sampler sampler(cuda_executable, precision, kShots, tier);
                const SamplingSurvivorResult survivors =
                    sampler.sample_survivors(kShots, false, 11u, block_size);
                CAPTURE(tier, precision, block_size);
                REQUIRE(survivors.total_shots == kShots);
                REQUIRE(survivors.passed_shots == kShots);
            }
        }
    }
}

TEST_CASE("CUDA sampler applies both asymmetric readout endpoints exactly") {
    const CudaExecutablePlan executable(plan_from(R"(
        M 0
        READOUT_NOISE(1, 0) rec[-1]
        X 1
        M 1
        READOUT_NOISE(0, 1) rec[-1]
    )"));
    require_cuda_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        SamplingOptions options;
        options.seed = uint64_t{3};
        options.coefficient_precision = precision;
        const SamplingResult result = clifft::sampling::cuda::sample(executable, 64, options);
        REQUIRE(result.measurements.size() == 128);
        for (size_t shot = 0; shot < 64; ++shot) {
            CAPTURE(precision, shot);
            REQUIRE(result.measurements[2 * shot] == 1);
            REQUIRE(result.measurements[2 * shot + 1] == 0);
        }
    }
}

TEST_CASE("CUDA sampler evaluates both observable value domains") {
    const SamplingPlan plan = plan_from(R"(
        X_ERROR(1) 1
        X_ERROR(1) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        READOUT_NOISE(1) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
    )");
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    constexpr uint32_t kShots = 16;
    const SamplingResult expected = clifft::sampling::sample(cpu_executable, kShots, uint64_t{11});
    REQUIRE(std::ranges::all_of(expected.measurements, [](uint8_t value) { return value == 0; }));
    for (uint32_t shot = 0; shot < kShots; ++shot) {
        REQUIRE(expected.observables[2 * shot] == 1);
        REQUIRE(expected.observables[2 * shot + 1] == 0);
    }
    require_cuda_device();

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        SamplingOptions options;
        options.seed = uint64_t{11};
        options.coefficient_precision = precision;
        const SamplingResult actual =
            clifft::sampling::cuda::sample(cuda_executable, kShots, options);
        CAPTURE(precision);
        require_same_rows(actual, expected);
    }
}

TEST_CASE("CUDA sampler matches the full categorical Pauli channel distribution") {
    const SamplingPlan plan = plan_from(R"(
        H 0
        CX 0 1
        PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0
        CX 0 1
        H 0
        M 0 1
    )");
    const CudaExecutablePlan cuda_executable(plan);
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
    require_cuda_device();

    constexpr uint32_t kShots = 50000;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        SamplingOptions options;
        options.seed = uint64_t{29};
        options.coefficient_precision = precision;
        const SamplingResult result =
            clifft::sampling::cuda::sample(cuda_executable, kShots, options);
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

TEST_CASE("CUDA survivor compaction retains complete rows") {
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
    const CudaExecutablePlan executable(plan);
    require_cuda_device();

    constexpr uint32_t kShots = 8192;
    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        SamplingOptions options;
        options.seed = uint64_t{43};
        options.coefficient_precision = precision;
        const SamplingSurvivorResult result =
            clifft::sampling::cuda::sample_survivors(executable, kShots, true, options);
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

TEST_CASE("CUDA sampler matches CPU survivor statistics with noise") {
    require_cuda_device();
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
    const CudaExecutablePlan cuda_executable(plan);
    const CpuExecutablePlan cpu_executable(plan);
    constexpr uint32_t kShots = 40000;

    SamplingOptions options;
    options.seed = uint64_t{91};
    const SamplingSurvivorResult gpu =
        clifft::sampling::cuda::sample_survivors(cuda_executable, kShots, true, options);
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

// The cooperative tiers are chosen by whether one shot fits the device's
// opt-in shared memory, so these cases run on both sides of that boundary with
// automatic selection, for both precisions, and compare the collapse and the
// state after it against the CPU exactly through forced replay, then the
// sampled rows statistically.
TEST_CASE("CUDA automatic tier selection at the shared-memory boundary matches the CPU") {
    require_cuda_device();
    constexpr uint32_t kShots = 4096;

    for (const auto& [precision, tolerance] : {std::pair{CoefficientPrecision::FP64, 1e-10},
                                               std::pair{CoefficientPrecision::FP32, 5e-5}}) {
        const std::optional<uint32_t> boundary = first_block_global_width(precision);
        CAPTURE(precision);
        REQUIRE(boundary.has_value());
        REQUIRE(*boundary > 8);

        for (const uint32_t width : {*boundary - 1, *boundary}) {
            const ExecutionTier expected_tier =
                width < *boundary ? ExecutionTier::BlockShared : ExecutionTier::BlockGlobal;
            const SamplingPlan plan = plan_from(wide_circuit_text(width, false));
            const CudaExecutablePlan cuda_executable(plan);
            const CpuExecutablePlan cpu_executable(plan);
            CAPTURE(width, expected_tier);
            REQUIRE(cuda_executable.peak_active_width() == width);
            REQUIRE(cuda_executable.num_records() == width);
            REQUIRE(cuda_executable.num_exp_vals() == 4);

            Sampler sampler(cuda_executable, precision, kShots);
            REQUIRE(sampler.execution_tier() == expected_tier);
            REQUIRE(sampler.max_concurrent_shots() >= 1);
            REQUIRE(sampler.max_concurrent_shots() <= kShots);
            const size_t slab_pool =
                sampler.max_concurrent_shots() * coefficient_slab_bytes(width, precision);
            if (expected_tier == ExecutionTier::BlockGlobal) {
                // Every resident block owns a global slab holding the whole shot.
                REQUIRE(sampler.allocated_device_bytes() >= slab_pool);
            } else {
                // The shared tier keeps every coefficient on chip.
                REQUIRE(sampler.allocated_device_bytes() < slab_pool);
            }

            // Forced replay pins the collapse: the log-probability of the path
            // and the expectation values taken after the measurement must
            // match the CPU executor on the same forced records.
            uint32_t reachable_paths = 0;
            for (const std::vector<uint8_t>& forced : forced_record_paths(width)) {
                clifft::sampling::Executor cpu(cpu_executable);
                const clifft::sampling::ReplayResult expected = cpu.replay_shot(forced);
                const clifft::sampling::cuda::ReplayResult actual = sampler.replay_shot(forced);
                CAPTURE(forced[0], forced[1]);
                REQUIRE(actual.reachable == expected.reachable);
                if (!expected.reachable) {
                    continue;
                }
                ++reachable_paths;
                REQUIRE_THAT(actual.log_probability,
                             Catch::Matchers::WithinAbs(expected.log_probability, tolerance));
                REQUIRE(actual.outputs.measurements == forced);
                REQUIRE(actual.outputs.exp_vals.size() == cpu.exp_vals().size());
                for (size_t index = 0; index < cpu.exp_vals().size(); ++index) {
                    CAPTURE(index);
                    REQUIRE_THAT(actual.outputs.exp_vals[index],
                                 Catch::Matchers::WithinAbs(cpu.exp_vals()[index], tolerance));
                }
            }
            REQUIRE(reachable_paths > 0);

            const SamplingResult cpu_rows =
                clifft::sampling::sample(cpu_executable, kShots, uint64_t{23});
            const SamplingResult gpu_rows = sampler.sample(kShots, uint64_t{23});
            REQUIRE(gpu_rows.measurements.size() == static_cast<size_t>(width) * kShots);
            REQUIRE(gpu_rows.exp_vals.size() == static_cast<size_t>(4) * kShots);
            for (uint32_t bit = 0; bit < width; ++bit) {
                double cpu_marginal = 0.0;
                double gpu_marginal = 0.0;
                for (uint32_t shot = 0; shot < kShots; ++shot) {
                    cpu_marginal += cpu_rows.measurements[static_cast<size_t>(width) * shot + bit];
                    gpu_marginal += gpu_rows.measurements[static_cast<size_t>(width) * shot + bit];
                }
                cpu_marginal /= kShots;
                gpu_marginal /= kShots;
                CAPTURE(bit, cpu_marginal, gpu_marginal);
                REQUIRE_THAT(gpu_marginal,
                             Catch::Matchers::WithinAbs(
                                 cpu_marginal, two_sample_tolerance(cpu_marginal, kShots, kShots)));
            }
            // The post-collapse expectation values vary with the sampled
            // outcome of qubit 0, so compare their means across shots.
            for (const size_t index : {size_t{2}, size_t{3}}) {
                double cpu_mean = 0.0;
                double gpu_mean = 0.0;
                double cpu_square = 0.0;
                for (uint32_t shot = 0; shot < kShots; ++shot) {
                    const double cpu_value = cpu_rows.exp_vals[4 * shot + index];
                    cpu_mean += cpu_value;
                    cpu_square += cpu_value * cpu_value;
                    gpu_mean += gpu_rows.exp_vals[4 * shot + index];
                }
                cpu_mean /= kShots;
                gpu_mean /= kShots;
                const double cpu_std =
                    std::sqrt(std::max(cpu_square / kShots - cpu_mean * cpu_mean, 0.0));
                CAPTURE(index, cpu_mean, gpu_mean, cpu_std);
                REQUIRE_THAT(gpu_mean,
                             Catch::Matchers::WithinAbs(
                                 cpu_mean, 6.0 * cpu_std * std::sqrt(2.0 / kShots) + 1e-3));
            }
        }
    }
}

TEST_CASE("CUDA cooperative tiers at the shared-memory boundary match CPU survivor statistics") {
    require_cuda_device();
    constexpr uint32_t kShots = 4096;

    for (const CoefficientPrecision precision :
         {CoefficientPrecision::FP64, CoefficientPrecision::FP32}) {
        const std::optional<uint32_t> boundary = first_block_global_width(precision);
        CAPTURE(precision);
        REQUIRE(boundary.has_value());

        for (const uint32_t width : {*boundary - 1, *boundary}) {
            const ExecutionTier expected_tier =
                width < *boundary ? ExecutionTier::BlockShared : ExecutionTier::BlockGlobal;
            const SamplingPlan plan = noisy_postselected_plan(width);
            const CudaExecutablePlan cuda_executable(plan);
            const CpuExecutablePlan cpu_executable(plan);
            CAPTURE(width, expected_tier);
            REQUIRE(cuda_executable.peak_active_width() == width);
            REQUIRE(cuda_executable.has_postselection());
            REQUIRE(cuda_executable.num_visible_records() == width);
            REQUIRE(cuda_executable.num_detectors() == 1);
            REQUIRE(cuda_executable.num_observables() == 1);

            const SamplingSurvivorResult expected =
                clifft::sampling::sample_survivors(cpu_executable, kShots, uint64_t{31}, true);
            REQUIRE(expected.total_shots == kShots);
            REQUIRE(expected.passed_shots > 0);
            REQUIRE(expected.passed_shots < kShots);

            Sampler sampler(cuda_executable, precision, kShots);
            REQUIRE(sampler.execution_tier() == expected_tier);
            const SamplingSurvivorResult actual =
                sampler.sample_survivors(kShots, true, uint64_t{31});
            REQUIRE(actual.total_shots == kShots);
            REQUIRE(actual.passed_shots > 0);
            REQUIRE(actual.passed_shots < kShots);

            const double cpu_pass = static_cast<double>(expected.passed_shots) / kShots;
            const double gpu_pass = static_cast<double>(actual.passed_shots) / kShots;
            CAPTURE(cpu_pass, gpu_pass);
            REQUIRE_THAT(gpu_pass, Catch::Matchers::WithinAbs(
                                       cpu_pass, two_sample_tolerance(cpu_pass, kShots, kShots)));

            // Every retained row is complete and self-consistent: the
            // postselected detector reads the measurement of qubit 1, so
            // that column is zero in every survivor, and the observable is
            // the parity of the last two records of the same row.
            const size_t rows = actual.passed_shots;
            REQUIRE(actual.measurements.size() == rows * width);
            REQUIRE(actual.detectors.size() == rows);
            REQUIRE(actual.observables.size() == rows);
            REQUIRE(actual.exp_vals.size() == rows * 4);
            REQUIRE(actual.observable_ones.size() == 1);
            size_t misaligned_rows = 0;
            uint64_t observable_ones = 0;
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t* record = &actual.measurements[row * width];
                const uint8_t parity = static_cast<uint8_t>(record[width - 1] ^ record[width - 2]);
                observable_ones += actual.observables[row];
                if (actual.detectors[row] != 0 || record[1] != 0 ||
                    actual.observables[row] != parity) {
                    ++misaligned_rows;
                }
            }
            REQUIRE(misaligned_rows == 0);
            REQUIRE(actual.observable_ones[0] == observable_ones);
            REQUIRE(actual.logical_errors == observable_ones);

            const double cpu_rows = expected.passed_shots;
            const double gpu_rows = actual.passed_shots;
            const double cpu_ones = static_cast<double>(expected.observable_ones[0]) / cpu_rows;
            const double gpu_ones = static_cast<double>(actual.observable_ones[0]) / gpu_rows;
            CAPTURE(cpu_ones, gpu_ones);
            REQUIRE_THAT(gpu_ones,
                         Catch::Matchers::WithinAbs(
                             cpu_ones, two_sample_tolerance(cpu_ones, cpu_rows, gpu_rows)));
            for (uint32_t bit = 0; bit < width; ++bit) {
                double cpu_marginal = 0.0;
                double gpu_marginal = 0.0;
                for (size_t row = 0; row < expected.passed_shots; ++row) {
                    cpu_marginal += expected.measurements[row * width + bit];
                }
                for (size_t row = 0; row < rows; ++row) {
                    gpu_marginal += actual.measurements[row * width + bit];
                }
                cpu_marginal /= cpu_rows;
                gpu_marginal /= gpu_rows;
                CAPTURE(bit, cpu_marginal, gpu_marginal);
                REQUIRE_THAT(
                    gpu_marginal,
                    Catch::Matchers::WithinAbs(
                        cpu_marginal, two_sample_tolerance(cpu_marginal, cpu_rows, gpu_rows)));
            }
        }
    }
}
