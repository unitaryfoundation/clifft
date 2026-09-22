// Retained sampling and sampler construction benchmarks for the CUDA backend.
// See benchmarks/README.md for timing boundaries and workload controls.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using clifft::sampling::SamplingPlan;
using clifft::sampling::SamplingResult;
using clifft::sampling::SamplingSurvivorResult;
using clifft::sampling::cuda::CoefficientPrecision;
using clifft::sampling::cuda::ExecutionTier;
using CpuExecutablePlan = clifft::sampling::ExecutablePlan;
using CudaExecutablePlan = clifft::sampling::cuda::ExecutablePlan;

struct Options {
    std::vector<std::string> fixtures;
    uint32_t shots = 100000;
    uint64_t seed = 42;
    std::vector<uint32_t> cpu_threads = {1, 0};  // 0 = hardware concurrency
    uint32_t block_size = clifft::sampling::cuda::kDefaultBlockSize;
    std::vector<CoefficientPrecision> precisions = {CoefficientPrecision::FP64};
    bool postselect = false;
    std::optional<std::pair<uint32_t, uint32_t>> width_sweep;
    std::vector<uint32_t> concurrency_sweep;
};

struct Workload {
    std::string name;
    SamplingPlan plan;
    bool survivors = false;
};

std::vector<uint32_t> parse_list(const std::string& text) {
    std::vector<uint32_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return values;
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: clifft_cuda_bench [--shots N] [--seed S] [--threads a,b,...]\n"
                 "       [--block-size N] [--precision fp64|fp32|both] [--postselect]\n"
                 "       [--width-sweep LO..HI] [--concurrency-sweep a,b,...]\n"
                 "       [Google Benchmark flags] fixture.stim ...\n");
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string value = argv[arg];
        auto next = [&](const char* flag) -> std::string {
            if (arg + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", flag);
                usage();
            }
            return argv[++arg];
        };
        if (value == "--shots") {
            options.shots = static_cast<uint32_t>(std::stoul(next("--shots")));
        } else if (value == "--seed") {
            options.seed = std::stoull(next("--seed"));
        } else if (value == "--threads") {
            options.cpu_threads = parse_list(next("--threads"));
        } else if (value == "--block-size") {
            options.block_size = static_cast<uint32_t>(std::stoul(next("--block-size")));
        } else if (value == "--precision") {
            const std::string precision = next("--precision");
            if (precision == "fp64") {
                options.precisions = {CoefficientPrecision::FP64};
            } else if (precision == "fp32") {
                options.precisions = {CoefficientPrecision::FP32};
            } else if (precision == "both") {
                options.precisions = {CoefficientPrecision::FP64, CoefficientPrecision::FP32};
            } else {
                usage();
            }
        } else if (value == "--postselect") {
            options.postselect = true;
        } else if (value == "--width-sweep") {
            const std::string range = next("--width-sweep");
            const size_t dots = range.find("..");
            if (dots == std::string::npos) {
                usage();
            }
            options.width_sweep = {static_cast<uint32_t>(std::stoul(range.substr(0, dots))),
                                   static_cast<uint32_t>(std::stoul(range.substr(dots + 2)))};
        } else if (value == "--concurrency-sweep") {
            options.concurrency_sweep = parse_list(next("--concurrency-sweep"));
        } else if (!value.empty() && value[0] == '-') {
            usage();
        } else {
            options.fixtures.push_back(value);
        }
    }
    if (options.shots == 0 || options.cpu_threads.empty() ||
        (options.width_sweep &&
         (options.width_sweep->first == 0 || options.width_sweep->second > 30 ||
          options.width_sweep->first > options.width_sweep->second))) {
        usage();
    }
    if (options.fixtures.empty() && !options.width_sweep) {
        usage();
    }
    return options;
}

// Same family as the shared-memory boundary tests in tests/test_cuda_sampler.cc:
// k promoted coordinates, rotations, expectation values before and after a
// collapsing measurement, and a full readout.
std::string width_circuit_text(uint32_t width) {
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
    if (width >= 3) {
        text << "R_PAULI(0.21) X0*Y1\nEXP_VAL X0\nEXP_VAL Z1*Z2\nM 0\n"
                "R_PAULI(0.13) Z1*X2\nEXP_VAL X1\nEXP_VAL Y2\nM";
        for (uint32_t qubit = 1; qubit < width; ++qubit) {
            text << " " << qubit;
        }
    } else {
        text << "R_PAULI(0.21) X0\nEXP_VAL X0\nM";
        for (uint32_t qubit = 0; qubit < width; ++qubit) {
            text << " " << qubit;
        }
    }
    text << "\n";
    return text.str();
}

SamplingPlan plan_hir(clifft::HirModule hir, bool postselect_all) {
    clifft::default_hir_pass_manager().run(hir);
    if (!postselect_all) {
        return clifft::sampling::plan_sampling(hir);
    }
    const SamplingPlan probe = clifft::sampling::plan_sampling(hir);
    const std::vector<uint8_t> mask(probe.num_detectors, uint8_t{1});
    clifft::sampling::SamplingPlanOptions options;
    options.postselection_mask = mask;
    return clifft::sampling::plan_sampling(hir, options);
}

std::vector<Workload> build_workloads(const Options& options) {
    std::vector<Workload> workloads;
    if (options.width_sweep) {
        for (uint32_t width = options.width_sweep->first; width <= options.width_sweep->second;
             ++width) {
            Workload workload;
            workload.name = "width-" + std::to_string(width);
            workload.plan =
                plan_hir(clifft::trace(clifft::parse(width_circuit_text(width))), false);
            workloads.push_back(std::move(workload));
        }
    }
    for (const std::string& path : options.fixtures) {
        Workload workload;
        workload.name = std::filesystem::path(path).stem().string();
        workload.plan = plan_hir(clifft::trace(clifft::parse_file(path)), options.postselect);
        workload.survivors = options.postselect && workload.plan.num_detectors > 0;
        workloads.push_back(std::move(workload));
    }
    return workloads;
}

const char* tier_name(ExecutionTier tier) {
    switch (tier) {
        case ExecutionTier::Auto:
            return "auto";
        case ExecutionTier::ThreadPerShot:
            return "thread_per_shot";
        case ExecutionTier::BlockShared:
            return "block_shared";
        case ExecutionTier::BlockGlobal:
            return "block_global";
    }
    return "?";
}

const char* precision_name(CoefficientPrecision precision) {
    return precision == CoefficientPrecision::FP32 ? "fp32" : "fp64";
}

struct GpuRun {
    ExecutionTier requested;
    uint32_t max_concurrent_shots = 0;
};

template <typename Result>
bool valid_rows(const Result& result, const SamplingPlan& plan, size_t rows) {
    return result.measurements.size() == rows * plan.num_visible_records &&
           result.detectors.size() == rows * plan.num_detectors &&
           result.observables.size() == rows * plan.num_observables &&
           result.exp_vals.size() == rows * plan.num_exp_vals;
}

bool valid_output(const SamplingResult& result, const SamplingPlan& plan, uint32_t shots) {
    return valid_rows(result, plan, shots);
}

bool valid_output(const SamplingSurvivorResult& result, const SamplingPlan& plan, uint32_t shots) {
    return result.total_shots == shots && result.passed_shots <= shots &&
           result.logical_errors <= result.passed_shots &&
           result.observable_ones.size() == plan.num_observables &&
           std::all_of(result.observable_ones.begin(), result.observable_ones.end(),
                       [&](uint64_t ones) { return ones <= result.passed_shots; }) &&
           valid_rows(result, plan, result.passed_shots);
}

template <typename Sample>
void measure_sampling(benchmark::State& state, const Workload& workload, const Options& options,
                      Sample sample) {
    // Google Benchmark starts timing at the iteration loop. Keep a small
    // output-shape check here; statistical conformance belongs in the tests.
    const uint32_t warmup = std::min(options.shots, uint32_t{64});
    if (!valid_output(sample(warmup), workload.plan, warmup)) {
        state.SkipWithError("warmup returned inconsistent output shapes or counts");
        return;
    }
    for (auto _ : state) {
        auto result = sample(options.shots);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * options.shots);
}

void run_cpu(benchmark::State& state, const Workload& workload, const Options& options,
             uint32_t threads) {
    try {
        const CpuExecutablePlan executable(workload.plan);
        state.counters["active_width"] = executable.peak_active_width();
        if (workload.survivors) {
            measure_sampling(state, workload, options, [&](uint32_t shots) {
                return clifft::sampling::sample_survivors(executable, shots, options.seed, true,
                                                          threads);
            });
        } else {
            measure_sampling(state, workload, options, [&](uint32_t shots) {
                return clifft::sampling::sample(executable, shots, options.seed, threads);
            });
        }
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
    }
}

void describe_sampler(benchmark::State& state, const clifft::sampling::cuda::Sampler& sampler) {
    state.SetLabel(tier_name(sampler.execution_tier()));
    state.counters["device_bytes"] = static_cast<double>(sampler.allocated_device_bytes());
    state.counters["max_concurrent_shots"] = sampler.max_concurrent_shots();
}

void run_cuda(benchmark::State& state, const Workload& workload, const Options& options,
              CoefficientPrecision precision, GpuRun run, bool construct_only) {
    if (!clifft::sampling::cuda::is_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    try {
        const CudaExecutablePlan executable(workload.plan);
        state.counters["active_width"] = executable.peak_active_width();
        const uint32_t max_batch =
            std::min(options.shots, clifft::sampling::cuda::kDefaultMaxBatchShots);
        auto make_sampler = [&] {
            return clifft::sampling::cuda::Sampler(executable, precision, max_batch, run.requested,
                                                   run.max_concurrent_shots);
        };
        if (construct_only) {
            // Exclude context initialization and one-time kernel setup, and
            // release the probe so it does not reduce the measured memory budget.
            {
                const auto probe = make_sampler();
                describe_sampler(state, probe);
            }
            for (auto _ : state) {
                auto sampler = make_sampler();
                benchmark::DoNotOptimize(sampler);
            }
        } else {
            auto sampler = make_sampler();
            describe_sampler(state, sampler);
            if (workload.survivors) {
                measure_sampling(state, workload, options, [&](uint32_t shots) {
                    return sampler.sample_survivors(shots, true, options.seed, options.block_size);
                });
            } else {
                measure_sampling(state, workload, options, [&](uint32_t shots) {
                    return sampler.sample(shots, options.seed, options.block_size);
                });
            }
        }
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
    }
}

void register_workload(const Workload& workload, const Options& options) {
    const std::string operation = workload.survivors ? "sample_survivors" : "sample";
    const std::string shot_label = "/shots:" + std::to_string(options.shots);
    for (const uint32_t threads : options.cpu_threads) {
        const std::string name = workload.name + "/cpu/" + operation +
                                 "/threads:" + std::to_string(threads) + shot_label;
        benchmark::RegisterBenchmark(name.c_str(),
                                     [&workload, &options, threads](auto& state) {
                                         run_cpu(state, workload, options, threads);
                                     })
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    }
    std::vector<GpuRun> runs = {{ExecutionTier::Auto},
                                {ExecutionTier::ThreadPerShot},
                                {ExecutionTier::BlockShared},
                                {ExecutionTier::BlockGlobal}};
    for (const uint32_t cap : options.concurrency_sweep) {
        runs.push_back({ExecutionTier::Auto, cap});
    }
    for (const CoefficientPrecision precision : options.precisions) {
        for (const GpuRun run : runs) {
            for (const bool construct_only : {false, true}) {
                const std::string name =
                    workload.name + "/cuda/" + (construct_only ? "construct_destroy" : operation) +
                    "/" + tier_name(run.requested) + "/" + precision_name(precision) + shot_label +
                    "/block:" + std::to_string(options.block_size) +
                    "/cap:" + std::to_string(run.max_concurrent_shots);
                benchmark::RegisterBenchmark(
                    name.c_str(),
                    [&workload, &options, precision, run, construct_only](auto& state) {
                        run_cuda(state, workload, options, precision, run, construct_only);
                    })
                    ->UseRealTime()
                    ->Unit(benchmark::kMillisecond);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    try {
        const Options options = parse_options(argc, argv);
        const auto workloads = build_workloads(options);
        for (const Workload& workload : workloads) {
            register_workload(workload, options);
        }
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        benchmark::Shutdown();
        return 1;
    }
    return 0;
}
