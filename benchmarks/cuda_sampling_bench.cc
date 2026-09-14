// CPU-versus-CUDA evidence for the experimental CUDA sampling backend.
//
// Each workload is compiled through the production pipeline (parse -> trace
// -> default HIR passes -> plan_sampling; --postselect marks every detector
// as postselected so survivor sampling runs). The CPU sampler is timed at the
// requested thread budgets, then every CUDA execution tier is timed twice:
// construction (program upload plus workspace allocation) and retained
// sampling of the same shot count after one warm-up call.
//
// Correctness is checked, not assumed: per-record marginals (and, for
// survivor sampling, the pass rate and observable-one rates) of every CUDA
// run are compared with the first CPU run under a two-sample binomial
// tolerance, and the largest |z| is reported per row. Exact row equality is
// not expected: CPU and CUDA draw from separate random-stream domains.
//
//   --width-sweep LO..HI   replace fixtures with a synthetic width-k family
//                          (k Hadamard/T pairs, CX layers, rotations,
//                          expectation values, measurements) to show where
//                          ThreadPerShot stops paying against BlockShared.
//   --concurrency-sweep a,b,...
//                          rerun the automatic tier of each workload with
//                          these max_concurrent_shots caps.
//
// Output is CSV on stdout:
//   workload,width,backend,variant,precision,shots,construct_s,seconds,
//   shots_per_s,device_bytes,max_z,passed
// with diagnostics on stderr.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Statistics of one run in a backend-independent shape.
struct Summary {
    double rows = 0;   // shots, or survivors when postselecting
    double total = 0;  // shots requested
    std::vector<double> marginals;
    std::vector<double> observable_rates;
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

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
                 "       fixture.stim ...\n");
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

Summary summarize(const SamplingResult& result, uint32_t shots, uint32_t records,
                  uint32_t observables) {
    Summary summary;
    summary.rows = shots;
    summary.total = shots;
    summary.marginals.assign(records, 0.0);
    summary.observable_rates.assign(observables, 0.0);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        for (uint32_t bit = 0; bit < records; ++bit) {
            summary.marginals[bit] +=
                result.measurements[static_cast<size_t>(shot) * records + bit];
        }
        for (uint32_t bit = 0; bit < observables; ++bit) {
            summary.observable_rates[bit] +=
                result.observables[static_cast<size_t>(shot) * observables + bit];
        }
    }
    for (double& value : summary.marginals) {
        value /= std::max(summary.rows, 1.0);
    }
    for (double& value : summary.observable_rates) {
        value /= std::max(summary.rows, 1.0);
    }
    return summary;
}

Summary summarize(const SamplingSurvivorResult& result, uint32_t records) {
    Summary summary;
    summary.rows = result.passed_shots;
    summary.total = result.total_shots;
    summary.marginals.assign(records, 0.0);
    for (uint32_t row = 0; row < result.passed_shots; ++row) {
        for (uint32_t bit = 0; bit < records; ++bit) {
            summary.marginals[bit] += result.measurements[static_cast<size_t>(row) * records + bit];
        }
    }
    for (double& value : summary.marginals) {
        value /= std::max(summary.rows, 1.0);
    }
    for (const uint64_t ones : result.observable_ones) {
        summary.observable_rates.push_back(static_cast<double>(ones) / std::max(summary.rows, 1.0));
    }
    return summary;
}

double z_score(double left, double left_n, double right, double right_n) {
    const double pooled = (left * left_n + right * right_n) / (left_n + right_n);
    const double variance = pooled * (1.0 - pooled) * (1.0 / left_n + 1.0 / right_n);
    if (variance <= 0.0) {
        return left == right ? 0.0 : INFINITY;
    }
    return std::fabs(left - right) / std::sqrt(variance);
}

// Largest |z| over the pass rate (survivor sampling only), every record
// marginal, and every observable rate.
double max_z(const Summary& reference, const Summary& candidate, bool survivors) {
    double worst = 0.0;
    if (survivors) {
        worst = z_score(reference.rows / reference.total, reference.total,
                        candidate.rows / candidate.total, candidate.total);
    }
    for (size_t bit = 0; bit < reference.marginals.size(); ++bit) {
        worst = std::max(worst, z_score(reference.marginals[bit], reference.rows,
                                        candidate.marginals[bit], candidate.rows));
    }
    for (size_t bit = 0; bit < reference.observable_rates.size(); ++bit) {
        worst = std::max(worst, z_score(reference.observable_rates[bit], reference.rows,
                                        candidate.observable_rates[bit], candidate.rows));
    }
    return worst;
}

void emit(const Workload& workload, uint32_t width, const char* backend, const std::string& variant,
          const char* precision, uint32_t shots, double construct_s, double seconds,
          size_t device_bytes, double z) {
    std::printf("%s,%u,%s,%s,%s,%u,%.6f,%.6f,%.1f,%zu,%.2f,%s\n", workload.name.c_str(), width,
                backend, variant.c_str(), precision, shots, construct_s, seconds,
                seconds > 0.0 ? shots / seconds : 0.0, device_bytes, z, z <= 5.0 ? "yes" : "no");
    std::fflush(stdout);
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

void run_workload(const Workload& workload, const Options& options) {
    const CpuExecutablePlan cpu_executable(workload.plan);
    const CudaExecutablePlan cuda_executable(workload.plan);
    const uint32_t width = cuda_executable.peak_active_width();
    const uint32_t records = cuda_executable.num_visible_records();
    const uint32_t observables = cuda_executable.num_observables();
    const uint32_t warmup = std::max<uint32_t>(options.shots / 10, 1);
    std::fprintf(stderr, "[%s] width=%u records=%u actions=%u survivors=%s\n",
                 workload.name.c_str(), width, records, cuda_executable.num_actions(),
                 workload.survivors ? "yes" : "no");

    std::optional<Summary> reference;
    for (const uint32_t threads : options.cpu_threads) {
        Summary summary;
        double seconds = 0.0;
        if (workload.survivors) {
            (void)clifft::sampling::sample_survivors(cpu_executable, warmup, options.seed, false,
                                                     threads);
            const auto start = std::chrono::steady_clock::now();
            const SamplingSurvivorResult result = clifft::sampling::sample_survivors(
                cpu_executable, options.shots, options.seed, true, threads);
            seconds = seconds_since(start);
            summary = summarize(result, records);
        } else {
            (void)clifft::sampling::sample(cpu_executable, warmup, options.seed, threads);
            const auto start = std::chrono::steady_clock::now();
            const SamplingResult result =
                clifft::sampling::sample(cpu_executable, options.shots, options.seed, threads);
            seconds = seconds_since(start);
            summary = summarize(result, options.shots, records, observables);
        }
        const double z = reference ? max_z(*reference, summary, workload.survivors) : 0.0;
        if (!reference) {
            reference = summary;
        }
        emit(workload, width, "cpu", "threads=" + std::to_string(threads), "fp64", options.shots,
             0.0, seconds, 0, z);
    }

    std::vector<GpuRun> runs = {{ExecutionTier::Auto},
                                {ExecutionTier::ThreadPerShot},
                                {ExecutionTier::BlockShared},
                                {ExecutionTier::BlockGlobal}};
    for (const uint32_t cap : options.concurrency_sweep) {
        runs.push_back({ExecutionTier::Auto, cap});
    }
    const uint32_t max_batch =
        std::min(options.shots, clifft::sampling::cuda::kDefaultMaxBatchShots);
    for (const CoefficientPrecision precision : options.precisions) {
        for (const GpuRun& run : runs) {
            std::string variant = tier_name(run.requested);
            try {
                const auto construct_start = std::chrono::steady_clock::now();
                clifft::sampling::cuda::Sampler sampler(cuda_executable, precision, max_batch,
                                                        run.requested, run.max_concurrent_shots);
                const double construct_s = seconds_since(construct_start);
                if (run.requested == ExecutionTier::Auto) {
                    variant += std::string("=") + tier_name(sampler.execution_tier());
                }
                if (run.max_concurrent_shots != 0) {
                    variant += "(cap=" + std::to_string(sampler.max_concurrent_shots()) + ")";
                }
                Summary summary;
                double seconds = 0.0;
                if (workload.survivors) {
                    (void)sampler.sample_survivors(warmup, false, options.seed, options.block_size);
                    const auto start = std::chrono::steady_clock::now();
                    const SamplingSurvivorResult result = sampler.sample_survivors(
                        options.shots, true, options.seed, options.block_size);
                    seconds = seconds_since(start);
                    summary = summarize(result, records);
                } else {
                    (void)sampler.sample(warmup, options.seed, options.block_size);
                    const auto start = std::chrono::steady_clock::now();
                    const SamplingResult result =
                        sampler.sample(options.shots, options.seed, options.block_size);
                    seconds = seconds_since(start);
                    summary = summarize(result, options.shots, records, observables);
                }
                emit(workload, width, "cuda", variant, precision_name(precision), options.shots,
                     construct_s, seconds, sampler.allocated_device_bytes(),
                     max_z(*reference, summary, workload.survivors));
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[%s] %s %s skipped: %s\n", workload.name.c_str(),
                             variant.c_str(), precision_name(precision), error.what());
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    if (!clifft::sampling::cuda::is_available()) {
        std::fprintf(stderr, "no CUDA device: %s\n",
                     clifft::sampling::cuda::backend_info().c_str());
        return 1;
    }
    std::fprintf(stderr, "%s\n", clifft::sampling::cuda::backend_info().c_str());
    std::printf(
        "workload,width,backend,variant,precision,shots,construct_s,seconds,"
        "shots_per_s,device_bytes,max_z,passed\n");
    for (const Workload& workload : build_workloads(options)) {
        run_workload(workload, options);
    }
    return 0;
}
