// GPU-vs-CPU throughput benchmark for the symbolic sampler.
//
// Compiles a .stim circuit with the production pipeline (parse -> trace ->
// default HIR passes -> plan_sampling), then races:
//   * the CPU sampler at one or more thread counts (the production baseline,
//     including cross-shot and intra-shot parallelism as shipped), and
//   * the CUDA device interpreter per execution tier (thread-per-shot,
//     block-per-shot shared, block-per-shot global), when built with
//     CLIFFT_ENABLE_CUDA and a device is present.
//
// Validation before timing, on request (--validate N):
//   * replays N GPU-sampled record rows through the CPU executable's
//     record_log_probabilities and rejects any unreachable row;
//   * compares per-record marginal one-frequencies between CPU and GPU
//     samples of the same size against a binomial tolerance.
//
// Output: CSV rows "circuit,backend,tier_or_threads,shots,seconds,shots_per_s"
// on stdout; diagnostics on stderr.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/cuda/executable.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#ifdef CLIFFT_BENCH_HAVE_CUDA
#include "clifft/sampling/cuda/sampler.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Options {
    std::vector<std::string> circuits;
    uint32_t shots = 100000;
    uint64_t seed = 42;
    std::vector<uint32_t> cpu_threads = {1, 0};  // 0 = hardware concurrency
    uint32_t validate_rows = 64;
    uint32_t block_size = 256;
    bool run_cpu = true;
    bool run_gpu = true;
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void emit_row(const std::string& circuit, const char* backend, const std::string& variant,
              uint32_t shots, double seconds) {
    std::printf("%s,%s,%s,%u,%.6f,%.1f\n", circuit.c_str(), backend, variant.c_str(), shots,
                seconds, seconds > 0 ? shots / seconds : 0.0);
    std::fflush(stdout);
}

clifft::sampling::SamplingPlan compile_plan(const std::string& path) {
    clifft::HirModule hir = clifft::trace(clifft::parse_file(path));
    clifft::HirPassManager passes = clifft::default_hir_pass_manager();
    passes.run(hir);
    return clifft::sampling::plan_sampling(hir);
}

bool validate_marginals(const clifft::sampling::SamplingResult& cpu,
                        const clifft::sampling::SamplingResult& gpu, uint32_t shots,
                        uint32_t num_records, const std::string& name) {
    bool ok = true;
    for (uint32_t record = 0; record < num_records; ++record) {
        uint64_t cpu_ones = 0;
        uint64_t gpu_ones = 0;
        for (uint32_t shot = 0; shot < shots; ++shot) {
            cpu_ones += cpu.measurements[static_cast<size_t>(shot) * num_records + record];
            gpu_ones += gpu.measurements[static_cast<size_t>(shot) * num_records + record];
        }
        const double p = static_cast<double>(cpu_ones) / shots;
        const double q = static_cast<double>(gpu_ones) / shots;
        // Two independent binomial estimates of the same marginal: allow six
        // combined standard errors plus a floor for tiny probabilities.
        const double sigma = std::sqrt(2.0 * std::max(p * (1 - p), q * (1 - q)) / shots);
        const double tolerance = 6.0 * sigma + 3.0 / shots;
        if (std::abs(p - q) > tolerance) {
            std::fprintf(stderr, "[validate] %s record %u marginal mismatch cpu=%.5f gpu=%.5f\n",
                         name.c_str(), record, p, q);
            ok = false;
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string value = argv[arg];
        auto next = [&](const char* flag) -> std::string {
            if (arg + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++arg];
        };
        if (value == "--shots") {
            options.shots = static_cast<uint32_t>(std::stoul(next("--shots")));
        } else if (value == "--seed") {
            options.seed = std::stoull(next("--seed"));
        } else if (value == "--threads") {
            options.cpu_threads.clear();
            std::string list = next("--threads");
            size_t begin = 0;
            while (begin <= list.size()) {
                const size_t comma = std::min(list.find(',', begin), list.size());
                options.cpu_threads.push_back(
                    static_cast<uint32_t>(std::stoul(list.substr(begin, comma - begin))));
                begin = comma + 1;
            }
        } else if (value == "--validate") {
            options.validate_rows = static_cast<uint32_t>(std::stoul(next("--validate")));
        } else if (value == "--block-size") {
            options.block_size = static_cast<uint32_t>(std::stoul(next("--block-size")));
        } else if (value == "--cpu-only") {
            options.run_gpu = false;
        } else if (value == "--gpu-only") {
            options.run_cpu = false;
        } else if (!value.empty() && value[0] == '-') {
            std::fprintf(stderr,
                         "usage: bench_sampling [--shots N] [--seed S] [--threads a,b,...]\n"
                         "       [--validate N] [--block-size N] [--cpu-only|--gpu-only]\n"
                         "       circuit.stim [more.stim ...]\n");
            return 2;
        } else {
            options.circuits.push_back(value);
        }
    }
    if (options.circuits.empty()) {
        std::fprintf(stderr, "no circuits given\n");
        return 2;
    }

    std::printf("circuit,backend,variant,shots,seconds,shots_per_s\n");
    for (const std::string& path : options.circuits) {
        const std::string name = std::filesystem::path(path).stem().string();
        clifft::sampling::SamplingPlan plan;
        try {
            plan = compile_plan(path);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[compile] %s failed: %s\n", name.c_str(), error.what());
            continue;
        }
        const clifft::sampling::ExecutablePlan cpu_plan(plan);
        std::fprintf(stderr, "[plan] %s actions=%zu peak_width=%u records=%u\n", name.c_str(),
                     cpu_plan.num_actions(), cpu_plan.peak_active_width(),
                     cpu_plan.num_visible_records());
        const bool postselected = cpu_plan.has_postselection();

        std::optional<clifft::sampling::SamplingResult> cpu_reference;
        if (options.run_cpu) {
            for (uint32_t threads : options.cpu_threads) {
                const auto start = std::chrono::steady_clock::now();
                if (postselected) {
                    (void)clifft::sampling::sample_survivors(cpu_plan, options.shots, options.seed,
                                                             false, threads);
                } else {
                    auto result =
                        clifft::sampling::sample(cpu_plan, options.shots, options.seed, threads);
                    if (!cpu_reference.has_value()) {
                        cpu_reference = std::move(result);
                    }
                }
                emit_row(name, "cpu", "threads=" + std::to_string(threads), options.shots,
                         seconds_since(start));
            }
        }

#ifdef CLIFFT_BENCH_HAVE_CUDA
        if (!options.run_gpu) {
            continue;
        }
        namespace cuda = clifft::sampling::cuda;
        if (!cuda::is_available()) {
            std::fprintf(stderr, "[gpu] no CUDA device; skipping\n");
            continue;
        }
        cuda::Executable executable(plan);
        cuda::SamplingOptions gpu_options;
        gpu_options.seed = options.seed;
        gpu_options.block_size = options.block_size;

        const cuda::ExecutionTier auto_tier = cuda::selected_tier(executable, gpu_options);
        struct TierRun {
            cuda::ExecutionTier tier;
            const char* label;
        };
        const TierRun tiers[] = {
            {cuda::ExecutionTier::ThreadPerShot, "thread_per_shot"},
            {cuda::ExecutionTier::BlockShared, "block_shared"},
            {cuda::ExecutionTier::BlockGlobal, "block_global"},
        };
        for (const TierRun& run : tiers) {
            gpu_options.tier = run.tier;
            const bool is_auto = run.tier == auto_tier;
            std::optional<clifft::sampling::SamplingResult> gpu_result;
            try {
                // Warm-up launch outside the timed region (context + JIT).
                if (postselected) {
                    (void)cuda::sample_survivors(executable, 1, false, gpu_options);
                } else {
                    (void)cuda::sample(executable, 1, gpu_options);
                }
                const auto start = std::chrono::steady_clock::now();
                if (postselected) {
                    (void)cuda::sample_survivors(executable, options.shots, false, gpu_options);
                    emit_row(name, "cuda", std::string(run.label) + (is_auto ? "*" : ""),
                             options.shots, seconds_since(start));
                } else {
                    gpu_result = cuda::sample(executable, options.shots, gpu_options);
                    emit_row(name, "cuda", std::string(run.label) + (is_auto ? "*" : ""),
                             options.shots, seconds_since(start));
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[gpu] %s tier %s unavailable: %s\n", name.c_str(), run.label,
                             error.what());
                continue;
            }

            if (gpu_result.has_value() && options.validate_rows > 0) {
                const uint32_t num_records = cpu_plan.num_visible_records();
                if (cpu_reference.has_value() && num_records > 0) {
                    (void)validate_marginals(*cpu_reference, *gpu_result, options.shots,
                                             num_records, name);
                }
                if (num_records > 0 && cpu_plan.num_hidden_records() == 0 &&
                    executable.noise_sites().empty()) {
                    const uint32_t rows = std::min(options.validate_rows, options.shots);
                    std::vector<uint8_t> forced(static_cast<size_t>(rows) * num_records);
                    std::copy_n(gpu_result->measurements.begin(), forced.size(), forced.begin());
                    const std::vector<double> log_probabilities =
                        clifft::sampling::record_log_probabilities(cpu_plan, forced, rows);
                    uint32_t unreachable = 0;
                    for (double value : log_probabilities) {
                        if (value <= std::numeric_limits<double>::lowest()) {
                            ++unreachable;
                        }
                    }
                    if (unreachable != 0) {
                        std::fprintf(stderr,
                                     "[validate] %s tier %s: %u of %u GPU rows unreachable on "
                                     "CPU replay\n",
                                     name.c_str(), run.label, unreachable, rows);
                    } else {
                        std::fprintf(stderr, "[validate] %s tier %s: %u rows replayed OK\n",
                                     name.c_str(), run.label, rows);
                    }
                }
            }
        }
#else
        if (options.run_gpu) {
            std::fprintf(stderr, "[gpu] built without CLIFFT_ENABLE_CUDA; CPU rows only for %s\n",
                         name.c_str());
        }
#endif
    }
    return 0;
}
