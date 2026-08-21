// Clifft sampling profiler
//
// Compiles a circuit once, then repeatedly samples it through the public C++
// path. This keeps compilation and file IO outside the measured interval while
// retaining executor construction, state initialization, and result collection.
//
// See tools/profile/README.md for build instructions.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultShots = 1;
constexpr int kDefaultThreads = 1;
constexpr int kDefaultWarmups = 2;
constexpr int kDefaultRepetitions = 20;
constexpr int kDefaultGeneratedDepth = 20;
constexpr uint64_t kSeed = 42;

int get_env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::stoi(value);
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file: " << path << "\n";
        std::exit(1);
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string generate_circuit(int width, int depth) {
    std::ostringstream circuit;
    for (int layer = 0; layer < depth; ++layer) {
        for (int qubit = 0; qubit < width; ++qubit) {
            const double offset = static_cast<double>((layer * width + qubit) % 17) / 1000.0;
            circuit << "U3(" << 0.123 + offset << ',' << 0.234 + offset << ',' << -0.345 - offset
                    << ") " << qubit << '\n';
        }
        for (int qubit = layer & 1; qubit + 1 < width; qubit += 2) {
            circuit << "CX " << qubit << ' ' << qubit + 1 << '\n';
        }
    }
    circuit << "M";
    for (int qubit = 0; qubit < width; ++qubit) {
        circuit << ' ' << qubit;
    }
    circuit << '\n';
    return circuit.str();
}

uint64_t mix(uint64_t checksum, uint64_t value) {
    checksum ^= value + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);
    return checksum;
}

uint64_t checksum_result(const clifft::sampling::SamplingResult& result) {
    uint64_t checksum = 0;
    for (uint8_t value : result.measurements) {
        checksum = mix(checksum, value);
    }
    for (uint8_t value : result.detectors) {
        checksum = mix(checksum, value);
    }
    for (uint8_t value : result.observables) {
        checksum = mix(checksum, value);
    }
    for (double value : result.exp_vals) {
        checksum = mix(checksum, std::bit_cast<uint64_t>(value));
    }
    return checksum;
}

void summarize(std::vector<double> samples_ms, int shots) {
    std::sort(samples_ms.begin(), samples_ms.end());
    const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    const double mean = sum / static_cast<double>(samples_ms.size());
    const double median = samples_ms[samples_ms.size() / 2];
    const double p95 = samples_ms[(samples_ms.size() * 95 - 1) / 100];

    std::cout << "Sampling distribution:\n";
    std::cout << "  min:    " << samples_ms.front() << " ms\n";
    std::cout << "  median: " << median << " ms\n";
    std::cout << "  mean:   " << mean << " ms\n";
    std::cout << "  p95:    " << p95 << " ms\n";
    std::cout << "  max:    " << samples_ms.back() << " ms\n";
    std::cout << "  median per shot: " << median / static_cast<double>(shots) << " ms\n";
}

}  // namespace

int main() {
    const char* circuit_file = std::getenv("CLIFFT_CIRCUIT_FILE");
    const int shots = get_env_int("CLIFFT_PROFILE_SHOTS", kDefaultShots);
    const int threads = get_env_int("CLIFFT_PROFILE_THREADS", kDefaultThreads);
    const int warmups = get_env_int("CLIFFT_PROFILE_WARMUPS", kDefaultWarmups);
    const int repetitions = get_env_int("CLIFFT_PROFILE_REPETITIONS", kDefaultRepetitions);
    const int generated_width = get_env_int("CLIFFT_PROFILE_GENERATED_WIDTH", 0);
    const int generated_depth =
        get_env_int("CLIFFT_PROFILE_GENERATED_DEPTH", kDefaultGeneratedDepth);
    const char* shot_workers_value = std::getenv("CLIFFT_PROFILE_SHOT_WORKERS");
    const char* intra_workers_value = std::getenv("CLIFFT_PROFILE_INTRA_SHOT_WORKERS");
    const char* min_active_width_value = std::getenv("CLIFFT_PROFILE_INTRA_SHOT_MIN_ACTIVE_WIDTH");
    const bool has_layout = shot_workers_value != nullptr || intra_workers_value != nullptr;
    const int shot_workers = get_env_int("CLIFFT_PROFILE_SHOT_WORKERS", 0);
    const int intra_shot_workers = get_env_int("CLIFFT_PROFILE_INTRA_SHOT_WORKERS", 0);
    const int intra_shot_min_active_width =
        get_env_int("CLIFFT_PROFILE_INTRA_SHOT_MIN_ACTIVE_WIDTH",
                    static_cast<int>(clifft::kDefaultIntraShotMinActiveWidth));
    if (shots < 1 || threads < 0 || warmups < 0 || repetitions < 1) {
        std::cerr << "Error: shots and repetitions must be positive; threads and warmups must be "
                     "non-negative\n";
        return 1;
    }
    if ((circuit_file == nullptr || std::string(circuit_file).empty()) && generated_width < 1) {
        std::cerr << "Error: set CLIFFT_CIRCUIT_FILE or CLIFFT_PROFILE_GENERATED_WIDTH\n";
        return 1;
    }
    if (generated_width < 0 || generated_depth < 1) {
        std::cerr << "Error: generated width must be non-negative and depth must be positive\n";
        return 1;
    }
    if (has_layout && (shot_workers < 1 || intra_shot_workers < 1)) {
        std::cerr << "Error: set both layout worker counts to positive values\n";
        return 1;
    }
    if (min_active_width_value != nullptr && !has_layout) {
        std::cerr << "Error: the intra-shot minimum active width requires an explicit layout\n";
        return 1;
    }
    if (intra_shot_min_active_width < 0) {
        std::cerr << "Error: the intra-shot minimum active width must be non-negative\n";
        return 1;
    }

    std::cout << "Clifft Sampling Profiler\n";
    std::cout << "========================\n";
    if (circuit_file != nullptr && !std::string(circuit_file).empty()) {
        std::cout << "Circuit:     " << circuit_file << "\n";
    } else {
        std::cout << "Circuit:     generated width " << generated_width << ", depth "
                  << generated_depth << "\n";
    }
    std::cout << "Shots:       " << shots << "\n";
    std::cout << "Threads:     " << threads << "\n";
    if (has_layout) {
        std::cout << "Layout:      " << shot_workers << " shot x " << intra_shot_workers
                  << " intra-shot\n";
        std::cout << "Min width:   " << intra_shot_min_active_width << "\n";
    }
    std::cout << "Warmups:     " << warmups << "\n";
    std::cout << "Repetitions: " << repetitions << "\n\n";

    const std::string circuit_text = circuit_file != nullptr && !std::string(circuit_file).empty()
                                         ? read_file(circuit_file)
                                         : generate_circuit(generated_width, generated_depth);
    clifft::Circuit circuit = clifft::parse(circuit_text);
    clifft::HirModule hir = clifft::trace(circuit);
    auto pass_manager = clifft::default_hir_pass_manager();
    pass_manager.run(hir);
    clifft::sampling::SamplingPlan plan = clifft::sampling::plan_sampling(hir);

    std::cout << "Plan: " << hir.num_qubits << " qubits, peak active width "
              << plan.peak_active_width << ", " << plan.actions.size() << " actions\n\n";
    clifft::sampling::ExecutablePlan program(plan);
    const std::optional<clifft::sampling::ThreadLayout> thread_layout =
        has_layout
            ? std::optional<clifft::sampling::ThreadLayout>{{.shot_workers = static_cast<uint32_t>(
                                                                 shot_workers),
                                                             .intra_shot_workers =
                                                                 static_cast<uint32_t>(
                                                                     intra_shot_workers),
                                                             .intra_shot_min_active_width =
                                                                 static_cast<uint32_t>(
                                                                     intra_shot_min_active_width)}}
            : std::nullopt;

    uint64_t checksum = 0;
    for (int iteration = 0; iteration < warmups; ++iteration) {
        checksum = mix(checksum, checksum_result(clifft::sampling::sample(
                                     program, static_cast<uint32_t>(shots), kSeed + iteration,
                                     static_cast<uint32_t>(threads), thread_layout)));
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(repetitions));
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        auto result = clifft::sampling::sample(program, static_cast<uint32_t>(shots),
                                               kSeed + warmups + iteration,
                                               static_cast<uint32_t>(threads), thread_layout);
        const auto end = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        checksum = mix(checksum, checksum_result(result));
    }

    summarize(samples_ms, shots);
    std::cout << "Checksum: " << checksum << "\n";
    return 0;
}
