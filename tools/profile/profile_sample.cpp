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
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/fault_sampling.h"
#include "clifft/util/intra_shot_parallel.h"
#include "clifft/util/shot_parallel.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr int kDefaultShots = 1;
constexpr int kDefaultThreads = 1;
constexpr int kDefaultWarmups = 2;
constexpr int kDefaultRepetitions = 20;
constexpr int kDefaultGeneratedDepth = 20;
constexpr uint64_t kSeed = 42;

enum class ProfileApi : uint8_t {
    Sample,
    SampleSurvivors,
    SampleK,
    SampleKSurvivors,
};

struct SamplingSummary {
    double minimum = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

int get_env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::stoi(value);
}

std::string get_env_string(const char* name, std::string default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::move(default_value) : std::string(value);
}

ProfileApi parse_profile_api(std::string_view value) {
    if (value == "sample") {
        return ProfileApi::Sample;
    }
    if (value == "sample_survivors") {
        return ProfileApi::SampleSurvivors;
    }
    if (value == "sample_k") {
        return ProfileApi::SampleK;
    }
    if (value == "sample_k_survivors") {
        return ProfileApi::SampleKSurvivors;
    }
    std::cerr << "Error: API must be sample, sample_survivors, sample_k, or "
                 "sample_k_survivors\n";
    std::exit(1);
}

const char* profile_api_name(ProfileApi api) noexcept {
    switch (api) {
        case ProfileApi::Sample:
            return "sample";
        case ProfileApi::SampleSurvivors:
            return "sample_survivors";
        case ProfileApi::SampleK:
            return "sample_k";
        case ProfileApi::SampleKSurvivors:
            return "sample_k_survivors";
    }
    return "unknown";
}

bool is_survivor_api(ProfileApi api) noexcept {
    return api == ProfileApi::SampleSurvivors || api == ProfileApi::SampleKSurvivors;
}

bool is_fixed_k_api(ProfileApi api) noexcept {
    return api == ProfileApi::SampleK || api == ProfileApi::SampleKSurvivors;
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

uint64_t checksum_result(const clifft::sampling::SamplingSurvivorResult& result) {
    uint64_t checksum = mix(result.total_shots, result.passed_shots);
    checksum = mix(checksum, result.logical_errors);
    for (uint64_t value : result.observable_ones) {
        checksum = mix(checksum, value);
    }
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

using ProfileResult =
    std::variant<clifft::sampling::SamplingResult, clifft::sampling::SamplingSurvivorResult>;

uint64_t checksum_result(const ProfileResult& result) {
    return std::visit([](const auto& typed) { return checksum_result(typed); }, result);
}

SamplingSummary summarize(std::vector<double> samples_ms, int shots) {
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
    return {.minimum = samples_ms.front(),
            .median = median,
            .mean = mean,
            .p95 = p95,
            .maximum = samples_ms.back()};
}

std::vector<uint8_t> make_postselection_mask(std::string_view mode, uint32_t detectors) {
    std::vector<uint8_t> mask(detectors, 0);
    if (mode == "none") {
        return mask;
    }
    if (mode == "all") {
        std::ranges::fill(mask, uint8_t{1});
        return mask;
    }
    if (mode == "first-half") {
        std::ranges::fill(mask.begin(), mask.begin() + (mask.size() + 1) / 2, uint8_t{1});
        return mask;
    }
    if (mode == "last-half") {
        std::ranges::fill(mask.begin() + mask.size() / 2, mask.end(), uint8_t{1});
        return mask;
    }
    if (mode == "alternating") {
        for (size_t detector = 0; detector < mask.size(); detector += 2) {
            mask[detector] = 1;
        }
        return mask;
    }
    std::cerr << "Error: postselection must be none, all, first-half, last-half, or "
                 "alternating\n";
    std::exit(1);
}

clifft::sampling::ThreadLayout resolve_profile_thread_layout(
    const clifft::sampling::ExecutablePlan& plan, uint32_t shots, uint32_t threads,
    std::optional<clifft::sampling::ThreadLayout> override) {
    if (override.has_value()) {
        override->shot_workers = std::min(override->shot_workers, shots);
        if (!clifft::should_parallelize_intra_shot(plan.peak_active_width(),
                                                   override->intra_shot_workers,
                                                   override->intra_shot_min_active_width)) {
            override->intra_shot_workers = 1;
        }
        return *override;
    }
    const uint32_t budget = clifft::resolve_thread_budget(threads);
    if (shots < budget &&
        clifft::should_parallelize_intra_shot(plan.peak_active_width(), budget,
                                              clifft::kDefaultIntraShotMinActiveWidth)) {
        return {.shot_workers = 1,
                .intra_shot_workers = budget,
                .intra_shot_min_active_width = clifft::kDefaultIntraShotMinActiveWidth};
    }
    return {.shot_workers = std::min(shots, budget), .intra_shot_workers = 1};
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
    const char* api_value = std::getenv("CLIFFT_PROFILE_API");
    const bool legacy_aggregate_survivors =
        get_env_int("CLIFFT_PROFILE_AGGREGATE_SURVIVORS", 0) != 0;
    const ProfileApi api = parse_profile_api(
        api_value == nullptr ? (legacy_aggregate_survivors ? "sample_survivors" : "sample")
                             : api_value);
    const bool keep_records = get_env_int("CLIFFT_PROFILE_KEEP_RECORDS", 0) != 0;
    const char* fixed_k_value = std::getenv("CLIFFT_PROFILE_FIXED_K");
    const int fixed_k = get_env_int("CLIFFT_PROFILE_FIXED_K", 1);
    const bool legacy_postselect_all = get_env_int("CLIFFT_PROFILE_POSTSELECT_ALL", 0) != 0;
    const std::string postselection =
        get_env_string("CLIFFT_PROFILE_POSTSELECTION",
                       legacy_postselect_all ? std::string("all") : std::string("none"));
    const char* shot_workers_value = std::getenv("CLIFFT_PROFILE_SHOT_WORKERS");
    const char* intra_workers_value = std::getenv("CLIFFT_PROFILE_INTRA_SHOT_WORKERS");
    const char* min_active_width_value = std::getenv("CLIFFT_PROFILE_INTRA_SHOT_MIN_ACTIVE_WIDTH");
    const char* batch_size_value = std::getenv("CLIFFT_PROFILE_BATCH_SIZE");
    const bool batch_is_auto =
        batch_size_value == nullptr || std::string_view(batch_size_value) == "auto";
    const bool has_layout = shot_workers_value != nullptr || intra_workers_value != nullptr;
    const int shot_workers = get_env_int("CLIFFT_PROFILE_SHOT_WORKERS", 0);
    const int intra_shot_workers = get_env_int("CLIFFT_PROFILE_INTRA_SHOT_WORKERS", 0);
    const int intra_shot_min_active_width =
        get_env_int("CLIFFT_PROFILE_INTRA_SHOT_MIN_ACTIVE_WIDTH",
                    static_cast<int>(clifft::kDefaultIntraShotMinActiveWidth));
    const int batch_size = batch_is_auto ? 0 : std::stoi(batch_size_value);
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
    if (!batch_is_auto && batch_size < 1) {
        std::cerr << "Error: batch size must be a positive integer or auto\n";
        return 1;
    }
    if (!is_survivor_api(api) && postselection != "none") {
        std::cerr << "Error: postselection requires a survivor sampling API\n";
        return 1;
    }
    if (!is_survivor_api(api) && keep_records) {
        std::cerr << "Error: keep_records requires a survivor sampling API\n";
        return 1;
    }
    if (!is_fixed_k_api(api) && fixed_k_value != nullptr) {
        std::cerr << "Error: fixed k requires sample_k or sample_k_survivors\n";
        return 1;
    }
    if (fixed_k < 0) {
        std::cerr << "Error: fixed k must be non-negative\n";
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
    std::cout << "API:         " << profile_api_name(api) << "\n";
    if (is_survivor_api(api)) {
        std::cout << "Keep rows:   " << (keep_records ? "yes" : "no") << "\n";
        std::cout << "Postselect:  " << postselection << "\n";
    }
    if (is_fixed_k_api(api)) {
        std::cout << "Fixed k:     " << fixed_k << "\n";
    }
    if (has_layout) {
        std::cout << "Layout:      " << shot_workers << " shot x " << intra_shot_workers
                  << " intra-shot\n";
        std::cout << "Min width:   " << intra_shot_min_active_width << "\n";
    }
    std::cout << "Warmups:     " << warmups << "\n";
    std::cout << "Repetitions: " << repetitions << "\n";
    std::cout << "Batch req:   "
              << (batch_is_auto ? std::string("auto") : std::to_string(batch_size)) << "\n";
    std::cout << "\n";

    const std::string circuit_text = circuit_file != nullptr && !std::string(circuit_file).empty()
                                         ? read_file(circuit_file)
                                         : generate_circuit(generated_width, generated_depth);
    clifft::Circuit circuit = clifft::parse(circuit_text);
    clifft::HirModule hir = clifft::trace(circuit);
    auto pass_manager = clifft::default_hir_pass_manager();
    pass_manager.run(hir);
    const std::vector<uint8_t> postselection_mask =
        make_postselection_mask(postselection, hir.num_detectors);
    clifft::sampling::SamplingPlanOptions plan_options;
    plan_options.postselection_mask = postselection_mask;
    clifft::sampling::SamplingPlan plan = clifft::sampling::plan_sampling(hir, plan_options);

    std::cout << "Plan: " << hir.num_qubits << " qubits, peak active width "
              << plan.peak_active_width << ", " << plan.actions.size() << " actions\n";
    clifft::sampling::ExecutablePlan program(plan);
    std::cout << "Batch work:  " << program.estimated_batch_lane_work()
              << " estimated coefficient visits/lane\n\n";
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
    const std::optional<uint32_t> requested_batch_size =
        batch_is_auto ? std::nullopt : std::optional<uint32_t>{static_cast<uint32_t>(batch_size)};
    const clifft::sampling::ThreadLayout resolved_layout = resolve_profile_thread_layout(
        program, static_cast<uint32_t>(shots), static_cast<uint32_t>(threads), thread_layout);
    const clifft::sampling::BatchOutputMode output_mode =
        is_survivor_api(api) && !keep_records
            ? clifft::sampling::BatchOutputMode::AggregateSurvivors
            : clifft::sampling::BatchOutputMode::Rows;
    const clifft::sampling::BatchSamplingMode sampling_mode =
        is_fixed_k_api(api) ? clifft::sampling::BatchSamplingMode::FixedFaults
                            : clifft::sampling::BatchSamplingMode::Ordinary;
    uint64_t additional_worker_bytes =
        is_survivor_api(api) ? static_cast<uint64_t>(program.num_observables()) * sizeof(uint64_t)
                             : 0;
    std::unique_ptr<clifft::KFaultDistribution> fault_distribution;
    if (is_fixed_k_api(api)) {
        fault_distribution = std::make_unique<clifft::KFaultDistribution>(
            program.noise_site_probabilities(), static_cast<uint32_t>(fixed_k));
        additional_worker_bytes += fault_distribution->worker_scratch_bytes();
    }
    const clifft::sampling::BatchExecutionPolicy batch_policy =
        clifft::sampling::resolve_batch_execution_policy(
            program, static_cast<uint32_t>(shots), resolved_layout.shot_workers,
            resolved_layout.intra_shot_workers, output_mode, requested_batch_size, sampling_mode,
            additional_worker_bytes);
    const uint32_t effective_workers =
        batch_policy.lane_capacity > 1 ? batch_policy.worker_count : resolved_layout.shot_workers;
    const uint64_t batch_worker_bytes =
        batch_policy.lane_capacity > 1
            ? clifft::sampling::batch_detail::batch_worker_storage_bytes(
                  program, batch_policy.lane_capacity, output_mode, sampling_mode) +
                  additional_worker_bytes
            : 0;

    std::cout << "Policy: " << batch_policy.lane_capacity << " lanes x " << effective_workers
              << " workers; layout " << resolved_layout.shot_workers << " shot x "
              << resolved_layout.intra_shot_workers << " intra-shot";
    if (batch_worker_bytes != 0) {
        std::cout << "; " << batch_worker_bytes << " retained bytes/worker";
    }
    std::cout << "\n\n";

    const auto run_sample = [&](uint64_t sample_seed) -> ProfileResult {
        switch (api) {
            case ProfileApi::Sample:
                return clifft::sampling::sample(program, static_cast<uint32_t>(shots), sample_seed,
                                                static_cast<uint32_t>(threads), thread_layout,
                                                requested_batch_size);
            case ProfileApi::SampleSurvivors:
                return clifft::sampling::sample_survivors(
                    program, static_cast<uint32_t>(shots), sample_seed, keep_records,
                    static_cast<uint32_t>(threads), thread_layout, requested_batch_size);
            case ProfileApi::SampleK:
                return clifft::sampling::sample_k(program, static_cast<uint32_t>(shots),
                                                  static_cast<uint32_t>(fixed_k), sample_seed,
                                                  static_cast<uint32_t>(threads), thread_layout,
                                                  requested_batch_size);
            case ProfileApi::SampleKSurvivors:
                return clifft::sampling::sample_k_survivors(
                    program, static_cast<uint32_t>(shots), static_cast<uint32_t>(fixed_k),
                    sample_seed, keep_records, static_cast<uint32_t>(threads), thread_layout,
                    requested_batch_size);
        }
        std::abort();
    };

    uint64_t checksum = 0;
    for (int iteration = 0; iteration < warmups; ++iteration) {
        const ProfileResult result = run_sample(kSeed + iteration);
        checksum = mix(checksum, checksum_result(result));
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(repetitions));
    uint32_t passed_shots = static_cast<uint32_t>(shots);
    size_t retained_rows = static_cast<size_t>(shots);
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const ProfileResult result = run_sample(kSeed + warmups + iteration);
        const auto end = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        checksum = mix(checksum, checksum_result(result));
        if (const auto* survivor = std::get_if<clifft::sampling::SamplingSurvivorResult>(&result)) {
            passed_shots = survivor->passed_shots;
            retained_rows = keep_records ? survivor->passed_shots : 0;
        }
    }

    const SamplingSummary summary = summarize(samples_ms, shots);
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << std::setprecision(10) << "RESULT" << " api=" << profile_api_name(api)
              << " keep_records=" << static_cast<int>(keep_records)
              << " fixed_k=" << (is_fixed_k_api(api) ? fixed_k : -1)
              << " postselection=" << (is_survivor_api(api) ? postselection : "none")
              << " requested_batch="
              << (batch_is_auto ? std::string("auto") : std::to_string(batch_size))
              << " effective_batch=" << batch_policy.lane_capacity
              << " effective_workers=" << effective_workers
              << " shot_workers=" << resolved_layout.shot_workers
              << " intra_shot_workers=" << resolved_layout.intra_shot_workers
              << " batch_lane_work=" << program.estimated_batch_lane_work()
              << " worker_bytes=" << batch_worker_bytes << " median_ms=" << summary.median
              << " mean_ms=" << summary.mean << " passed_shots=" << passed_shots
              << " survival=" << static_cast<double>(passed_shots) / static_cast<double>(shots)
              << " retained_rows=" << retained_rows << " checksum=" << checksum << "\n";
    return 0;
}
