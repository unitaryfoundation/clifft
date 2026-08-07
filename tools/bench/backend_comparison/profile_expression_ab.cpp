// Same-build A/B harness for the scalar affine-expression executor prototype.

#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/fault_sampling.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class Mode { Survivors, ImportanceK0 };

struct RunResult {
    uint64_t checksum = 0;
    uint64_t passed_shots = 0;
    uint64_t logical_errors = 0;
    clifft::sampling::ExpressionExecutionStats stats;
};

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("cannot open circuit: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

clifft::sampling::ExpressionEvaluationMode parse_evaluator(std::string_view value) {
    if (value == "direct") {
        return clifft::sampling::ExpressionEvaluationMode::Direct;
    }
    if (value == "incremental") {
        return clifft::sampling::ExpressionEvaluationMode::Incremental;
    }
    throw std::invalid_argument("evaluator must be direct or incremental");
}

Mode parse_mode(std::string_view value) {
    if (value == "survivors") {
        return Mode::Survivors;
    }
    if (value == "importance-k0") {
        return Mode::ImportanceK0;
    }
    throw std::invalid_argument("mode must be survivors or importance-k0");
}

clifft::sampling::ExecutablePlan compile_symbolic(const std::string& text) {
    clifft::HirModule hir = clifft::trace(clifft::parse(text));
    auto passes = clifft::default_hir_pass_manager();
    passes.run(hir);
    clifft::ReferenceSyndrome reference = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> postselection(hir.num_detectors, 1);
    return clifft::sampling::ExecutablePlan(clifft::sampling::plan_sampling(
        hir, {postselection, reference.detectors, reference.observables}));
}

void add_stats(clifft::sampling::ExpressionExecutionStats& total,
               const clifft::sampling::ExpressionExecutionStats& value) {
    total.shots += value.shots;
    total.discarded_shots += value.discarded_shots;
    total.expression_evaluations += value.expression_evaluations;
    total.direct_term_visits += value.direct_term_visits;
    total.true_symbol_assignments += value.true_symbol_assignments;
    total.weighted_true_fanout += value.weighted_true_fanout;
    total.accumulator_resets += value.accumulator_resets;
    total.propagated_edges += value.propagated_edges;
}

RunResult run(const clifft::sampling::ExecutablePlan& plan,
              clifft::sampling::ExpressionEvaluationMode evaluator, Mode mode, uint32_t shots,
              uint64_t seed, bool collect_census) {
    clifft::sampling::Executor executor(plan, seed, evaluator, collect_census);
    std::optional<clifft::KFaultSampler> fault_sampler;
    if (mode == Mode::ImportanceK0) {
        fault_sampler.emplace(plan.noise_site_probabilities(), 0);
    }

    RunResult result;
    std::vector<uint64_t> observable_ones(plan.num_observables(), 0);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        if (fault_sampler.has_value()) {
            executor.run_shot(*fault_sampler);
        } else {
            executor.run_shot();
        }
        if (executor.discarded()) {
            continue;
        }
        ++result.passed_shots;
        bool logical_error = false;
        for (uint32_t observable = 0; observable < plan.num_observables(); ++observable) {
            const bool value = executor.observables()[observable] != 0;
            observable_ones[observable] += static_cast<uint64_t>(value);
            logical_error |= value;
        }
        result.logical_errors += static_cast<uint64_t>(logical_error);
    }
    result.checksum = result.passed_shots + result.logical_errors;
    for (uint64_t count : observable_ones) {
        result.checksum += count;
    }
    result.stats = executor.expression_stats();
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 8) {
        std::cerr << "usage: profile_expression_ab CIRCUIT EVALUATOR MODE SHOTS "
                     "[ITERATIONS] [--census] [--seed-base=N]\n";
        return 2;
    }

    try {
        const std::string text = read_file(argv[1]);
        const auto evaluator = parse_evaluator(argv[2]);
        const Mode mode = parse_mode(argv[3]);
        const uint32_t shots = static_cast<uint32_t>(std::stoul(argv[4]));
        uint32_t iterations = 1;
        bool collect_census = false;
        uint64_t seed_base = 280;
        for (int i = 5; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--census") {
                collect_census = true;
            } else if (argument.starts_with("--seed-base=")) {
                seed_base = std::stoull(std::string(argument.substr(12)));
            } else {
                iterations = static_cast<uint32_t>(std::stoul(argv[i]));
            }
        }
        if (iterations == 0) {
            throw std::invalid_argument("iterations must be positive");
        }

        const clifft::sampling::ExecutablePlan plan = compile_symbolic(text);
        (void)run(plan, evaluator, mode, 2, 17, false);

        RunResult total;
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < iterations; ++i) {
            RunResult current = run(plan, evaluator, mode, shots, seed_base + i, collect_census);
            total.checksum += current.checksum;
            total.passed_shots += current.passed_shots;
            total.logical_errors += current.logical_errors;
            add_stats(total.stats, current.stats);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        const auto& stats = total.stats;
        std::cout << "{\"evaluator\":\"" << argv[2] << "\",\"mode\":\"" << argv[3]
                  << "\",\"shots\":" << shots << ",\"iterations\":" << iterations
                  << ",\"seed_base\":" << seed_base
                  << ",\"census\":" << (collect_census ? "true" : "false")
                  << ",\"seconds\":" << seconds << ",\"checksum\":" << total.checksum
                  << ",\"passed_shots\":" << total.passed_shots
                  << ",\"logical_errors\":" << total.logical_errors
                  << ",\"expression_stats\":{\"shots\":" << stats.shots
                  << ",\"discarded_shots\":" << stats.discarded_shots
                  << ",\"expression_evaluations\":" << stats.expression_evaluations
                  << ",\"direct_term_visits\":" << stats.direct_term_visits
                  << ",\"true_symbol_assignments\":" << stats.true_symbol_assignments
                  << ",\"weighted_true_fanout\":" << stats.weighted_true_fanout
                  << ",\"accumulator_resets\":" << stats.accumulator_resets
                  << ",\"propagated_edges\":" << stats.propagated_edges << "}}\n";
    } catch (const std::exception& error) {
        std::cerr << "profile_expression_ab: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
