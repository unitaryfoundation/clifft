// Native matched-backend harness for perf attribution.

#include "clifft/api/reference_syndrome.h"
#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/sample.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/svm/svm.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

enum class Backend { Legacy, Symbolic };
enum class Mode { Raw, Survivors, ImportanceK0, Noncomputational, Compile };

void set_perf_events_enabled(bool enabled) {
#ifdef __linux__
    static_cast<void>(prctl(enabled ? PR_TASK_PERF_EVENTS_ENABLE : PR_TASK_PERF_EVENTS_DISABLE));
#else
    static_cast<void>(enabled);
#endif
}

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("cannot open circuit: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

Backend parse_backend(std::string_view value) {
    if (value == "legacy") {
        return Backend::Legacy;
    }
    if (value == "symbolic") {
        return Backend::Symbolic;
    }
    throw std::invalid_argument("backend must be legacy or symbolic");
}

Mode parse_mode(std::string_view value) {
    if (value == "raw") {
        return Mode::Raw;
    }
    if (value == "survivors") {
        return Mode::Survivors;
    }
    if (value == "importance-k0") {
        return Mode::ImportanceK0;
    }
    if (value == "noncomp") {
        return Mode::Noncomputational;
    }
    if (value == "compile") {
        return Mode::Compile;
    }
    throw std::invalid_argument("mode must be raw, survivors, importance-k0, noncomp, or compile");
}

clifft::HirModule optimized_hir(const std::string& text) {
    clifft::HirModule hir = clifft::trace(clifft::parse(text));
    auto passes = clifft::default_hir_pass_manager();
    passes.run(hir);
    return hir;
}

clifft::CompiledModule compile_legacy(const std::string& text, bool postselect_all = true) {
    clifft::HirModule hir = optimized_hir(text);
    clifft::ReferenceSyndrome reference = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> postselection(postselect_all ? hir.num_detectors : 0, 1);
    clifft::CompiledModule program =
        clifft::lower(std::move(hir), postselection, reference.detectors, reference.observables);
    auto passes = clifft::default_bytecode_pass_manager();
    passes.run(program);
    return program;
}

clifft::sampling::ExecutablePlan compile_symbolic(const std::string& text,
                                                  bool postselect_all = true) {
    clifft::HirModule hir = optimized_hir(text);
    clifft::ReferenceSyndrome reference = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> postselection(postselect_all ? hir.num_detectors : 0, 1);
    return clifft::sampling::ExecutablePlan(clifft::sampling::plan_sampling(
        hir, {postselection, reference.detectors, reference.observables}));
}

clifft::NonComputationalModel low_leak_model() {
    std::vector<std::vector<double>> transition(5, std::vector<double>(5, 0.0));
    transition[static_cast<size_t>(clifft::Level::LeakE)][static_cast<size_t>(clifft::Level::E)] =
        0.008;
    transition[static_cast<size_t>(clifft::Level::Lost)][static_cast<size_t>(clifft::Level::E)] =
        0.002;

    std::vector<std::vector<double>> classifier(3, std::vector<double>(5, 0.0));
    classifier[0][static_cast<size_t>(clifft::Level::G)] = 1.0;
    classifier[0][static_cast<size_t>(clifft::Level::LeakG)] = 1.0;
    classifier[1][static_cast<size_t>(clifft::Level::E)] = 1.0;
    classifier[1][static_cast<size_t>(clifft::Level::LeakE)] = 1.0;
    classifier[2][static_cast<size_t>(clifft::Level::Lost)] = 1.0;

    return clifft::NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0}, {{"S", transition}},
                                                    classifier, {});
}

uint64_t run_legacy(const clifft::CompiledModule& program, Mode mode, uint32_t shots,
                    uint64_t seed) {
    if (mode == Mode::Raw) {
        clifft::SampleResult result = clifft::sample(program, shots, seed);
        return result.measurements.size() + result.detectors.size() + result.observables.size() +
               result.exp_vals.size();
    }
    clifft::SurvivorResult result = mode == Mode::ImportanceK0
                                        ? clifft::sample_k_survivors(program, shots, 0, seed, false)
                                        : clifft::sample_survivors(program, shots, seed, false);
    return result.passed_shots + result.logical_errors + result.observable_ones.size();
}

uint64_t run_symbolic(const clifft::sampling::ExecutablePlan& program, Mode mode, uint32_t shots,
                      uint64_t seed) {
    if (mode == Mode::Raw) {
        clifft::sampling::SamplingResult result = clifft::sampling::sample(program, shots, seed);
        return result.measurements.size() + result.detectors.size() + result.observables.size() +
               result.exp_vals.size();
    }
    clifft::sampling::SamplingSurvivorResult result =
        mode == Mode::ImportanceK0
            ? clifft::sampling::sample_k_survivors(program, shots, 0, seed, false)
            : clifft::sampling::sample_survivors(program, shots, seed, false);
    return result.passed_shots + result.logical_errors + result.observable_ones.size();
}

uint64_t run_noncomp(const clifft::Circuit& circuit, const clifft::NonComputationalModel& model,
                     Backend backend, uint32_t shots, uint64_t seed) {
    clifft::NonComputationalSample result =
        backend == Backend::Legacy
            ? clifft::sample_noncomputational(circuit, model, shots, seed)
            : clifft::sample_noncomputational_experimental(circuit, model, shots, seed);
    return result.measurements.size() + result.detectors.size() + result.observables.size() +
           result.final_status.size() + result.heralds.size();
}

}  // namespace

int main(int argc, char** argv) {
    set_perf_events_enabled(false);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: profile_backend_comparison CIRCUIT BACKEND MODE SHOTS "
                     "[ITERATIONS]\n";
        return 2;
    }

    try {
        const std::string text = read_file(argv[1]);
        const Backend backend = parse_backend(argv[2]);
        const Mode mode = parse_mode(argv[3]);
        const uint32_t shots = static_cast<uint32_t>(std::stoul(argv[4]));
        const uint32_t iterations = argc == 6 ? static_cast<uint32_t>(std::stoul(argv[5])) : 1;
        if (iterations == 0) {
            throw std::invalid_argument("iterations must be positive");
        }

        uint64_t checksum = 0;
        double seconds = 0.0;

        if (mode == Mode::Compile) {
            set_perf_events_enabled(true);
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                if (backend == Backend::Legacy) {
                    checksum += compile_legacy(text).bytecode.size();
                } else {
                    checksum += compile_symbolic(text).num_actions();
                }
            }
            seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            set_perf_events_enabled(false);
        } else if (mode == Mode::Noncomputational) {
            const clifft::Circuit circuit = clifft::parse(text);
            const clifft::NonComputationalModel model = low_leak_model();
            checksum += run_noncomp(circuit, model, backend, 2, 17);
            set_perf_events_enabled(true);
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                checksum += run_noncomp(circuit, model, backend, shots, 280 + i);
            }
            seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            set_perf_events_enabled(false);
        } else if (backend == Backend::Legacy) {
            const clifft::CompiledModule program = compile_legacy(text, mode != Mode::Raw);
            checksum += run_legacy(program, mode, 2, 17);
            set_perf_events_enabled(true);
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                checksum += run_legacy(program, mode, shots, 280 + i);
            }
            seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            set_perf_events_enabled(false);
        } else {
            const clifft::sampling::ExecutablePlan program =
                compile_symbolic(text, mode != Mode::Raw);
            checksum += run_symbolic(program, mode, 2, 17);
            set_perf_events_enabled(true);
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < iterations; ++i) {
                checksum += run_symbolic(program, mode, shots, 280 + i);
            }
            seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            set_perf_events_enabled(false);
        }

        std::cout << "backend=" << argv[2] << " mode=" << argv[3] << " shots=" << shots
                  << " iterations=" << iterations << " seconds=" << seconds
                  << " checksum=" << checksum << "\n";
    } catch (const std::exception& error) {
        std::cerr << "profile_backend_comparison: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
