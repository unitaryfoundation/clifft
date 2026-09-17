#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include "fault_specialization.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
using namespace clifft;
using namespace clifft::sampling;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Compiled {
    SamplingPlan plan;
    std::unique_ptr<ExecutablePlan> executable;
    double milliseconds;
};

Compiled compile(const Circuit& circuit, bool postselect) {
    const auto start = Clock::now();
    auto hir = trace(circuit);
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    std::vector<uint8_t> mask(circuit.num_detectors, static_cast<uint8_t>(postselect));
    SamplingPlanOptions options;
    options.postselection_mask = mask;
    auto plan = plan_sampling(hir, options);
    auto executable = std::make_unique<ExecutablePlan>(plan);
    return {std::move(plan), std::move(executable), elapsed(start)};
}

class Binding {
    std::vector<std::vector<size_t>> offsets_;
    size_t count_ = 0;

  public:
    Binding(const fault_study::Prepared& prepared, const SamplingPlan& plan) {
        if (prepared.sites.size() != plan.presampled_noise_sites.size()) {
            throw std::invalid_argument("physical noise sites do not match shared-plan sites");
        }
        std::vector<size_t> symbol_offsets(plan.symbols.size());
        for (size_t i = 0; i < plan.symbols.size(); ++i) {
            if (plan.symbols[i] == SymbolKind::Presampled) {
                symbol_offsets[i] = count_++;
            }
        }
        for (size_t i = 0; i < prepared.sites.size(); ++i) {
            const auto& physical = prepared.sites[i];
            const auto& symbolic = plan.presampled_noise_sites[i];
            if (physical.outcomes.size() != symbolic.outcomes.size() ||
                std::abs(physical.probability - symbolic.total_probability) > 1e-14) {
                throw std::invalid_argument("physical and symbolic noise distributions differ");
            }
            std::vector<size_t> offsets;
            for (size_t j = 0; j < physical.outcomes.size(); ++j) {
                if (physical.outcomes[j].probability != symbolic.outcomes[j].probability) {
                    throw std::invalid_argument("noise channel order or probabilities changed");
                }
                offsets.push_back(symbol_offsets[index(symbolic.outcomes[j].symbol)]);
            }
            offsets_.push_back(std::move(offsets));
        }
    }

    std::vector<uint8_t> values(const std::vector<int>& choices) const {
        std::vector<uint8_t> result(count_);
        for (size_t i = 0; i < choices.size(); ++i) {
            if (choices[i] >= 0) {
                result[offsets_[i].at(choices[i])] = 1;
            }
        }
        return result;
    }
};

void write_metrics(const Compiled& compiled, int shots, uint64_t seed,
                   const std::vector<uint8_t>* fixed = nullptr) {
    const auto& plan = compiled.plan;
    long double visits = 0;
    for (const auto& action : plan.actions) {
        visits += std::ldexp(static_cast<long double>(predicted_dense_passes(action.action)),
                             std::max(action.active_before, action.active_after));
    }
    std::cout << "{\"compile_ms\":" << compiled.milliseconds
              << ",\"peak_width\":" << plan.peak_active_width
              << ",\"semantic_actions\":" << plan.actions.size()
              << ",\"executable_actions\":" << compiled.executable->num_actions()
              << ",\"symbols\":" << plan.symbols.size() << ",\"unfused_visits\":" << visits;
    if (plan.peak_active_width > 22) {
        std::cout << ",\"shot_ms\":null}";
        return;
    }
    auto start = Clock::now();
    Executor executor(*compiled.executable, seed);
    const auto init_ms = elapsed(start);
    auto run = [&] {
        if (fixed) {
            executor.run_shot(*fixed);
        } else {
            executor.run_shot();
        }
    };
    start = Clock::now();
    run();
    const auto first_ms = elapsed(start);
    int discarded = 0;
    std::vector<uint32_t> ones(plan.num_observables);
    start = Clock::now();
    for (int i = 0; i < shots; ++i) {
        run();
        discarded += executor.discarded();
        if (!executor.discarded()) {
            for (size_t j = 0; j < ones.size(); ++j) {
                ones[j] += executor.observables()[j];
            }
        }
    }
    const auto shot_ms = elapsed(start) / shots;
    std::cout << ",\"init_ms\":" << init_ms << ",\"first_shot_ms\":" << first_ms
              << ",\"shot_ms\":" << shot_ms << ",\"shots\":" << shots
              << ",\"discarded\":" << discarded << ",\"survivor_observable_ones\":[";
    for (size_t j = 0; j < ones.size(); ++j) {
        std::cout << (j ? "," : "") << ones[j];
    }
    std::cout << "]}";
}

std::vector<double> record_distribution(const Compiled& compiled,
                                        std::span<const uint8_t> presampled = {}) {
    const auto& plan = compiled.plan;
    const uint32_t total = plan.num_visible_records + plan.num_hidden_records;
    if (total > 12 || compiled.executable->has_readout_noise() || plan.peak_active_width > 14) {
        throw std::invalid_argument(
            "exact oracle requires <=12 records, width <=14 and no readout");
    }
    Executor executor(*compiled.executable, 0);
    std::vector<double> probabilities(size_t{1} << plan.num_visible_records);
    std::vector<uint8_t> records(total);
    for (uint32_t bits = 0; bits < (1u << total); ++bits) {
        for (uint32_t j = 0; j < total; ++j) {
            records[j] = (bits >> j) & 1;
        }
        const auto result = executor.replay_shot(records, presampled);
        if (result.reachable) {
            probabilities[bits & (probabilities.size() - 1)] += std::exp(result.log_probability);
        }
    }
    return probabilities;
}

void save_circuit(const Circuit& circuit, const std::filesystem::path& path) {
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << std::setprecision(17);
    uint32_t records = 0;
    for (const auto& node : circuit.nodes) {
        output << gate_name(node.gate);
        if (!node.args.empty()) {
            output << '(';
            for (size_t j = 0; j < node.args.size(); ++j) {
                output << (j ? "," : "") << node.args[j];
            }
            output << ')';
        }
        const bool product = node.gate == GateType::MPP || node.gate == GateType::SPP ||
                             node.gate == GateType::SPP_DAG || node.gate == GateType::TPP ||
                             node.gate == GateType::TPP_DAG || node.gate == GateType::R_PAULI ||
                             node.gate == GateType::EXP_VAL;
        for (size_t j = 0; j < node.targets.size(); ++j) {
            const auto target = node.targets[j];
            output << (product && j ? '*' : ' ');
            if (target.is_inverted()) {
                output << '!';
            }
            if (target.is_rec()) {
                output << "rec[-" << records - target.value() << ']';
            } else {
                if (target.has_pauli()) {
                    output << target.pauli_char();
                }
                output << target.value();
            }
        }
        output << '\n';
        if (is_measurement(node.gate)) {
            records += node.gate == GateType::MPP ? 1 : node.targets.size();
        }
    }
}

void measure_latency(const std::string& strategy, const std::string& text, int shots) {
    if ((strategy != "shared" && strategy != "specialized") || shots < 1 || shots > 4096) {
        throw std::invalid_argument("latency requires shared|specialized and 1..4096 shots");
    }
    int discarded = 0;
    uint64_t observable_ones = 0;
    auto collect = [&](const Executor& executor) {
        discarded += executor.discarded();
        if (!executor.discarded() && !executor.observables().empty()) {
            observable_ones += executor.observables()[0];
        }
    };
    const auto start = Clock::now();
    {
        const auto circuit = parse(text);
        if (strategy == "shared") {
            const auto compiled = compile(circuit, true);
            if (compiled.plan.peak_active_width > 22) {
                throw std::invalid_argument("latency sampling requires width <=22");
            }
            Executor executor(*compiled.executable, 42);
            for (int i = 0; i < shots; ++i) {
                executor.run_shot();
                collect(executor);
            }
        } else {
            const fault_study::Prepared prepared(circuit);
            Xoshiro256PlusPlus rng(42);
            for (int i = 0; i < shots; ++i) {
                const auto specialized = prepared.materialize(prepared.draw(rng));
                const auto compiled = compile(specialized, true);
                if (compiled.plan.peak_active_width > 22) {
                    throw std::invalid_argument("latency sampling requires width <=22");
                }
                Executor executor(*compiled.executable, 1000 + i);
                executor.run_shot();
                collect(executor);
            }
        }
    }
    const auto milliseconds = elapsed(start);
    std::cout << std::setprecision(17) << "{\"strategy\":\"" << strategy << "\",\"shots\":" << shots
              << ",\"discarded\":" << discarded
              << ",\"survivor_observable_zero_ones\":" << observable_ones
              << ",\"task_ms\":" << milliseconds << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--latency") {
            std::ifstream input(argv[3]);
            if (!input || std::string(argv[5]) != "--postselect-all") {
                throw std::invalid_argument(
                    "usage: --latency shared|specialized CIRCUIT SHOTS --postselect-all");
            }
            std::ostringstream text;
            text << input.rdbuf();
            measure_latency(argv[2], text.str(), std::stoi(argv[4]));
            return 0;
        }
        if (argc < 2 || argc > 7) {
            throw std::invalid_argument(
                "usage: profile_fault_specialization CIRCUIT [PATTERNS=16] [SHOTS=128] "
                "[K=-1] [OUTPUT_DIR] [POSTSELECT=1]");
        }
        const int patterns = argc > 2 ? std::stoi(argv[2]) : 16;
        const int shots = argc > 3 ? std::stoi(argv[3]) : 128;
        const int k = argc > 4 ? std::stoi(argv[4]) : -1;
        const int postselect = argc > 6 ? std::stoi(argv[6]) : 1;
        if (patterns < 1 || shots < 1 || static_cast<int64_t>(patterns) * shots > 10000000 ||
            k < -1 || (postselect != 0 && postselect != 1)) {
            throw std::invalid_argument(
                "invalid patterns, shots, quantum fault count, or postselection");
        }
        std::ifstream input(argv[1]);
        if (!input) {
            throw std::invalid_argument("cannot open input circuit");
        }
        std::ostringstream contents;
        contents << input.rdbuf();
        auto start = Clock::now();
        const auto circuit = parse(contents.str());
        const auto parse_ms = elapsed(start);
        start = Clock::now();
        const fault_study::Prepared prepared(circuit);
        const auto common_ms = elapsed(start);
        start = Clock::now();
        std::unique_ptr<KFaultSampler> fixed;
        if (k >= 0) {
            fixed = std::make_unique<KFaultSampler>(prepared.probabilities(), k);
        }
        const auto stratum_ms = elapsed(start);
        const auto baseline = compile(circuit, postselect);
        const Binding binding(prepared, baseline.plan);
        std::cout << std::setprecision(17)
                  << "{\"schema\":1,\"quantum_sites\":" << prepared.sites.size()
                  << ",\"parse_ms\":" << parse_ms << ",\"common_ms\":" << common_ms
                  << ",\"stratum_prepare_ms\":" << stratum_ms << ",\"quantum_k\":" << k
                  << ",\"postselect\":" << postselect << ",\"seed\":42,\"baseline\":";
        write_metrics(baseline, shots * patterns, 42);
        if (argc > 5) {
            std::filesystem::create_directories(argv[5]);
        }
        Xoshiro256PlusPlus rng(42);
        std::cout << ",\"patterns\":[";
        for (int i = 0; i < patterns; ++i) {
            start = Clock::now();
            const auto choices = prepared.draw(rng, fixed.get());
            const auto draw_ms = elapsed(start);
            start = Clock::now();
            const auto specialized = prepared.materialize(choices);
            const auto materialize_ms = elapsed(start);
            const auto compiled = compile(specialized, postselect);
            const auto bound = binding.values(choices);
            const auto faults =
                std::count_if(choices.begin(), choices.end(), [](int c) { return c >= 0; });
            std::cout << (i ? "," : "") << "{\"quantum_faults\":" << faults
                      << ",\"draw_ms\":" << draw_ms << ",\"materialize_ms\":" << materialize_ms
                      << ",\"metrics\":";
            write_metrics(compiled, shots, 1000 + i);
            std::cout << ",\"shared_conditioned\":";
            write_metrics(baseline, shots, 1000 + i, &bound);
            if (compiled.plan.num_visible_records + compiled.plan.num_hidden_records <= 12 &&
                !compiled.executable->has_readout_noise() &&
                compiled.plan.peak_active_width <= 14 && baseline.plan.peak_active_width <= 14) {
                auto unselected = compile(specialized, false);
                const auto reference = compile(circuit, false);
                const Binding oracle_binding(prepared, reference.plan);
                const auto actual = record_distribution(unselected);
                const auto expected =
                    record_distribution(reference, oracle_binding.values(choices));
                for (size_t j = 0; j < actual.size(); ++j) {
                    if (std::abs(actual[j] - expected[j]) > 1e-10) {
                        throw std::runtime_error("conditioned shared-plan oracle disagrees");
                    }
                }
                std::cout << ",\"record_probabilities\":[";
                for (size_t j = 0; j < actual.size(); ++j) {
                    std::cout << (j ? "," : "") << actual[j];
                }
                std::cout << ']';
            }
            if (argc > 5) {
                save_circuit(specialized,
                             std::filesystem::path(argv[5]) / (std::to_string(i) + ".stim"));
            }
            std::cout << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
