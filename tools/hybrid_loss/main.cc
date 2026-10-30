#include "clifft/circuit/parser.h"
#include "clifft/noncomp/program.h"

#include "hybrid.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace clifft;
using Clock = std::chrono::steady_clock;

NonComputationalModel load_model(const std::string& path) {
    std::ifstream in(path);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    std::vector<double> initial(5);
    for (double& p : initial)
        in >> p;
    bool restore;
    unsigned damping;
    size_t count;
    in >> restore >> damping >> count;
    std::map<std::string, std::vector<std::vector<double>>> transitions;
    for (size_t i = 0; i < count; ++i) {
        std::string name;
        in >> std::quoted(name);
        auto& matrix = transitions[name];
        matrix.assign(5, std::vector<double>(5));
        for (auto& row : matrix)
            for (double& p : row)
                in >> p;
    }
    in >> count;
    std::optional<std::vector<std::vector<double>>> classifier;
    if (count) {
        classifier.emplace(count, std::vector<double>(5));
        for (auto& row : *classifier)
            for (double& p : row)
                in >> p;
    }
    GatePartnerEffects effects;
    for (auto* dest : {&effects.leaked, &effects.lost}) {
        bool present;
        in >> present;
        if (present) {
            dest->emplace();
            for (double& p : dest->value().pauli)
                in >> p;
            in >> dest->value().spread_probability;
        }
    }
    in >> count;
    std::vector<InteractionRule> rules(count);
    for (auto& r : rules) {
        unsigned status;
        in >> std::quoted(r.gate) >> r.source_operand >> status;
        r.source_status = status ? InteractionSource::Lost : InteractionSource::Leaked;
        for (double& p : r.effect.pauli)
            in >> p;
        in >> r.effect.spread_probability;
    }
    return NonComputationalModel::from_spec(
        initial, transitions, classifier,
        {.reset_restores_lost = restore,
         .damping = damping ? DampingPolicy::Neglect : DampingPolicy::Exact},
        effects, rules);
}

template <class T>
void save(const std::string& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2)
            args[argv[i]] = argv[i + 1];
        auto start = Clock::now();
        const auto circuit = parse_file(args.at("--circuit"));
        const auto model = load_model(args.at("--model"));
        const auto shots = static_cast<uint32_t>(std::stoul(args.at("--shots")));
        const auto seed = args.contains("--seed") ? std::stoull(args.at("--seed")) : 42;
        const auto capacity = args.contains("--capacity") ? std::stoul(args.at("--capacity")) : 16;
        const bool final_status =
            args.contains("--final-status") && args.at("--final-status") == "1";
        const auto request =
            PreparedNonComputationalRequest(circuit, model, {.final_status = final_status});
        const double preparation = std::chrono::duration<double>(Clock::now() - start).count();
        start = Clock::now();
        NonComputationalSample output;
        if (args.at("--mode") == "hybrid")
            output = experimental::sample_hybrid(request, shots, seed, capacity);
        else if (args.at("--mode") == "trajectory")
            output = request.sample(shots, seed, 1, true);
        else
            throw std::invalid_argument("mode must be hybrid or trajectory");
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        long peak_rss_kib = 0;
#if defined(__linux__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
            peak_rss_kib = usage.ru_maxrss;
#endif
        if (args.contains("--output")) {
            const auto& prefix = args.at("--output");
            save(prefix + ".measurements", output.measurements);
            save(prefix + ".heralds", output.heralds);
            save(prefix + ".detectors", output.detectors);
            save(prefix + ".observables", output.observables);
            save(prefix + ".status", output.final_status);
        }
        std::cout << std::setprecision(12) << "{\"mode\":\"" << args.at("--mode")
                  << "\",\"shots\":" << shots << ",\"preparation_seconds\":" << preparation
                  << ",\"sampling_seconds\":" << elapsed
                  << ",\"us_per_shot\":" << elapsed * 1e6 / shots
                  << ",\"qubits\":" << output.num_qubits
                  << ",\"measurements\":" << output.num_measurements
                  << ",\"detectors\":" << output.num_detectors
                  << ",\"observables\":" << output.num_observables
                  << ",\"peak_width\":" << *output.stats->peak_active_width
                  << ",\"continuations\":" << output.stats->continuation_compilations
                  << ",\"peak_rss_kib\":" << peak_rss_kib << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
