// Offline elementary-gate oracle for explicitly exposed reset trajectories.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

double prefix_log_probability(const std::string& text, std::span<const uint8_t> records) {
    auto hir = clifft::trace(clifft::parse(text));
    auto passes = clifft::default_hir_pass_manager();
    passes.run(hir);
    auto plan = clifft::sampling::plan_sampling(hir);
    if (plan.peak_active_width > 20 || plan.num_hidden_records ||
        plan.num_visible_records != records.size())
        throw std::invalid_argument("prefix exceeds oracle contract");
    clifft::sampling::ExecutablePlan executable(plan);
    clifft::sampling::Executor executor(executable);
    auto result = executor.replay_shot(records);
    if (!result.reachable)
        throw std::runtime_error("sampled prefix is unreachable");
    return result.log_probability;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4 && argc != 5)
            throw std::invalid_argument(
                "usage: msc_record_probe CIRCUIT SEED [--prefixes | --replay BITS]");
        if (argc == 4 && std::string(argv[3]) != "--prefixes")
            throw std::invalid_argument("unknown option");
        if (argc == 5 && std::string(argv[3]) != "--replay")
            throw std::invalid_argument("unknown replay option");
        std::ifstream input(argv[1]);
        if (!input)
            throw std::invalid_argument("cannot open input");
        std::ostringstream text;
        text << input.rdbuf();
        auto hir = clifft::trace(clifft::parse(text.str()));
        auto passes = clifft::default_hir_pass_manager();
        passes.run(hir);
        auto plan = clifft::sampling::plan_sampling(hir);
        if (plan.peak_active_width > 20 || plan.num_hidden_records)
            throw std::invalid_argument("oracle capacity or exposed-record contract exceeded");
        clifft::sampling::ExecutablePlan executable(plan);
        clifft::sampling::Executor executor(executable, std::stoull(argv[2]));
        std::vector<uint8_t> records;
        if (argc == 5) {
            const std::string bits(argv[4]);
            if (bits.size() != plan.num_visible_records)
                throw std::invalid_argument("replay record count differs");
            for (char b : bits) {
                if (b != '0' && b != '1')
                    throw std::invalid_argument("nonbinary replay record");
                records.push_back(b - '0');
            }
        } else {
            executor.run_shot();
            auto visible = executor.visible_records();
            records.assign(visible.begin(), visible.end());
        }
        auto replay = executor.replay_shot(records);
        if (!replay.reachable)
            throw std::runtime_error("sampled trajectory is unreachable");
        std::cout << std::setprecision(17) << "{\"peak_active_width\":" << plan.peak_active_width
                  << ",\"log_probability\":" << replay.log_probability << ",\"outcomes\":[";
        for (size_t k = 0; k < records.size(); ++k)
            std::cout << (k ? "," : "") << unsigned(records[k]);
        std::cout << ']';
        if (argc == 4) {
            std::cout << ",\"probabilities\":[";
            std::istringstream source(text.str());
            std::string line;
            std::string prefix;
            size_t count = 0;
            double previous = 0;
            while (std::getline(source, line)) {
                prefix += line + '\n';
                if (line.starts_with("M ") || line.starts_with("MX ") || line.starts_with("MPP ")) {
                    ++count;
                    const double current = prefix_log_probability(
                        prefix, std::span<const uint8_t>(records).first(count));
                    std::cout << (count == 1 ? "" : ",") << std::exp(current - previous);
                    previous = current;
                }
            }
            std::cout << ']';
        }
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
