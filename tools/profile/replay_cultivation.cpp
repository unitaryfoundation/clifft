// Conditional physical-history reference for offline gadget validation.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include "study_schedule.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace clifft;
    using namespace clifft::sampling;
    if (argc < 3 || argc > 5)
        throw std::invalid_argument(
            "usage: replay_cultivation circuit.stim seed [records.txt [off|budgeted|unbounded]]");
    std::ifstream input(argv[1]);
    if (!input)
        throw std::invalid_argument("cannot read circuit");
    std::ostringstream text;
    text << input.rdbuf();
    auto hir = trace(parse(text.str()));
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    apply_study_schedule(hir, argc == 5 ? argv[4] : "off");
    auto plan = plan_sampling(hir);
    for (const auto& planned : plan.actions)
        if (std::holds_alternative<ApplyReadoutNoise>(planned.action))
            throw std::invalid_argument("use inverted targets for fixed readout flips in replay");
    ExecutablePlan executable(plan);
    Executor executor(executable, std::stoull(argv[2]));
    std::vector<uint8_t> records;
    if (argc >= 4) {
        std::ifstream forced(argv[3]);
        if (!forced)
            throw std::invalid_argument("cannot read forced records");
        char bit;
        while (forced >> bit) {
            if (bit != '0' && bit != '1')
                throw std::invalid_argument("records must be bits");
            records.push_back(bit - '0');
        }
        if (records.size() != plan.num_visible_records + plan.num_hidden_records)
            throw std::invalid_argument("wrong record count");
    } else {
        executor.run_shot();
        records.assign(executor.visible_records().begin(), executor.visible_records().end());
        records.insert(records.end(), executor.hidden_records().begin(),
                       executor.hidden_records().end());
    }
    const auto replay = executor.replay_shot(records);
    std::cout << std::setprecision(17) << "{\"reachable\":" << (replay.reachable ? "true" : "false")
              << ",\"log_probability\":" << replay.log_probability
              << ",\"visible\":" << plan.num_visible_records
              << ",\"hidden\":" << plan.num_hidden_records
              << ",\"peak_active_width\":" << plan.peak_active_width << ",\"records\":\"";
    for (auto bit : records)
        std::cout << unsigned(bit);
    std::cout << "\",\"expectation_values\":[";
    bool first = true;
    for (auto value : executor.exp_vals()) {
        std::cout << (first ? "" : ",") << value;
        first = false;
    }
    std::cout << "]}\n";
}
