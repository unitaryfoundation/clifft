// Export a completed prefix trajectory for offline active-state diagnostics.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace clifft;
    using namespace clifft::sampling;
    if (argc != 4)
        throw std::invalid_argument("usage: snapshot_cultivation circuit.stim state.bin seed");
    std::ifstream input(argv[1]);
    if (!input)
        throw std::invalid_argument("cannot read circuit");
    std::ostringstream text;
    text << input.rdbuf();
    auto hir = trace(parse(text.str()));
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    auto plan = plan_sampling(hir);
    ExecutablePlan executable(plan);
    Executor executor(executable, std::stoull(argv[3]));
    executor.run_shot();
    const auto& state = executor.state();
    std::ofstream output(argv[2], std::ios::binary);
    // Native doubles, real array followed by imaginary array. This export runs
    // after dispatch; it does not add inspection or allocation to hot execution.
    output.write(reinterpret_cast<const char*>(state.real_data()),
                 static_cast<std::streamsize>(state.size() * sizeof(double)));
    output.write(reinterpret_cast<const char*>(state.imag_data()),
                 static_cast<std::streamsize>(state.size() * sizeof(double)));
    if (!output)
        throw std::runtime_error("cannot write state");
    std::cout << "{\"active_width\":" << state.active_width()
              << ",\"peak_active_width\":" << plan.peak_active_width << ",\"records\":[";
    bool first = true;
    for (auto bit : executor.visible_records()) {
        if (!first)
            std::cout << ',';
        std::cout << unsigned(bit);
        first = false;
    }
    std::cout << "]}\n";
}
