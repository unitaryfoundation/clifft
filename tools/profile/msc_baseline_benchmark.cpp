// Original-circuit many-shot comparison for the standalone MSC research sampler.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("usage: msc_baseline_benchmark CIRCUIT SEED SHOTS_OR_plan");
        std::ifstream file(argv[1]);
        if (!file)
            throw std::invalid_argument("cannot read circuit");
        std::ostringstream source;
        source << file.rdbuf();
        auto hir = clifft::trace(clifft::parse(source.str()));
        auto passes = clifft::default_hir_pass_manager();
        passes.run(hir);
        auto plan = clifft::sampling::plan_sampling(hir);
        if (std::string(argv[3]) == "plan") {
            std::cout << std::setprecision(17)
                      << "{\"peak_active_width\":" << plan.peak_active_width
                      << ",\"dense_coefficient_bytes\":" << std::ldexp(16., plan.peak_active_width)
                      << "}\n";
            return 0;
        }
        if (plan.peak_active_width > 20)
            throw std::invalid_argument("baseline exceeds memory bound");
        clifft::sampling::ExecutablePlan executable(plan);
        clifft::sampling::Executor executor(executable, std::stoull(argv[2]));
        unsigned shots = std::stoul(argv[3]);
        if (!shots)
            throw std::invalid_argument("zero shots");
        unsigned accepted = 0, checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (unsigned k = 0; k < shots; ++k) {
            executor.run_shot();
            accepted += std::none_of(executor.detectors().begin(), executor.detectors().end(),
                                     [](auto bit) { return bit != 0; });
            for (auto bit : executor.visible_records())
                checksum += bit;
            for (auto bit : executor.observables())
                checksum += bit;
        }
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(17) << "{\"shots\":" << shots << ",\"accepted\":" << accepted
                  << ",\"seconds\":" << seconds
                  << ",\"microseconds_per_shot\":" << seconds * 1e6 / shots
                  << ",\"peak_active_width\":" << plan.peak_active_width
                  << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
