// Command-line driver for the exact active-width certificate search.
//
// Loads a .stim circuit, traces and optimizes it exactly as the library
// pipeline would, then runs the budgeted exact search and prints the
// resulting bounds. See README.md for what "certificate" means here and
// active_width_search.h for the algorithm and its certificate scope.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/schedule_dependence.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "active_width_search.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace clifft;
using namespace clifft::research;

namespace {

void print_usage(std::ostream& out) {
    out << "usage: width_certificate <circuit.stim> "
           "[--pipeline none|peephole|production] [--no-noise-transparency] "
           "[--budget N] [--print-order]\n";
}

struct Args {
    std::string circuit_path;
    std::string pipeline = "production";
    bool noise_transparent = true;
    uint64_t budget = 200000;
    bool print_order = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pipeline") {
            if (i + 1 >= argc) {
                std::cerr << "--pipeline requires a value\n";
                print_usage(std::cerr);
                std::exit(1);
            }
            args.pipeline = argv[++i];
        } else if (arg == "--no-noise-transparency") {
            args.noise_transparent = false;
        } else if (arg == "--budget") {
            if (i + 1 >= argc) {
                std::cerr << "--budget requires a value\n";
                print_usage(std::cerr);
                std::exit(1);
            }
            args.budget = std::stoull(argv[++i]);
        } else if (arg == "--print-order") {
            args.print_order = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(std::cout);
            std::exit(0);
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            print_usage(std::cerr);
            std::exit(1);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 1) {
        print_usage(std::cerr);
        std::exit(1);
    }
    args.circuit_path = positional.front();

    if (args.pipeline != "none" && args.pipeline != "peephole" && args.pipeline != "production") {
        std::cerr << "unknown --pipeline value: " << args.pipeline
                  << " (expected none, peephole, or production)\n";
        std::exit(1);
    }

    return args;
}

// The three pipelines the driver understands, named the same way the
// library's own default-enabled passes would run them (PeepholeFusionPass,
// then StatevectorSqueezePass), rather than going through
// default_hir_pass_manager(): this driver wants a fixed definition of
// "production" it controls, not whatever the pass registry currently
// defaults to.
void apply_pipeline(HirModule& hir, const std::string& pipeline) {
    if (pipeline == "none") {
        return;
    }
    PeepholeFusionPass{}.run(hir);
    if (pipeline == "production") {
        StatevectorSqueezePass{}.run(hir);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    const Circuit circuit = parse_file(args.circuit_path);
    HirModule hir = trace(circuit);
    apply_pipeline(hir, args.pipeline);

    ScheduleDependenceOptions dependence_options;
    dependence_options.noise_transparent = args.noise_transparent;
    const ScheduleDependence dependence = ScheduleDependence::build(hir, dependence_options);

    WidthSearchOptions search_options;
    search_options.node_budget = args.budget;
    const WidthSearchResult result = search_width_schedule(hir, dependence, search_options);

    std::cout << "circuit:         " << args.circuit_path << "\n";
    std::cout << "pipeline:        " << args.pipeline << "\n";
    std::cout << "relation:        can_swap"
              << (result.noise_transparent ? " + noise-transparent" : "") << "\n";
    std::cout << "incumbent peak:  " << result.incumbent_peak << "\n";
    std::cout << "upper bound:     " << result.upper_bound << "\n";
    std::cout << "lower bound:     " << result.lower_bound << "\n";
    std::cout << "optimal:         " << (result.optimal() ? "yes" : "no") << "\n";
    std::cout << "explored nodes:  " << result.explored_nodes << "\n";
    std::cout << "budget exhausted:" << (result.budget_exhausted ? " yes" : " no") << "\n";

    if (args.print_order) {
        std::cout << "witness order:  ";
        for (uint32_t op : result.best_order) {
            std::cout << ' ' << op;
        }
        std::cout << "\n";
    }

    return 0;
}
