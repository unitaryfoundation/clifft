// Static SamplingPlan metadata extractor for backend-comparison measurements.

#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("cannot open circuit: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string_view action_name(const clifft::sampling::SamplingAction& action) {
    return std::visit(
        [](const auto& typed) -> std::string_view {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, clifft::sampling::RotateActivePauli>) {
                return "rotate_active";
            } else if constexpr (std::is_same_v<T, clifft::sampling::PromoteDormantRotation>) {
                return "promote_dormant";
            } else if constexpr (std::is_same_v<T, clifft::sampling::MeasureActivePauli>) {
                return "measure_active";
            } else if constexpr (std::is_same_v<T, clifft::sampling::MeasureDormantRandom>) {
                return "measure_dormant";
            } else if constexpr (std::is_same_v<T, clifft::sampling::RecordClassical>) {
                return "record_classical";
            } else if constexpr (std::is_same_v<T, clifft::sampling::DefineSymbol>) {
                return "define_symbol";
            } else if constexpr (std::is_same_v<T, clifft::sampling::ApplyReadoutNoise>) {
                return "readout_noise";
            } else if constexpr (std::is_same_v<T, clifft::sampling::WriteDetector>) {
                return "write_detector";
            } else if constexpr (std::is_same_v<T, clifft::sampling::WriteObservable>) {
                return "write_observable";
            } else if constexpr (std::is_same_v<T, clifft::sampling::WriteExpectationValue>) {
                return "write_exp_val";
            } else if constexpr (std::is_same_v<T, clifft::sampling::ApplyInstrument>) {
                return "apply_instrument";
            } else if constexpr (std::is_same_v<T, clifft::sampling::InstrumentBoundary>) {
                return "instrument_boundary";
            }
        },
        action);
}

const clifft::sampling::AffineBool* action_expression(
    const clifft::sampling::SamplingAction& action) {
    return std::visit(
        [](const auto& typed) -> const clifft::sampling::AffineBool* {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, clifft::sampling::RotateActivePauli> ||
                          std::is_same_v<T, clifft::sampling::PromoteDormantRotation>) {
                return &typed.sign;
            } else if constexpr (std::is_same_v<T, clifft::sampling::MeasureActivePauli> ||
                                 std::is_same_v<T, clifft::sampling::MeasureDormantRandom> ||
                                 std::is_same_v<T, clifft::sampling::RecordClassical> ||
                                 std::is_same_v<T, clifft::sampling::WriteDetector> ||
                                 std::is_same_v<T, clifft::sampling::WriteObservable>) {
                return &typed.outcome;
            } else if constexpr (std::is_same_v<T, clifft::sampling::DefineSymbol>) {
                return &typed.value;
            } else if constexpr (std::is_same_v<T, clifft::sampling::ApplyReadoutNoise>) {
                return &typed.source;
            } else if constexpr (std::is_same_v<T, clifft::sampling::WriteExpectationValue> ||
                                 std::is_same_v<T, clifft::sampling::ApplyInstrument>) {
                return &typed.sign;
            } else {
                return nullptr;
            }
        },
        action);
}

void print_u32_vector(const std::vector<uint32_t>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: profile_sampling_plan CIRCUIT [--postselect-all]\n";
        return 2;
    }
    const bool postselect_all = argc == 3 && std::string_view(argv[2]) == "--postselect-all";
    if (argc == 3 && !postselect_all) {
        std::cerr << "profile_sampling_plan: unknown option " << argv[2] << "\n";
        return 2;
    }

    try {
        std::string text = read_file(argv[1]);
        clifft::HirModule hir = clifft::trace(clifft::parse(text));
        auto hir_passes = clifft::default_hir_pass_manager();
        hir_passes.run(hir);

        clifft::ReferenceSyndrome reference = clifft::compute_reference_syndrome(hir);
        std::vector<uint8_t> postselection;
        if (postselect_all) {
            postselection.assign(hir.num_detectors, 1);
        }
        clifft::sampling::SamplingPlan plan = clifft::sampling::plan_sampling(
            hir, {postselection, reference.detectors, reference.observables});

        std::map<std::string_view, uint64_t> action_counts;
        uint64_t expression_count = 0;
        uint64_t expression_terms = 0;
        uint64_t max_expression_terms = 0;
        uint64_t predicted_dense_passes = 0;
        uint64_t predicted_coefficient_visits = 0;
        std::vector<uint32_t> exp_val_widths;

        for (const auto& planned : plan.actions) {
            ++action_counts[action_name(planned.action)];
            const auto* expression = action_expression(planned.action);
            if (expression != nullptr) {
                const uint64_t terms = expression->terms().size();
                ++expression_count;
                expression_terms += terms;
                max_expression_terms = std::max(max_expression_terms, terms);
            }
            const uint64_t passes = clifft::sampling::predicted_dense_passes(planned.action);
            predicted_dense_passes += passes;
            predicted_coefficient_visits += passes * (uint64_t{1} << planned.active_before);
            if (std::holds_alternative<clifft::sampling::WriteExpectationValue>(planned.action)) {
                exp_val_widths.push_back(planned.active_before);
            }
        }

        std::cout << "{\n";
        std::cout << "  \"initial_active_width\": " << plan.initial_active_width << ",\n";
        std::cout << "  \"max_active_width\": " << plan.max_active_width << ",\n";
        std::cout << "  \"num_actions\": " << plan.actions.size() << ",\n";
        std::cout << "  \"num_symbols\": " << plan.symbols.size() << ",\n";
        std::cout << "  \"expression_count\": " << expression_count << ",\n";
        std::cout << "  \"expression_terms\": " << expression_terms << ",\n";
        std::cout << "  \"max_expression_terms\": " << max_expression_terms << ",\n";
        std::cout << "  \"predicted_dense_passes\": " << predicted_dense_passes << ",\n";
        std::cout << "  \"predicted_coefficient_visits\": " << predicted_coefficient_visits
                  << ",\n";
        std::cout << "  \"exp_val_widths\": ";
        print_u32_vector(exp_val_widths);
        std::cout << ",\n  \"action_counts\": {";
        size_t index = 0;
        for (const auto& [name, count] : action_counts) {
            if (index++ != 0) {
                std::cout << ",";
            }
            std::cout << "\n    \"" << name << "\": " << count;
        }
        if (!action_counts.empty()) {
            std::cout << "\n  ";
        }
        std::cout << "}\n}\n";
    } catch (const std::exception& error) {
        std::cerr << "profile_sampling_plan: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
