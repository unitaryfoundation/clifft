// Minimal static census for scalar expression and active-operation prototypes.

#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <bit>
#include <compare>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

struct ExpressionKey {
    bool constant = false;
    std::vector<uint32_t> terms;

    auto operator<=>(const ExpressionKey&) const = default;
};

using RotationKey = std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>;

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
            } else if constexpr (std::is_same_v<T,
                                                clifft::sampling::PromoteDormantRotation>) {
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
            } else if constexpr (std::is_same_v<T,
                                                clifft::sampling::WriteExpectationValue>) {
                return "write_exp_val";
            } else if constexpr (std::is_same_v<T, clifft::sampling::ApplyInstrument>) {
                return "apply_instrument";
            } else {
                return "instrument_boundary";
            }
        },
        action);
}

std::string_view symbol_kind_name(clifft::sampling::SymbolKind kind) {
    using clifft::sampling::SymbolKind;
    switch (kind) {
        case SymbolKind::Unused:
            return "unused";
        case SymbolKind::Presampled:
            return "presampled";
        case SymbolKind::Derived:
            return "derived";
        case SymbolKind::Branch:
            return "branch";
        case SymbolKind::Readout:
            return "readout";
        case SymbolKind::Instrument:
            return "instrument";
    }
    throw std::logic_error("unhandled symbol kind");
}

std::optional<ExpressionKey> executed_expression(
    const clifft::sampling::SamplingAction& action) {
    return std::visit(
        [](const auto& typed) -> std::optional<ExpressionKey> {
            using T = std::decay_t<decltype(typed)>;
            const clifft::sampling::AffineBool* expression = nullptr;
            std::optional<uint32_t> excluded_symbol;
            if constexpr (std::is_same_v<T, clifft::sampling::RotateActivePauli> ||
                          std::is_same_v<T, clifft::sampling::PromoteDormantRotation>) {
                expression = &typed.sign;
            } else if constexpr (std::is_same_v<T, clifft::sampling::MeasureActivePauli> ||
                                 std::is_same_v<T,
                                                clifft::sampling::MeasureDormantRandom>) {
                expression = &typed.outcome;
                excluded_symbol = clifft::sampling::index(typed.branch);
            } else if constexpr (std::is_same_v<T, clifft::sampling::RecordClassical> ||
                                 std::is_same_v<T, clifft::sampling::WriteDetector> ||
                                 std::is_same_v<T, clifft::sampling::WriteObservable>) {
                expression = &typed.outcome;
            } else if constexpr (std::is_same_v<T, clifft::sampling::DefineSymbol>) {
                expression = &typed.value;
            } else if constexpr (std::is_same_v<T,
                                                clifft::sampling::ApplyReadoutNoise>) {
                expression = &typed.source;
            } else if constexpr (std::is_same_v<T,
                                                clifft::sampling::WriteExpectationValue> ||
                                 std::is_same_v<T, clifft::sampling::ApplyInstrument>) {
                expression = &typed.sign;
            } else {
                return std::nullopt;
            }

            ExpressionKey result;
            result.constant = expression->constant();
            result.terms.reserve(expression->terms().size());
            for (clifft::sampling::SymbolId term : expression->terms()) {
                const uint32_t symbol = clifft::sampling::index(term);
                if (!excluded_symbol.has_value() || symbol != *excluded_symbol) {
                    result.terms.push_back(symbol);
                }
            }
            return result;
        },
        action);
}

template <typename Key>
void print_map(const std::map<Key, uint64_t>& values) {
    std::cout << "{";
    size_t position = 0;
    for (const auto& [key, value] : values) {
        if (position++ != 0) {
            std::cout << ",";
        }
        std::cout << "\n    \"" << key << "\": " << value;
    }
    if (!values.empty()) {
        std::cout << "\n  ";
    }
    std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: profile_expression_census CIRCUIT [--postselect-all]\n";
        return 2;
    }
    const bool postselect_all = argc == 3 && std::string_view(argv[2]) == "--postselect-all";
    if (argc == 3 && !postselect_all) {
        std::cerr << "profile_expression_census: unknown option " << argv[2] << "\n";
        return 2;
    }

    try {
        clifft::HirModule hir = clifft::trace(clifft::parse(read_file(argv[1])));
        auto passes = clifft::default_hir_pass_manager();
        passes.run(hir);
        clifft::ReferenceSyndrome reference = clifft::compute_reference_syndrome(hir);
        std::vector<uint8_t> postselection;
        if (postselect_all) {
            postselection.assign(hir.num_detectors, 1);
        }
        const clifft::sampling::SamplingPlan plan = clifft::sampling::plan_sampling(
            hir, {postselection, reference.detectors, reference.observables});

        uint64_t expression_count = 0;
        uint64_t expression_terms = 0;
        uint64_t duplicate_expression_uses = 0;
        uint64_t duplicate_term_visits_per_full_shot = 0;
        std::map<ExpressionKey, uint64_t> expression_uses;
        std::map<std::string_view, uint64_t> terms_by_consumer;
        std::map<std::string_view, uint64_t> terms_by_symbol_kind;

        uint64_t rotation_count = 0;
        uint64_t rotation_coefficient_visits = 0;
        uint64_t consecutive_rotation_pairs = 0;
        std::map<std::string, uint64_t> rotations_by_width;
        std::map<std::string, uint64_t> rotation_visits_by_width;
        std::map<std::string_view, uint64_t> rotation_visits_by_shape;
        std::map<std::string_view, uint64_t> rotation_signs;
        std::map<RotationKey, uint64_t> rotation_uses;

        uint64_t active_measurement_count = 0;
        uint64_t active_measurement_coefficient_visits = 0;
        std::map<std::string, uint64_t> measurement_visits_by_width;
        std::map<std::string_view, uint64_t> measurement_visits_by_shape;

        bool previous_was_rotation = false;
        uint32_t previous_rotation_width = 0;
        for (const clifft::sampling::PlannedAction& planned : plan.actions) {
            const auto expression = executed_expression(planned.action);
            if (expression.has_value()) {
                ++expression_count;
                expression_terms += expression->terms.size();
                terms_by_consumer[action_name(planned.action)] += expression->terms.size();
                for (uint32_t symbol : expression->terms) {
                    terms_by_symbol_kind[symbol_kind_name(plan.symbols[symbol].kind)] += 1;
                }
                ++expression_uses[*expression];
            }

            if (const auto* rotation =
                    std::get_if<clifft::sampling::RotateActivePauli>(&planned.action)) {
                ++rotation_count;
                const uint64_t visits =
                    clifft::sampling::predicted_dense_passes(planned.action) *
                    (uint64_t{1} << planned.active_before);
                rotation_coefficient_visits += visits;
                const std::string width = std::to_string(planned.active_before);
                ++rotations_by_width[width];
                rotation_visits_by_width[width] += visits;
                const uint32_t x_weight = std::popcount(rotation->pauli.x);
                rotation_visits_by_shape[rotation->pauli.x == 0
                                             ? "diagonal"
                                             : (x_weight == 1 ? "single_x_bit" : "multi_x_bit")] +=
                    visits;
                rotation_signs[rotation->sign.terms().empty() ? "constant" : "shot_dependent"] +=
                    visits;
                ++rotation_uses[{planned.active_before, rotation->pauli.x, rotation->pauli.z,
                                 std::bit_cast<uint64_t>(rotation->half_turns)}];
                if (previous_was_rotation && previous_rotation_width == planned.active_before) {
                    ++consecutive_rotation_pairs;
                }
                previous_was_rotation = true;
                previous_rotation_width = planned.active_after;
            } else {
                previous_was_rotation = false;
            }

            if (const auto* measurement =
                    std::get_if<clifft::sampling::MeasureActivePauli>(&planned.action)) {
                ++active_measurement_count;
                const uint64_t visits =
                    clifft::sampling::predicted_dense_passes(planned.action) *
                    (uint64_t{1} << planned.active_before);
                active_measurement_coefficient_visits += visits;
                measurement_visits_by_width[std::to_string(planned.active_before)] += visits;
                measurement_visits_by_shape[measurement->pauli.x == 0 ? "diagonal"
                                                                       : "non_diagonal"] +=
                    visits;
            }
        }

        for (const auto& [expression, uses] : expression_uses) {
            if (uses > 1) {
                duplicate_expression_uses += uses - 1;
                duplicate_term_visits_per_full_shot +=
                    (uses - 1) * static_cast<uint64_t>(expression.terms.size());
            }
        }
        uint64_t repeated_rotation_uses = 0;
        for (const auto& [descriptor, uses] : rotation_uses) {
            (void)descriptor;
            if (uses > 1) {
                repeated_rotation_uses += uses - 1;
            }
        }

        std::cout << "{\n";
        std::cout << "  \"num_actions\": " << plan.actions.size() << ",\n";
        std::cout << "  \"num_symbols\": " << plan.symbols.size() << ",\n";
        std::cout << "  \"expression_count\": " << expression_count << ",\n";
        std::cout << "  \"expression_terms_per_full_shot\": " << expression_terms << ",\n";
        std::cout << "  \"unique_expressions\": " << expression_uses.size() << ",\n";
        std::cout << "  \"duplicate_expression_uses\": " << duplicate_expression_uses << ",\n";
        std::cout << "  \"duplicate_term_visits_per_full_shot\": "
                  << duplicate_term_visits_per_full_shot << ",\n";
        std::cout << "  \"expression_terms_by_consumer\": ";
        print_map(terms_by_consumer);
        std::cout << ",\n  \"expression_terms_by_symbol_kind\": ";
        print_map(terms_by_symbol_kind);
        std::cout << ",\n  \"rotation_count\": " << rotation_count << ",\n";
        std::cout << "  \"rotation_coefficient_visits\": " << rotation_coefficient_visits
                  << ",\n";
        std::cout << "  \"rotation_counts_by_width\": ";
        print_map(rotations_by_width);
        std::cout << ",\n  \"rotation_visits_by_width\": ";
        print_map(rotation_visits_by_width);
        std::cout << ",\n  \"rotation_visits_by_shape\": ";
        print_map(rotation_visits_by_shape);
        std::cout << ",\n  \"rotation_visits_by_sign\": ";
        print_map(rotation_signs);
        std::cout << ",\n  \"repeated_rotation_uses\": " << repeated_rotation_uses << ",\n";
        std::cout << "  \"consecutive_rotation_pairs\": " << consecutive_rotation_pairs
                  << ",\n";
        std::cout << "  \"active_measurement_count\": " << active_measurement_count << ",\n";
        std::cout << "  \"active_measurement_coefficient_visits\": "
                  << active_measurement_coefficient_visits << ",\n";
        std::cout << "  \"measurement_visits_by_width\": ";
        print_map(measurement_visits_by_width);
        std::cout << ",\n  \"measurement_visits_by_shape\": ";
        print_map(measurement_visits_by_shape);
        std::cout << "\n}\n";
    } catch (const std::exception& error) {
        std::cerr << "profile_expression_census: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
