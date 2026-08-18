// Emscripten/Embind bridge for the Clifft Playground.
//
// Exposes three functions to JavaScript:
//   get_available_passes() -> JSON string with pass registry
//   compile_to_json(source, passes_json) -> JSON string with HIR, sampling plans, source maps
//   simulate_wasm(source, shots, passes_json) -> JSON string with measurement histogram

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/hir_introspection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <emscripten/bind.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using json = nlohmann::json;

constexpr uint32_t MAX_SHOTS = 100000;
constexpr uint32_t MAX_OPS = 50000;
constexpr uint32_t MAX_BROWSER_ACTIVE_WIDTH = 24;

struct PipelineResult {
    clifft::HirModule hir;
    std::optional<clifft::sampling::SamplingPlan> plan;
    std::unique_ptr<clifft::sampling::ExecutablePlan> program;
    std::string error;
};

// Parse passes_json: {"hir": [...]}
// Empty string or "{}" means use defaults.
PipelineResult run_pipeline(const std::string& source, const std::string& passes_json,
                            bool retain_source_map) {
    PipelineResult result;
    try {
        auto circuit = clifft::parse(source, MAX_OPS);
        result.hir = clifft::trace(circuit);

        bool use_defaults = passes_json.empty() || passes_json == "{}";

        if (use_defaults) {
            auto hpm = clifft::default_hir_pass_manager();
            hpm.run(result.hir);
        } else {
            auto cfg = json::parse(passes_json);
            if (!cfg.is_object()) {
                throw std::invalid_argument("Pass configuration must be a JSON object");
            }
            for (const auto& [key, unused] : cfg.items()) {
                static_cast<void>(unused);
                if (key != "hir") {
                    throw std::invalid_argument("Unknown pass configuration key: " + key);
                }
            }
            if (cfg.contains("hir") && cfg["hir"].is_array()) {
                clifft::HirPassManager hpm;
                for (const auto& name : cfg["hir"]) {
                    hpm.add_pass(clifft::make_hir_pass(name.get<std::string>()));
                }
                hpm.run(result.hir);
            }
        }

        clifft::sampling::SamplingPlanOptions options;
        options.retain_source_map = retain_source_map;
        result.plan.emplace(clifft::sampling::plan_sampling(result.hir, options));
        result.program = std::make_unique<clifft::sampling::ExecutablePlan>(*result.plan);
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

std::string get_available_passes() {
    const json registry = json::parse(clifft::pass_registry_json());
    json hir_passes = json::array();
    for (const auto& pass : registry) {
        if (pass.at("kind") == "hir") {
            hir_passes.push_back(pass);
        }
    }
    return hir_passes.dump();
}

json source_map_json(const clifft::sampling::PlanSourceMap& source_map) {
    json entries = json::array();
    for (size_t i = 0; i < source_map.size(); ++i) {
        const auto lines = source_map.lines_for(i);
        entries.push_back(std::vector<uint32_t>(lines.begin(), lines.end()));
    }
    return entries;
}

json executable_source_map_json(const clifft::sampling::SamplingPlan& plan,
                                const clifft::sampling::ExecutablePlan& program) {
    json entries = json::array();
    for (size_t i = 0; i < program.num_actions(); ++i) {
        std::vector<uint32_t> source_lines;
        const auto range = program.action_plan_range(i);
        if (range.has_value()) {
            for (uint32_t action = range->begin; action < range->end; ++action) {
                for (uint32_t line : plan.source_map->lines_for(action)) {
                    source_lines.push_back(line);
                }
            }
        }
        std::ranges::sort(source_lines);
        source_lines.erase(std::unique(source_lines.begin(), source_lines.end()),
                           source_lines.end());
        entries.push_back(std::move(source_lines));
    }
    return entries;
}

std::string compile_to_json(const std::string& source, const std::string& passes_json) {
    auto result = run_pipeline(source, passes_json, true);
    if (!result.error.empty()) {
        return json({{"error", result.error}}).dump();
    }
    const auto& hir = result.hir;
    const auto& plan = *result.plan;
    const auto& program = *result.program;

    std::vector<std::string> hir_strs;
    hir_strs.reserve(hir.ops.size());
    for (const auto& op : hir.ops) {
        auto mask = op.has_mask() ? std::optional{hir.mask_view(op)} : std::nullopt;
        hir_strs.push_back(clifft::format_hir_op(op, mask));
    }

    std::vector<std::string> plan_strs;
    plan_strs.reserve(plan.actions.size());
    std::vector<uint32_t> active_width_history;
    active_width_history.reserve(plan.actions.size());
    for (size_t i = 0; i < plan.actions.size(); ++i) {
        plan_strs.push_back(plan.inspect_action_compact(i));
        active_width_history.push_back(plan.actions[i].active_after);
    }

    std::vector<std::string> program_strs;
    program_strs.reserve(program.num_actions());
    json program_plan_ranges = json::array();
    for (size_t i = 0; i < program.num_actions(); ++i) {
        program_strs.push_back(program.inspect_action(i));
        const auto range = program.action_plan_range(i).value();
        program_plan_ranges.push_back({{"begin", range.begin}, {"end", range.end}});
    }

    json j = {
        {"num_qubits", plan.num_qubits},
        {"peak_active_width", plan.peak_active_width},
        {"num_measurements", plan.num_visible_records},
        {"num_t_gates", hir.num_t_gates()},
        {"hir_ops", hir_strs},
        {"sampling_plan", plan_strs},
        {"wasm_program", program_strs},
        {"hir_source_map", hir.source_map},
        {"sampling_plan_source_map", source_map_json(*plan.source_map)},
        {"wasm_program_source_map", executable_source_map_json(plan, program)},
        {"wasm_program_plan_ranges", program_plan_ranges},
        {"active_width_history", active_width_history},
    };
    return j.dump();
}

// Extract EXP_VAL labels from source text by re-parsing lines.
// Returns one entry per Pauli product: {label, line} where line is 1-based.
// "EXP_VAL X0*Z2 Y1" produces two entries: ("X0*Z2", line) and ("Y1", line).
std::vector<std::pair<std::string, uint32_t>> extract_exp_val_labels(const std::string& source) {
    std::vector<std::pair<std::string, uint32_t>> labels;
    std::istringstream stream(source);
    std::string line;
    uint32_t line_num = 0;
    while (std::getline(stream, line)) {
        ++line_num;
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        std::string_view sv(line.data() + start, line.size() - start);
        if (sv.substr(0, 8) != "EXP_VAL " && sv != "EXP_VAL")
            continue;
        if (sv.size() <= 8)
            continue;
        // Extract the rest after "EXP_VAL "
        std::string_view rest = sv.substr(8);
        // Split on whitespace to get individual Pauli products
        size_t pos = 0;
        while (pos < rest.size()) {
            while (pos < rest.size() && (rest[pos] == ' ' || rest[pos] == '\t'))
                ++pos;
            if (pos >= rest.size())
                break;
            size_t end = pos;
            while (end < rest.size() && rest[end] != ' ' && rest[end] != '\t')
                ++end;
            labels.emplace_back(std::string(rest.substr(pos, end - pos)), line_num);
            pos = end;
        }
    }
    return labels;
}

std::string simulate_wasm(const std::string& source, uint32_t shots,
                          const std::string& passes_json) {
    if (shots > MAX_SHOTS) {
        return json({{"error", "ShotsLimitExceeded: max " + std::to_string(MAX_SHOTS)}}).dump();
    }

    auto result = run_pipeline(source, passes_json, false);
    if (!result.error.empty()) {
        return json({{"error", result.error}}).dump();
    }
    const auto& program = *result.program;

    if (program.peak_active_width() > MAX_BROWSER_ACTIVE_WIDTH) {
        return json({{"error", "MemoryLimitExceeded"}}).dump();
    }

    uint32_t n_meas = program.num_visible_records();
    uint32_t n_ev = program.num_exp_vals();

    if (n_meas == 0 && n_ev == 0) {
        return json({
                        {"histogram", json::object()},
                        {"shots", shots},
                        {"num_measurements", 0},
                        {"exp_vals", json::array()},
                    })
            .dump();
    }

    clifft::sampling::SamplingResult samples =
        clifft::sampling::sample(program, shots, std::nullopt);

    // Build measurement histogram
    std::unordered_map<std::string, uint32_t> histogram;
    if (n_meas > 0) {
        std::string key;
        key.reserve(n_meas);
        for (uint32_t shot = 0; shot < shots; ++shot) {
            key.clear();
            for (uint32_t m = 0; m < n_meas; ++m) {
                key += (samples.measurements[shot * n_meas + m] ? '1' : '0');
            }
            ++histogram[key];
        }
    }

    // Build exp_val statistics (mean and std per probe)
    json ev_arr = json::array();
    if (n_ev > 0) {
        auto labels = extract_exp_val_labels(source);
        for (uint32_t ei = 0; ei < n_ev; ++ei) {
            double sum = 0.0;
            double sum_sq = 0.0;
            for (uint32_t shot = 0; shot < shots; ++shot) {
                double v = samples.exp_vals[shot * n_ev + ei];
                sum += v;
                sum_sq += v * v;
            }
            double mean = sum / shots;
            double variance = (sum_sq / shots) - (mean * mean);
            double stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;

            json entry = {{"mean", mean}, {"std", stddev}};
            if (ei < labels.size()) {
                entry["label"] = labels[ei].first;
                entry["line"] = labels[ei].second;
            }
            ev_arr.push_back(entry);
        }
    }

    json j = {
        {"histogram", histogram},
        {"shots", shots},
        {"num_measurements", n_meas},
        {"exp_vals", ev_arr},
    };
    return j.dump();
}

}  // namespace

EMSCRIPTEN_BINDINGS(clifft_wasm) {
    emscripten::function("get_available_passes", &get_available_passes);
    emscripten::function("compile_to_json", &compile_to_json);
    emscripten::function("simulate_wasm", &simulate_wasm);
}
