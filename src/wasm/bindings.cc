// Emscripten/Embind bridge for the Clifft Playground.
//
// Exposes three functions to JavaScript:
//   get_available_passes() -> JSON string with pass registry
//   compile_to_json(source, passes_json) -> JSON string with HIR, bytecode, source maps
//   simulate_wasm(source, shots, passes_json) -> JSON string with measurement histogram

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"
#include "clifft/util/introspection.h"

#include <cmath>
#include <cstdint>
#include <emscripten/bind.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using json = nlohmann::json;

constexpr uint32_t MAX_SHOTS = 100000;
constexpr uint32_t MAX_OPS = 50000;
constexpr uint32_t MAX_PEAK_RANK = 24;

struct PipelineResult {
    clifft::HirModule hir;
    clifft::CompiledModule prog;
    std::string error;
};

// Parse passes_json: {"hir": [...], "bc": [...]}
// Empty string or "{}" means use defaults.
PipelineResult run_pipeline(const std::string& source, const std::string& passes_json) {
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
            if (cfg.contains("hir") && cfg["hir"].is_array()) {
                clifft::HirPassManager hpm;
                for (const auto& name : cfg["hir"]) {
                    hpm.add_pass(clifft::make_hir_pass(name.get<std::string>()));
                }
                hpm.run(result.hir);
            }
        }

        result.prog = clifft::lower(result.hir);

        if (use_defaults) {
            auto bpm = clifft::default_bytecode_pass_manager();
            bpm.run(result.prog);
        } else {
            auto cfg = json::parse(passes_json);
            if (cfg.contains("bc") && cfg["bc"].is_array()) {
                clifft::BytecodePassManager bpm;
                for (const auto& name : cfg["bc"]) {
                    bpm.add_pass(clifft::make_bytecode_pass(name.get<std::string>()));
                }
                bpm.run(result.prog);
            }
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

std::string get_available_passes() {
    return clifft::pass_registry_json();
}

std::string compile_to_json(const std::string& source, const std::string& passes_json) {
    auto result = run_pipeline(source, passes_json);
    if (!result.error.empty()) {
        return json({{"error", result.error}}).dump();
    }
    const auto& hir = result.hir;
    const auto& prog = result.prog;

    std::vector<std::string> hir_strs;
    hir_strs.reserve(hir.ops.size());
    for (const auto& op : hir.ops) {
        hir_strs.push_back(clifft::format_hir_op(op));
    }

    std::vector<std::string> bc_strs;
    bc_strs.reserve(prog.bytecode.size());
    for (const auto& instr : prog.bytecode) {
        bc_strs.push_back(clifft::format_instruction(instr));
    }

    json j = {
        {"num_qubits", prog.num_qubits},
        {"peak_rank", prog.peak_rank},
        {"num_measurements", prog.num_measurements},
        {"num_t_gates", hir.num_t_gates()},
        {"hir_ops", hir_strs},
        {"bytecode", bc_strs},
        {"hir_source_map", hir.source_map},
        {"bytecode_source_map",
         [&]() {
             json arr = json::array();
             for (size_t i = 0; i < prog.source_map.size(); ++i) {
                 auto lines = prog.source_map.lines_for(i);
                 arr.push_back(std::vector<uint32_t>(lines.begin(), lines.end()));
             }
             return arr;
         }()},
        {"active_k_history", prog.source_map.active_k_history()},
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

    auto result = run_pipeline(source, passes_json);
    if (!result.error.empty()) {
        return json({{"error", result.error}}).dump();
    }
    const auto& prog = result.prog;

    if (prog.peak_rank > MAX_PEAK_RANK) {
        return json({{"error", "MemoryLimitExceeded"}}).dump();
    }

    uint32_t n_meas = prog.num_measurements;
    uint32_t n_ev = prog.num_exp_vals;

    if (n_meas == 0 && n_ev == 0) {
        return json({
                        {"histogram", json::object()},
                        {"shots", shots},
                        {"num_measurements", 0},
                        {"exp_vals", json::array()},
                    })
            .dump();
    }

    clifft::SampleResult samples = clifft::sample(prog, shots, std::nullopt);

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
