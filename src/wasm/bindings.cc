// Emscripten/Embind bridge for the UCC Compiler Explorer.
//
// Exposes three functions to JavaScript:
//   get_available_passes() -> JSON string with pass registry
//   compile_to_json(source, passes_json) -> JSON string with HIR, bytecode, source maps
//   simulate_wasm(source, shots, passes_json) -> JSON string with measurement histogram

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/hir_pass_manager.h"
#include "ucc/optimizer/pass_factory.h"
#include "ucc/svm/svm.h"
#include "ucc/util/introspection.h"

#include <cstdint>
#include <emscripten/bind.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using json = nlohmann::json;

constexpr uint32_t MAX_SHOTS = 100000;
constexpr uint32_t MAX_OPS = 10000;
constexpr uint32_t MAX_PEAK_RANK = 20;

struct PipelineResult {
    ucc::HirModule hir;
    ucc::CompiledModule prog;
    std::string error;
};

// Parse passes_json: {"hir": [...], "bc": [...]}
// Empty string or "{}" means use defaults.
PipelineResult run_pipeline(const std::string& source, const std::string& passes_json) {
    PipelineResult result;
    try {
        auto circuit = ucc::parse(source, MAX_OPS);
        result.hir = ucc::trace(circuit);

        bool use_defaults = passes_json.empty() || passes_json == "{}";

        if (use_defaults) {
            auto hpm = ucc::default_hir_pass_manager();
            hpm.run(result.hir);
        } else {
            auto cfg = json::parse(passes_json);
            if (cfg.contains("hir") && cfg["hir"].is_array()) {
                ucc::HirPassManager hpm;
                for (const auto& name : cfg["hir"]) {
                    hpm.add_pass(ucc::make_hir_pass(name.get<std::string>()));
                }
                hpm.run(result.hir);
            }
        }

        result.prog = ucc::lower(result.hir);

        if (use_defaults) {
            auto bpm = ucc::default_bytecode_pass_manager();
            bpm.run(result.prog);
        } else {
            auto cfg = json::parse(passes_json);
            if (cfg.contains("bc") && cfg["bc"].is_array()) {
                ucc::BytecodePassManager bpm;
                for (const auto& name : cfg["bc"]) {
                    bpm.add_pass(ucc::make_bytecode_pass(name.get<std::string>()));
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
    return ucc::pass_registry_json();
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
        hir_strs.push_back(ucc::format_hir_op(op));
    }

    std::vector<std::string> bc_strs;
    bc_strs.reserve(prog.bytecode.size());
    for (const auto& instr : prog.bytecode) {
        bc_strs.push_back(ucc::format_instruction(instr));
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

    if (prog.num_measurements == 0) {
        return json({
                        {"histogram", json::object()},
                        {"shots", shots},
                        {"num_measurements", 0},
                    })
            .dump();
    }

    ucc::SampleResult samples = ucc::sample(prog, shots, std::nullopt);

    uint32_t n_meas = prog.num_measurements;
    std::unordered_map<std::string, uint32_t> histogram;
    std::string key;
    key.reserve(n_meas);
    for (uint32_t shot = 0; shot < shots; ++shot) {
        key.clear();
        for (uint32_t m = 0; m < n_meas; ++m) {
            key += (samples.measurements[shot * n_meas + m] ? '1' : '0');
        }
        ++histogram[key];
    }

    json j = {
        {"histogram", histogram},
        {"shots", shots},
        {"num_measurements", n_meas},
    };
    return j.dump();
}

}  // namespace

EMSCRIPTEN_BINDINGS(ucc_wasm) {
    emscripten::function("get_available_passes", &get_available_passes);
    emscripten::function("compile_to_json", &compile_to_json);
    emscripten::function("simulate_wasm", &simulate_wasm);
}
