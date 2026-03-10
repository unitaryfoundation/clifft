// Emscripten/Embind bridge for the UCC Compiler Explorer.
//
// Exposes two functions to JavaScript:
//   compile_to_json(source, optimize) -> JSON string with HIR, bytecode, source maps
//   simulate_wasm(source, shots, optimize) -> JSON string with measurement histogram

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/expand_t_pass.h"
#include "ucc/optimizer/multi_gate_pass.h"
#include "ucc/optimizer/noise_block_pass.h"
#include "ucc/optimizer/pass_manager.h"
#include "ucc/optimizer/peephole.h"
#include "ucc/optimizer/swap_meas_pass.h"
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

PipelineResult run_pipeline(const std::string& source, bool optimize) {
    PipelineResult result;
    try {
        auto circuit = ucc::parse(source, MAX_OPS);
        result.hir = ucc::trace(circuit);
        if (optimize) {
            ucc::PassManager pm;
            pm.add_pass(std::make_unique<ucc::PeepholeFusionPass>());
            pm.run(result.hir);
        }
        result.prog = ucc::lower(result.hir);
        if (optimize) {
            ucc::BytecodePassManager bpm;
            bpm.add_pass(std::make_unique<ucc::NoiseBlockPass>());
            bpm.add_pass(std::make_unique<ucc::MultiGatePass>());
            bpm.add_pass(std::make_unique<ucc::ExpandTPass>());
            bpm.add_pass(std::make_unique<ucc::SwapMeasPass>());
            bpm.run(result.prog);
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

std::string compile_to_json(const std::string& source, bool optimize) {
    auto result = run_pipeline(source, optimize);
    if (!result.error.empty()) {
        return json({{"error", result.error}}).dump();
    }
    const auto& hir = result.hir;
    const auto& prog = result.prog;

    // Format HIR ops as human-readable strings
    std::vector<std::string> hir_strs;
    hir_strs.reserve(hir.ops.size());
    for (const auto& op : hir.ops) {
        hir_strs.push_back(ucc::format_hir_op(op));
    }

    // Format bytecode as human-readable strings
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

std::string simulate_wasm(const std::string& source, uint32_t shots, bool optimize) {
    if (shots > MAX_SHOTS) {
        return json({{"error", "ShotsLimitExceeded: max " + std::to_string(MAX_SHOTS)}}).dump();
    }

    auto result = run_pipeline(source, optimize);
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

    // Aggregate measurement bitstrings into a histogram
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
    emscripten::function("compile_to_json", &compile_to_json);
    emscripten::function("simulate_wasm", &simulate_wasm);
}
