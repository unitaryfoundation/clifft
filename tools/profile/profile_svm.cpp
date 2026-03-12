// UCC SVM Profiling Tool
//
// Generates or loads a quantum circuit, compiles it, and runs N shots
// for profiling with Linux perf or other sampling profilers.
//
// See tools/profile/README.md for full usage instructions.

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/expand_t_pass.h"
#include "ucc/optimizer/hir_pass_manager.h"
#include "ucc/optimizer/multi_gate_pass.h"
#include "ucc/optimizer/noise_block_pass.h"
#include "ucc/optimizer/peephole.h"
#include "ucc/optimizer/swap_meas_pass.h"
#include "ucc/svm/svm.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace {

constexpr int kDefaultNumQubits = 50;
constexpr int kDefaultCliffordDepth = 5000;
constexpr int kDefaultTGates = 0;
constexpr uint32_t kDefaultShots = 100'000;
constexpr uint64_t kSeed = 42;

std::string generate_circuit(int num_qubits, int clifford_depth, int t_gates, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::ostringstream ss;

    for (int i = 0; i < clifford_depth; ++i) {
        int gate_type = rng() % 3;

        if (gate_type == 0) {
            int q = rng() % num_qubits;
            ss << "H " << q << "\n";
        } else if (gate_type == 1) {
            int q = rng() % num_qubits;
            ss << "S " << q << "\n";
        } else {
            int q1 = rng() % num_qubits;
            int q2 = rng() % num_qubits;
            while (q2 == q1) {
                q2 = rng() % num_qubits;
            }
            ss << "CX " << q1 << " " << q2 << "\n";
        }
    }

    for (int i = 0; i < t_gates; ++i) {
        int q = rng() % num_qubits;
        ss << "T " << q << "\n";
        if (i < t_gates - 1) {
            int q2 = rng() % num_qubits;
            ss << "H " << q2 << "\n";
        }
    }

    ss << "M";
    for (int i = 0; i < num_qubits; ++i) {
        ss << " " << i;
    }
    ss << "\n";

    return ss.str();
}

}  // namespace

int get_env_int(const char* name, int default_val) {
    const char* val = std::getenv(name);
    return val ? std::stoi(val) : default_val;
}

bool get_env_bool(const char* name) {
    const char* val = std::getenv(name);
    return val != nullptr && std::string(val) != "0" && std::string(val) != "";
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open file: " << path << "\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    int num_qubits = get_env_int("UCC_NUM_QUBITS", kDefaultNumQubits);
    int clifford_depth = get_env_int("UCC_CLIFFORD_DEPTH", kDefaultCliffordDepth);
    int t_gates = get_env_int("UCC_T_GATES", kDefaultTGates);
    uint32_t shots = static_cast<uint32_t>(get_env_int("UCC_SHOTS", kDefaultShots));
    const char* circuit_file = std::getenv("UCC_CIRCUIT_FILE");
    bool postselect_all = get_env_bool("UCC_POSTSELECT_ALL");

    std::cout << "UCC SVM Profiler\n";
    std::cout << "================\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    std::string circuit_text;
    if (circuit_file) {
        std::cout << "Circuit: " << circuit_file << "\n";
        std::cout << "Shots:   " << shots << "\n";
        std::cout << "Postselect all: " << (postselect_all ? "yes" : "no") << "\n\n";
        std::cout << "Loading circuit..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        circuit_text = read_file(circuit_file);
        t1 = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << " done (" << gen_ms << " ms, " << circuit_text.size() << " bytes)\n";
    } else {
        std::cout << "Circuit: " << num_qubits << " qubits, " << clifford_depth
                  << " Clifford gates";
        if (t_gates > 0) {
            std::cout << ", " << t_gates << " T-gates";
        }
        std::cout << "\n";
        std::cout << "Shots:   " << shots << "\n";
        std::cout << "Postselect all: " << (postselect_all ? "yes" : "no") << "\n";
        std::cout << "(Set UCC_CIRCUIT_FILE to profile a file, or UCC_NUM_QUBITS, "
                     "UCC_CLIFFORD_DEPTH, UCC_T_GATES, UCC_SHOTS)\n\n";

        std::cout << "Generating circuit..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        circuit_text = generate_circuit(num_qubits, clifford_depth, t_gates, kSeed);
        t1 = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << " done (" << gen_ms << " ms)\n";
    }

    // Parse
    std::cout << "Parsing..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::Circuit circuit = ucc::parse(circuit_text);
    t1 = std::chrono::high_resolution_clock::now();
    auto parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << parse_ms << " ms, " << circuit.nodes.size() << " ops)\n";

    // Frontend
    std::cout << "Frontend (Clifford absorption)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::HirModule hir = ucc::trace(circuit);
    ucc::HirPassManager pm;
    pm.add_pass(std::make_unique<ucc::PeepholeFusionPass>());
    pm.run(hir);
    t1 = std::chrono::high_resolution_clock::now();
    auto trace_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << trace_ms << " ms)\n";

    // Build postselection mask if requested
    std::vector<uint8_t> postselection_mask;
    if (postselect_all) {
        uint32_t num_det = 0;
        for (const auto& op : hir.ops) {
            if (op.op_type() == ucc::OpType::DETECTOR) {
                ++num_det;
            }
        }
        postselection_mask.assign(num_det, 1);
        std::cout << "Postselection: all " << num_det << " detectors\n";
    }

    // Backend
    std::cout << "Backend (bytecode generation)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::CompiledModule program = ucc::lower(hir, postselection_mask);
    t1 = std::chrono::high_resolution_clock::now();
    auto lower_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t pre_opt_count = program.bytecode.size();
    std::cout << " done (" << lower_ms << " ms, " << pre_opt_count << " instructions)\n";

    // Bytecode optimization passes
    std::cout << "Bytecode optimization..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::BytecodePassManager bpm;
    bpm.add_pass(std::make_unique<ucc::NoiseBlockPass>());
    bpm.add_pass(std::make_unique<ucc::MultiGatePass>());
    bpm.add_pass(std::make_unique<ucc::ExpandTPass>());
    bpm.add_pass(std::make_unique<ucc::SwapMeasPass>());
    bpm.run(program);
    t1 = std::chrono::high_resolution_clock::now();
    auto opt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << opt_ms << " ms, " << pre_opt_count << " -> "
              << program.bytecode.size() << " instructions)\n";

    std::cout << "\nCompilation total: " << (parse_ms + trace_ms + lower_ms + opt_ms) << " ms\n";
    size_t total_gates = circuit.nodes.size();
    std::cout << "Compression: " << total_gates << " gates -> " << program.bytecode.size()
              << " instructions (" << (double(total_gates) / program.bytecode.size()) << "x)\n";
    std::cout << "Peak rank: " << program.peak_rank << " (statevector size: 2^" << program.peak_rank
              << " = " << (1ULL << program.peak_rank) << ")\n\n";

    // Run sampling
    if (postselect_all) {
        std::cout << "Running " << shots << " shots (with postselection)..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        ucc::SurvivorResult result = ucc::sample_survivors(program, shots, 0, false);
        t1 = std::chrono::high_resolution_clock::now();
        auto sample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double us_per_shot = (sample_ms * 1000.0) / shots;
        uint32_t discards = shots - result.passed_shots;
        double discard_pct = 100.0 * discards / shots;

        std::cout << " done\n";
        std::cout << "\nSampling Results:\n";
        std::cout << "  Total:      " << sample_ms << " ms\n";
        std::cout << "  Per shot:   " << us_per_shot << " us\n";
        std::cout << "  Shots/s:    " << static_cast<uint64_t>(shots / (sample_ms / 1000.0))
                  << "\n";
        std::cout << "  Survivors:  " << result.passed_shots << " / " << shots << "\n";
        std::cout << "  Discards:   " << discards << " (" << discard_pct << "%)\n";
        std::cout << "  Errors:     " << result.logical_errors << "\n";
    } else {
        std::cout << "Running " << shots << " shots..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        ucc::SampleResult result = ucc::sample(program, shots, 0);
        t1 = std::chrono::high_resolution_clock::now();
        auto sample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double us_per_shot = (sample_ms * 1000.0) / shots;
        std::cout << " done\n";
        std::cout << "\nSampling Results:\n";
        std::cout << "  Total:    " << sample_ms << " ms\n";
        std::cout << "  Per shot: " << us_per_shot << " us\n";
        std::cout << "  Measurements: " << result.measurements.size() << " bytes\n";

        size_t ones = 0;
        for (uint8_t b : result.measurements) {
            ones += b;
        }
        std::cout << "  Ones fraction: " << (double(ones) / result.measurements.size()) << "\n";
    }

    return 0;
}
