// Clifft compilation profiler
//
// Sibling to profile_svm.cpp, but biased toward compilation rather than
// sampling. Loops parse -> trace -> HirPassManager -> lower ->
// BytecodePassManager many times so a `perf record` run captures the
// compile path instead of the sample loop.
//
// See tools/profile/README.md for build instructions.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultIterations = 20;
constexpr int kDefaultNumQubits = 50;
constexpr int kDefaultCliffordDepth = 5000;
constexpr int kDefaultTGates = 0;
constexpr uint64_t kSeed = 42;

std::string generate_circuit(int num_qubits, int clifford_depth, int t_gates, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::ostringstream ss;
    for (int i = 0; i < clifford_depth; ++i) {
        int gate_type = rng() % 3;
        if (gate_type == 0) {
            ss << "H " << (rng() % num_qubits) << "\n";
        } else if (gate_type == 1) {
            ss << "S " << (rng() % num_qubits) << "\n";
        } else {
            int q1 = rng() % num_qubits;
            int q2 = rng() % num_qubits;
            while (q2 == q1)
                q2 = rng() % num_qubits;
            ss << "CX " << q1 << " " << q2 << "\n";
        }
    }
    for (int i = 0; i < t_gates; ++i) {
        ss << "T " << (rng() % num_qubits) << "\n";
        if (i < t_gates - 1)
            ss << "H " << (rng() % num_qubits) << "\n";
    }
    ss << "M";
    for (int i = 0; i < num_qubits; ++i)
        ss << " " << i;
    ss << "\n";
    return ss.str();
}

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

struct StageTimings {
    double parse_ms = 0.0;
    double trace_ms = 0.0;  // includes HirPassManager
    double lower_ms = 0.0;
    double bpm_ms = 0.0;
};

double total_ms(const StageTimings& t) {
    return t.parse_ms + t.trace_ms + t.lower_ms + t.bpm_ms;
}

void summarize(const std::string& label, std::vector<double> samples) {
    if (samples.empty()) {
        std::cout << "  " << label << ": (no samples)\n";
        return;
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double v : samples)
        sum += v;
    double mean = sum / samples.size();
    double median = samples[samples.size() / 2];
    double p95 = samples[(samples.size() * 95) / 100];
    double min_v = samples.front();
    double max_v = samples.back();
    std::cout << "  " << label << ": min " << min_v << " ms, median " << median << " ms, mean "
              << mean << " ms, p95 " << p95 << " ms, max " << max_v << " ms\n";
}

}  // namespace

int main() {
    int iterations = get_env_int("CLIFFT_COMPILE_ITERATIONS", kDefaultIterations);
    int num_qubits = get_env_int("CLIFFT_NUM_QUBITS", kDefaultNumQubits);
    int clifford_depth = get_env_int("CLIFFT_CLIFFORD_DEPTH", kDefaultCliffordDepth);
    int t_gates = get_env_int("CLIFFT_T_GATES", kDefaultTGates);
    const char* circuit_file = std::getenv("CLIFFT_CIRCUIT_FILE");
    bool postselect_all = get_env_bool("CLIFFT_POSTSELECT_ALL");

    std::cout << "Clifft Compile Profiler\n";
    std::cout << "================\n";

    // Generate or load the circuit text once; we want to time compilation,
    // not file IO.
    std::string circuit_text;
    if (circuit_file) {
        std::cout << "Circuit: " << circuit_file << "\n";
        circuit_text = read_file(circuit_file);
        std::cout << "Loaded:  " << circuit_text.size() << " bytes\n";
    } else {
        std::cout << "Circuit: " << num_qubits << " qubits, " << clifford_depth
                  << " Clifford gates";
        if (t_gates > 0) {
            std::cout << ", " << t_gates << " T-gates";
        }
        std::cout << " (generated)\n";
        circuit_text = generate_circuit(num_qubits, clifford_depth, t_gates, kSeed);
    }
    std::cout << "Iterations:     " << iterations << "\n";
    std::cout << "Postselect all: " << (postselect_all ? "yes" : "no") << "\n";
    std::cout << "(Set CLIFFT_COMPILE_ITERATIONS, CLIFFT_CIRCUIT_FILE, CLIFFT_NUM_QUBITS, "
                 "CLIFFT_CLIFFORD_DEPTH, CLIFFT_T_GATES, CLIFFT_POSTSELECT_ALL)\n\n";

    std::vector<StageTimings> samples;
    samples.reserve(iterations);

    size_t parsed_ops = 0;
    size_t bytecode_size_pre = 0;
    size_t bytecode_size_post = 0;
    uint32_t peak_rank = 0;
    uint32_t total_qubits = 0;

    auto outer_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        StageTimings t{};

        // Parse
        auto t0 = std::chrono::high_resolution_clock::now();
        clifft::Circuit circuit = clifft::parse(circuit_text);
        auto t1 = std::chrono::high_resolution_clock::now();
        t.parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (iter == 0)
            parsed_ops = circuit.nodes.size();

        // Frontend (trace + default HIR passes)
        t0 = std::chrono::high_resolution_clock::now();
        clifft::HirModule hir = clifft::trace(circuit);
        auto pm = clifft::default_hir_pass_manager();
        pm.run(hir);
        t1 = std::chrono::high_resolution_clock::now();
        t.trace_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Build postselection mask if requested.
        std::vector<uint8_t> postselection_mask;
        if (postselect_all) {
            uint32_t num_det = 0;
            for (const auto& op : hir.ops) {
                if (op.op_type() == clifft::OpType::DETECTOR) {
                    ++num_det;
                }
            }
            postselection_mask.assign(num_det, 1);
        }

        // Backend lower()
        t0 = std::chrono::high_resolution_clock::now();
        clifft::CompiledModule program = clifft::lower(hir, postselection_mask);
        t1 = std::chrono::high_resolution_clock::now();
        t.lower_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (iter == 0) {
            bytecode_size_pre = program.bytecode.size();
            peak_rank = program.peak_rank;
            total_qubits = program.num_qubits;
        }

        // Bytecode pass manager (production default set).
        t0 = std::chrono::high_resolution_clock::now();
        auto bpm = clifft::default_bytecode_pass_manager();
        bpm.run(program);
        t1 = std::chrono::high_resolution_clock::now();
        t.bpm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (iter == 0)
            bytecode_size_post = program.bytecode.size();

        samples.push_back(t);

        // Drop into ostream once at the start to give a "first-run" baseline; the
        // rest of the iterations are the perf-record budget.
        if (iter == 0) {
            std::cout << "First-iteration timings (warmup):\n";
            std::cout << "  parse:  " << t.parse_ms << " ms\n";
            std::cout << "  trace:  " << t.trace_ms << " ms\n";
            std::cout << "  lower:  " << t.lower_ms << " ms\n";
            std::cout << "  bpm:    " << t.bpm_ms << " ms\n";
            std::cout << "  total:  " << total_ms(t) << " ms\n";
            std::cout << "Module: " << total_qubits << " qubits, peak_rank " << peak_rank << ", "
                      << parsed_ops << " parsed ops, " << bytecode_size_pre << " -> "
                      << bytecode_size_post << " bytecode instructions\n\n";
        }
    }

    auto outer_end = std::chrono::high_resolution_clock::now();
    double outer_ms = std::chrono::duration<double, std::milli>(outer_end - outer_start).count();

    std::vector<double> parse_v, trace_v, lower_v, bpm_v, total_v;
    for (const auto& s : samples) {
        parse_v.push_back(s.parse_ms);
        trace_v.push_back(s.trace_ms);
        lower_v.push_back(s.lower_ms);
        bpm_v.push_back(s.bpm_ms);
        total_v.push_back(total_ms(s));
    }

    std::cout << "Per-stage distribution across " << iterations << " iterations:\n";
    summarize("parse  ", parse_v);
    summarize("trace  ", trace_v);
    summarize("lower  ", lower_v);
    summarize("bpm    ", bpm_v);
    summarize("total  ", total_v);

    double mean_total = 0;
    for (double v : total_v)
        mean_total += v;
    mean_total /= iterations;
    double compiles_per_sec = mean_total > 0 ? 1000.0 / mean_total : 0.0;

    std::cout << "\nThroughput:\n";
    std::cout << "  Wall-clock for all " << iterations << " compiles: " << outer_ms << " ms\n";
    std::cout << "  Compiles/sec (mean total):                " << compiles_per_sec << "\n";

    return 0;
}
