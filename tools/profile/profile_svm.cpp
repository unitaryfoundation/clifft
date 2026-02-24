// UCC SVM Profiling Tool
//
// Generates or loads a quantum circuit, compiles it, and runs N shots
// for profiling with Linux perf or other sampling profilers.
//
// See tools/profile/README.md for full usage instructions.

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
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

// Configuration: set via environment variables or defaults
// UCC_NUM_QUBITS, UCC_CLIFFORD_DEPTH, UCC_T_GATES, UCC_SHOTS
constexpr int kDefaultNumQubits = 50;
constexpr int kDefaultCliffordDepth = 5000;
constexpr int kDefaultTGates = 0;  // T-gates to add (increases peak_rank)
constexpr uint32_t kDefaultShots = 100'000;
constexpr uint64_t kSeed = 42;

std::string generate_circuit(int num_qubits, int clifford_depth, int t_gates, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::ostringstream ss;

    // Random Clifford gates
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

    // T-gates (each adds a dimension to the statevector if not in span)
    for (int i = 0; i < t_gates; ++i) {
        int q = rng() % num_qubits;
        ss << "T " << q << "\n";
        // Add some Cliffords between T-gates for realism
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
    // Read configuration from environment
    int num_qubits = get_env_int("UCC_NUM_QUBITS", kDefaultNumQubits);
    int clifford_depth = get_env_int("UCC_CLIFFORD_DEPTH", kDefaultCliffordDepth);
    int t_gates = get_env_int("UCC_T_GATES", kDefaultTGates);
    uint32_t shots = static_cast<uint32_t>(get_env_int("UCC_SHOTS", kDefaultShots));
    const char* circuit_file = std::getenv("UCC_CIRCUIT_FILE");

    std::cout << "UCC SVM Profiler\n";
    std::cout << "================\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    std::string circuit_text;
    if (circuit_file) {
        std::cout << "Circuit: " << circuit_file << "\n";
        std::cout << "Shots:   " << shots << "\n\n";
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
        std::cout << "(Set UCC_CIRCUIT_FILE to profile a file, or UCC_NUM_QUBITS, "
                     "UCC_CLIFFORD_DEPTH, UCC_T_GATES, UCC_SHOTS)\n\n";

        std::cout << "Generating circuit..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        circuit_text = generate_circuit(num_qubits, clifford_depth, t_gates, kSeed);
        t1 = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << " done (" << gen_ms << " ms)\n";
    }

    // Parse circuit
    std::cout << "Parsing..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::Circuit circuit = ucc::parse(circuit_text);
    t1 = std::chrono::high_resolution_clock::now();
    auto parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << parse_ms << " ms, " << circuit.nodes.size() << " ops)\n";

    // Frontend: trace through Clifford simulator
    std::cout << "Frontend (Clifford absorption)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::HirModule hir = ucc::trace(circuit);
    t1 = std::chrono::high_resolution_clock::now();
    auto trace_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << trace_ms << " ms)\n";

    // Backend: lower to bytecode
    std::cout << "Backend (bytecode generation)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::CompiledModule program = ucc::lower(hir);
    t1 = std::chrono::high_resolution_clock::now();
    auto lower_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << lower_ms << " ms, " << program.bytecode.size() << " instructions)\n";

    std::cout << "\nCompilation total: " << (parse_ms + trace_ms + lower_ms) << " ms\n";
    size_t total_gates = circuit.nodes.size();
    std::cout << "Compression: " << total_gates << " gates -> " << program.bytecode.size()
              << " instructions (" << (double(total_gates) / program.bytecode.size()) << "x)\n";
    std::cout << "Peak rank: " << program.peak_rank << " (statevector size: 2^" << program.peak_rank
              << " = " << (1ULL << program.peak_rank) << ")\n\n";

    // Run sampling (this is what we want to profile)
    std::cout << "Running " << shots << " shots..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    ucc::SampleResult result = ucc::sample(program, shots, 0);
    t1 = std::chrono::high_resolution_clock::now();
    auto sample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double us_per_shot = (sample_ms * 1000.0) / shots;
    std::cout << " done\n";
    std::cout << "\nSampling Results:\n";
    std::cout << "  Total:    " << sample_ms << " ms\n";
    std::cout << "  Per shot: " << us_per_shot << " µs\n";
    std::cout << "  Measurements: " << result.measurements.size() << " bytes\n";

    // Sanity check: verify we got results
    size_t ones = 0;
    for (uint8_t b : result.measurements) {
        ones += b;
    }
    std::cout << "  Ones fraction: " << (double(ones) / result.measurements.size()) << "\n";

    return 0;
}
