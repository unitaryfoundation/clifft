// Clifft Strong-Simulation Profiling Tool
//
// Generates or loads a unitary quantum circuit, compiles it, and calls
// clifft::basis_probabilities() on a batch of computational-basis bitstrings.
// Used with Linux perf or other sampling profilers to investigate the
// hot path of the basis-state probability query.
//
// basis_probabilities() rejects measurement/feedback/noise/detector/observable
// opcodes, so this harness emits a unitary-only circuit (no trailing M).
//
// See tools/profile/README.md for full usage instructions.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

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

constexpr int kDefaultNumQubits = 20;
constexpr int kDefaultCliffordDepth = 200;
constexpr int kDefaultTGates = 20;
constexpr uint32_t kDefaultQueries = 100;
constexpr uint64_t kSeed = 42;

// Build a small unitary Clifford+T circuit with no terminal measurements.
// Generic generator: a random Clifford layer followed by a configurable
// T-gate layer interleaved with Hadamards.
std::string generate_circuit(int num_qubits, int clifford_depth, int t_gates, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::ostringstream ss;

    // When num_qubits == 1 there is no second qubit for CX, so drop CX from
    // the mix. Caller is expected to ensure num_qubits >= 1.
    const int num_gate_types = num_qubits >= 2 ? 3 : 2;
    for (int i = 0; i < clifford_depth; ++i) {
        int gate_type = rng() % num_gate_types;

        if (gate_type == 0) {
            ss << "H " << (rng() % num_qubits) << "\n";
        } else if (gate_type == 1) {
            ss << "S " << (rng() % num_qubits) << "\n";
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
        ss << "T " << (rng() % num_qubits) << "\n";
        if (i < t_gates - 1) {
            ss << "H " << (rng() % num_qubits) << "\n";
        }
    }

    return ss.str();
}

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

std::vector<uint64_t> generate_random_bitmasks(uint32_t num_qubits, uint32_t num_queries,
                                               size_t words_per_mask, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> masks(static_cast<size_t>(num_queries) * words_per_mask, 0);
    const uint32_t used_bits = num_qubits % 64;
    const uint64_t high_word_mask =
        used_bits == 0 ? ~uint64_t{0} : ((uint64_t{1} << used_bits) - 1ULL);
    for (uint32_t q = 0; q < num_queries; ++q) {
        for (size_t w = 0; w < words_per_mask; ++w) {
            uint64_t word = rng();
            if (w + 1 == words_per_mask) {
                word &= high_word_mask;
            }
            masks[static_cast<size_t>(q) * words_per_mask + w] = word;
        }
    }
    return masks;
}

}  // namespace

int main() {
    int num_qubits = get_env_int("CLIFFT_NUM_QUBITS", kDefaultNumQubits);
    int clifford_depth = get_env_int("CLIFFT_CLIFFORD_DEPTH", kDefaultCliffordDepth);
    int t_gates = get_env_int("CLIFFT_T_GATES", kDefaultTGates);
    int queries_raw = get_env_int("CLIFFT_QUERIES", static_cast<int>(kDefaultQueries));
    const char* circuit_file = std::getenv("CLIFFT_CIRCUIT_FILE");

    // CLIFFT_QUERIES is always used; reject zero up front so the per-query
    // timing math never divides by zero.
    if (queries_raw < 1) {
        std::cerr << "Error: CLIFFT_QUERIES must be >= 1 (got " << queries_raw << ")\n";
        return 1;
    }
    const uint32_t queries = static_cast<uint32_t>(queries_raw);

    std::cout << "Clifft Probability Profiler\n";
    std::cout << "================\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    std::string circuit_text;
    if (circuit_file) {
        std::cout << "Circuit: " << circuit_file << "\n";
        std::cout << "Queries: " << queries << "\n\n";
        std::cout << "Loading circuit..." << std::flush;
        t0 = std::chrono::high_resolution_clock::now();
        circuit_text = read_file(circuit_file);
        t1 = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << " done (" << gen_ms << " ms, " << circuit_text.size() << " bytes)\n";
    } else {
        // Random-generation inputs are only consumed when CLIFFT_CIRCUIT_FILE
        // is unset; validate them here so loading a real file isn't blocked by
        // unrelated env-var values.
        if (num_qubits < 1) {
            std::cerr << "Error: CLIFFT_NUM_QUBITS must be >= 1 (got " << num_qubits << ")\n";
            return 1;
        }
        if (clifford_depth < 0) {
            std::cerr << "Error: CLIFFT_CLIFFORD_DEPTH must be >= 0 (got " << clifford_depth
                      << ")\n";
            return 1;
        }
        if (t_gates < 0) {
            std::cerr << "Error: CLIFFT_T_GATES must be >= 0 (got " << t_gates << ")\n";
            return 1;
        }

        std::cout << "Circuit: " << num_qubits << " qubits, " << clifford_depth
                  << " Clifford gates";
        if (t_gates > 0) {
            std::cout << ", " << t_gates << " T-gates";
        }
        std::cout << " (unitary, no measurements)\n";
        std::cout << "Queries: " << queries << "\n";
        std::cout << "(Set CLIFFT_CIRCUIT_FILE for a real file, or CLIFFT_NUM_QUBITS, "
                     "CLIFFT_CLIFFORD_DEPTH, CLIFFT_T_GATES, CLIFFT_QUERIES)\n\n";

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
    clifft::Circuit circuit = clifft::parse(circuit_text);
    t1 = std::chrono::high_resolution_clock::now();
    auto parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << parse_ms << " ms, " << circuit.nodes.size() << " ops)\n";

    // Frontend (default HIR passes -- matches clifft.compile() so we profile
    // the same hot path the user sees).
    std::cout << "Frontend (Clifford absorption)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    clifft::HirModule hir = clifft::trace(circuit);
    auto pm = clifft::default_hir_pass_manager();
    pm.run(hir);
    t1 = std::chrono::high_resolution_clock::now();
    auto trace_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << trace_ms << " ms)\n";

    // Backend (no postselection - basis_probabilities() requires pure-state evolution)
    std::cout << "Backend (bytecode generation)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    clifft::CompiledModule program = clifft::lower(hir, {});
    t1 = std::chrono::high_resolution_clock::now();
    auto lower_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t pre_opt_count = program.bytecode.size();
    std::cout << " done (" << lower_ms << " ms, " << pre_opt_count << " instructions)\n";

    // Default bytecode optimization (matches clifft.compile()).
    std::cout << "Bytecode optimization..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto bpm = clifft::default_bytecode_pass_manager();
    bpm.run(program);
    t1 = std::chrono::high_resolution_clock::now();
    auto opt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << opt_ms << " ms, " << pre_opt_count << " -> "
              << program.bytecode.size() << " instructions)\n";

    std::cout << "\nCompilation total: " << (parse_ms + trace_ms + lower_ms + opt_ms) << " ms\n";
    std::cout << "Peak rank: " << program.peak_rank << " (statevector size: 2^" << program.peak_rank
              << " = " << (1ULL << program.peak_rank) << ")\n\n";

    // Build query bitmasks. Random uniform bitstrings cover both in-support
    // and out-of-support fast paths in the amplitude walk.
    const size_t words_per_mask = (static_cast<size_t>(program.num_qubits) + 63U) / 64U;
    std::cout << "Generating " << queries << " random bitstrings (" << words_per_mask
              << " word(s) each)..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> bitmasks =
        generate_random_bitmasks(program.num_qubits, queries, words_per_mask, kSeed ^ 0xabcdef);
    t1 = std::chrono::high_resolution_clock::now();
    auto masks_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << masks_ms << " ms)\n";

    // Run basis_probabilities()
    std::cout << "Running " << queries << " probability queries..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> probs = clifft::basis_probabilities(
        program, std::span<const uint64_t>(bitmasks), queries, words_per_mask);
    t1 = std::chrono::high_resolution_clock::now();
    auto prob_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double us_per_query = (prob_ms * 1000.0) / queries;
    std::cout << " done\n";
    std::cout << "\nProbability Results:\n";
    std::cout << "  Total:       " << prob_ms << " ms\n";
    std::cout << "  Per query:   " << us_per_query << " us\n";
    std::cout << "  Queries/s:   " << static_cast<uint64_t>(queries / (prob_ms / 1000.0)) << "\n";

    // Light sanity stats. Use the values to keep them from being optimized away
    // and to give the user a way to spot regressions.
    double sum_p = 0.0;
    size_t nonzero = 0;
    double max_p = 0.0;
    for (double p : probs) {
        sum_p += p;
        if (p > 0.0) {
            ++nonzero;
        }
        if (p > max_p) {
            max_p = p;
        }
    }
    std::cout << "  Nonzero:     " << nonzero << " / " << queries << "\n";
    std::cout << "  Sum:         " << sum_p << "\n";
    std::cout << "  Max:         " << max_p << "\n";

    return 0;
}
