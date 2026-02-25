// ucctrace: UCC circuit compilation and execution trace tool
//
// Reads a .stim circuit from a file or stdin, compiles it through the
// full UCC pipeline, and prints detailed diagnostics:
//   1. Parsed AST
//   2. Heisenberg IR (HIR) operations
//   3. Compiled bytecode (opcodes)
//   4. Step-by-step execution with statevector after each opcode
//   5. Final measurements
//
// The trace uses the REAL SVM execute() path with instrumentation,
// guaranteeing identical behavior to production execution.
//
// Usage:
//   ucctrace circuit.stim          # from file
//   cat circuit.stim | ucctrace    # from stdin
//   ucctrace -s 42 circuit.stim    # custom RNG seed
//   ucctrace --shots 100 circuit.stim  # multi-shot sampling

#include "ucc/backend/backend.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/svm/svm.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace ucc;

// =============================================================================
// Formatting Helpers
// =============================================================================

static const char* kSeparator =
    "========================================================================";

static std::string format_complex(std::complex<double> c) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    double re = c.real();
    double im = c.imag();
    if (std::abs(im) < 1e-15) {
        ss << re;
    } else if (std::abs(re) < 1e-15) {
        ss << im << "i";
    } else {
        ss << re << (im >= 0 ? "+" : "") << im << "i";
    }
    return ss.str();
}

static std::string format_pauli(uint64_t destab, uint64_t stab, uint32_t num_qubits) {
    std::string s;
    for (uint32_t q = 0; q < num_qubits; ++q) {
        bool x = (destab >> q) & 1;
        bool z = (stab >> q) & 1;
        if (x && z)
            s += 'Y';
        else if (x)
            s += 'X';
        else if (z)
            s += 'Z';
        else
            s += '_';
    }
    return s;
}

static std::string format_bits(uint32_t mask, uint32_t width) {
    std::string s;
    for (uint32_t i = 0; i < width; ++i) {
        s += ((mask >> i) & 1) ? '1' : '0';
    }
    return s;
}

static const char* opcode_name(Opcode op) {
    switch (op) {
        case Opcode::OP_BRANCH:
            return "BRANCH";
        case Opcode::OP_COLLIDE:
            return "COLLIDE";
        case Opcode::OP_SCALAR_PHASE:
            return "SCALAR_PHASE";
        case Opcode::OP_BRANCH_LCU:
            return "BRANCH_LCU";
        case Opcode::OP_COLLIDE_LCU:
            return "COLLIDE_LCU";
        case Opcode::OP_SCALAR_PHASE_LCU:
            return "SCALAR_PHASE_LCU";
        case Opcode::OP_MEASURE_MERGE:
            return "MEASURE_MERGE";
        case Opcode::OP_MEASURE_FILTER:
            return "MEASURE_FILTER";
        case Opcode::OP_MEASURE_DETERMINISTIC:
            return "MEASURE_DET";
        case Opcode::OP_AG_PIVOT:
            return "AG_PIVOT";
        case Opcode::OP_CONDITIONAL:
            return "CONDITIONAL";
        case Opcode::OP_INDEX_CNOT:
            return "INDEX_CNOT";
        case Opcode::OP_READOUT_NOISE:
            return "READOUT_NOISE";
        case Opcode::OP_DETECTOR:
            return "DETECTOR";
        case Opcode::OP_OBSERVABLE:
            return "OBSERVABLE";
        case Opcode::OP_POSTSELECT:
            return "POSTSELECT";
    }
    return "UNKNOWN";
}

static const char* optype_name(OpType t) {
    switch (t) {
        case OpType::T_GATE:
            return "T_GATE";
        case OpType::MEASURE:
            return "MEASURE";
        case OpType::CONDITIONAL_PAULI:
            return "CONDITIONAL_PAULI";
        case OpType::NOISE:
            return "NOISE";
        case OpType::READOUT_NOISE:
            return "READOUT_NOISE";
        case OpType::DETECTOR:
            return "DETECTOR";
        case OpType::OBSERVABLE:
            return "OBSERVABLE";
    }
    return "UNKNOWN";
}

static std::string phases_str[] = {"+1", "+i", "-1", "-i"};

// =============================================================================
// Section Printers
// =============================================================================

static void print_ast(const Circuit& circuit) {
    std::cout << kSeparator << "\n";
    std::cout << "PARSED AST  (" << circuit.nodes.size() << " nodes, " << circuit.num_qubits
              << " qubits, " << circuit.num_measurements << " measurements)\n";
    std::cout << kSeparator << "\n";

    for (size_t i = 0; i < circuit.nodes.size(); ++i) {
        const auto& node = circuit.nodes[i];
        std::cout << "  [" << i << "] " << ucc::gate_name(node.gate);
        if (node.arg != 0.0) {
            std::cout << "(" << node.arg << ")";
        }
        for (const auto& t : node.targets) {
            std::cout << " ";
            if (t.is_rec()) {
                std::cout << "rec[" << t.value() << "]";
            } else if (t.has_pauli()) {
                std::cout << t.pauli_char() << t.value();
            } else {
                std::cout << t.value();
            }
            if (t.is_inverted())
                std::cout << "!";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_hir(const HirModule& hir) {
    std::cout << kSeparator << "\n";
    std::cout << "HEISENBERG IR  (" << hir.ops.size() << " ops, " << hir.num_qubits
              << " qubits, T-count=" << hir.num_t_gates() << ", meas=" << hir.num_measurements
              << ")\n";
    std::cout << "  global_weight = " << format_complex(hir.global_weight) << "\n";
    std::cout << "  ag_matrices: " << hir.ag_matrices.size() << "\n";
    std::cout << "  noise_sites: " << hir.noise_sites.size() << "\n";
    std::cout << kSeparator << "\n";

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const auto& op = hir.ops[i];
        std::cout << "  [" << i << "] " << optype_name(op.op_type());

        uint64_t destab = static_cast<uint64_t>(op.destab_mask());
        uint64_t stab = static_cast<uint64_t>(op.stab_mask());

        switch (op.op_type()) {
            case OpType::T_GATE:
                std::cout << (op.is_dagger() ? " (T_DAG)" : " (T)");
                std::cout << "  sign=" << (op.sign() ? "-" : "+");
                std::cout << "  pauli=" << format_pauli(destab, stab, hir.num_qubits);
                break;

            case OpType::MEASURE:
                std::cout << "  meas_idx=" << static_cast<uint32_t>(op.meas_record_idx());
                std::cout << "  sign=" << (op.sign() ? "-" : "+");
                std::cout << "  pauli=" << format_pauli(destab, stab, hir.num_qubits);
                if (op.ag_matrix_idx() != AgMatrixIdx::None) {
                    std::cout << "  ag_idx=" << static_cast<uint32_t>(op.ag_matrix_idx());
                    std::cout << "  ag_ref=" << (int)op.ag_ref_outcome();
                }
                if (op.is_hidden())
                    std::cout << "  [hidden]";
                break;

            case OpType::CONDITIONAL_PAULI:
                std::cout << "  ctrl=" << static_cast<uint32_t>(op.controlling_meas());
                std::cout << "  sign=" << (op.sign() ? "-" : "+");
                std::cout << "  pauli=" << format_pauli(destab, stab, hir.num_qubits);
                if (op.use_last_outcome())
                    std::cout << "  [use_last]";
                break;

            case OpType::NOISE:
                std::cout << "  site_idx=" << static_cast<uint32_t>(op.noise_site_idx());
                break;

            case OpType::READOUT_NOISE:
                std::cout << "  entry_idx=" << static_cast<uint32_t>(op.readout_noise_idx());
                break;

            case OpType::DETECTOR:
                std::cout << "  det_idx=" << static_cast<uint32_t>(op.detector_idx());
                break;

            case OpType::OBSERVABLE:
                std::cout << "  obs_idx=" << static_cast<uint32_t>(op.observable_idx());
                break;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_bytecode(const CompiledModule& prog) {
    std::cout << kSeparator << "\n";
    std::cout << "COMPILED BYTECODE  (" << prog.bytecode.size()
              << " instructions, peak_rank=" << prog.peak_rank << ", meas=" << prog.num_measurements
              << ")\n";
    if (!prog.constant_pool.noise_schedule.empty()) {
        std::cout << "  noise_sites: " << prog.constant_pool.noise_schedule.size() << "\n";
    }
    if (!prog.constant_pool.gf2_basis.empty()) {
        std::cout << "  gf2_basis: " << prog.constant_pool.gf2_basis.size() << " vectors\n";
    }
    std::cout << kSeparator << "\n";

    for (size_t i = 0; i < prog.bytecode.size(); ++i) {
        const auto& instr = prog.bytecode[i];
        std::cout << "  [" << std::setw(3) << i << "] " << std::left << std::setw(16)
                  << opcode_name(instr.opcode) << std::right;

        switch (instr.opcode) {
            case Opcode::OP_BRANCH:
            case Opcode::OP_COLLIDE:
            case Opcode::OP_SCALAR_PHASE:
                std::cout << "  x_mask=" << format_bits(instr.branch.x_mask, prog.peak_rank);
                std::cout << "  bit=" << instr.branch.bit_index;
                std::cout << "  phase=" << phases_str[instr.base_phase_idx];
                std::cout << "  comm=" << format_bits(instr.commutation_mask, prog.peak_rank);
                if (instr.flags & Instruction::FLAG_IS_DAGGER)
                    std::cout << "  [dag]";
                break;

            case Opcode::OP_MEASURE_MERGE:
            case Opcode::OP_MEASURE_FILTER:
            case Opcode::OP_MEASURE_DETERMINISTIC:
                std::cout << "  x_mask=" << format_bits(instr.branch.x_mask, prog.peak_rank);
                std::cout << "  bit=" << instr.branch.bit_index;
                std::cout << "  phase=" << phases_str[instr.base_phase_idx];
                std::cout << "  comm=" << format_bits(instr.commutation_mask, prog.peak_rank);
                std::cout << "  ag_ref=" << (int)instr.ag_ref_outcome;
                if (instr.flags & Instruction::FLAG_HIDDEN)
                    std::cout << "  [hidden]";
                break;

            case Opcode::OP_AG_PIVOT:
                std::cout << "  ag_idx=" << instr.meta.payload_idx;
                std::cout << "  ag_ref=" << (int)instr.ag_ref_outcome;
                if (instr.flags & Instruction::FLAG_HIDDEN)
                    std::cout << "  [hidden]";
                if (instr.flags & Instruction::FLAG_REUSE_OUTCOME)
                    std::cout << "  [reuse]";
                break;

            case Opcode::OP_CONDITIONAL:
                std::cout << "  ctrl_meas=" << instr.meta.controlling_meas;
                std::cout << "  phase=" << phases_str[instr.base_phase_idx];
                if (instr.flags & Instruction::FLAG_USE_LAST_OUTCOME)
                    std::cout << "  [use_last]";
                break;

            case Opcode::OP_READOUT_NOISE:
                std::cout << "  meas=" << instr.readout.meas_idx;
                std::cout << "  prob=" << instr.readout.prob;
                break;

            case Opcode::OP_DETECTOR:
                std::cout << "  target_idx=" << instr.detector.target_idx;
                break;

            case Opcode::OP_OBSERVABLE:
                std::cout << "  target_idx=" << instr.observable.target_idx;
                std::cout << "  obs=" << instr.observable.obs_idx;
                break;

            default:
                break;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// =============================================================================
// Trace Output (uses real SVM execute_traced)
// =============================================================================

static void print_trace(const CompiledModule& /*prog*/, const std::vector<TraceEntry>& trace) {
    std::cout << kSeparator << "\n";
    std::cout << "TRACED EXECUTION  (" << trace.size() << " steps, uses real SVM path)\n";
    std::cout << kSeparator << "\n";

    for (const auto& entry : trace) {
        std::cout << "  [" << std::setw(3) << entry.pc << "] " << opcode_name(entry.opcode);
        if (!entry.detail.empty()) {
            std::cout << " -> " << entry.detail;
        }
        std::cout << "\n";

        // Print non-zero amplitudes
        uint32_t rank = entry.rank_after;
        uint64_t size = entry.v.size();
        bool any_nonzero = false;
        for (uint64_t a = 0; a < size; ++a) {
            if (std::abs(entry.v[a]) > 1e-15) {
                if (!any_nonzero) {
                    std::cout << "    v[] = {";
                    any_nonzero = true;
                } else {
                    std::cout << ", ";
                }
                std::cout << format_bits(static_cast<uint32_t>(a), rank) << ":"
                          << format_complex(entry.v[a]);
            }
        }
        if (any_nonzero) {
            std::cout << "}\n";
        } else {
            std::cout << "    v[] = {all zero}\n";
        }
        std::cout << "    frame: destab_signs=0x" << std::hex << entry.destab_signs
                  << " stab_signs=0x" << entry.stab_signs << std::dec << "\n\n";
    }
}

static void print_measurements(const SchrodingerState& state, uint32_t num_meas) {
    std::cout << kSeparator << "\n";
    std::cout << "MEASUREMENTS  (" << num_meas << " total)\n";
    std::cout << kSeparator << "\n";
    if (num_meas == 0) {
        std::cout << "  (none)\n";
    } else {
        std::cout << "  ";
        for (uint32_t i = 0; i < num_meas; ++i) {
            std::cout << (int)state.meas_record[i];
        }
        std::cout << "\n";
        for (uint32_t i = 0; i < num_meas; ++i) {
            std::cout << "  meas[" << i << "] = " << (int)state.meas_record[i] << "\n";
        }
    }

    if (!state.det_record.empty()) {
        std::cout << "\n  Detectors: ";
        for (size_t i = 0; i < state.det_record.size(); ++i) {
            std::cout << (int)state.det_record[i];
        }
        std::cout << "\n";
    }
    if (!state.obs_record.empty()) {
        std::cout << "\n  Observables: ";
        for (size_t i = 0; i < state.obs_record.size(); ++i) {
            std::cout << (int)state.obs_record[i];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_statevector(const SchrodingerState& state, const CompiledModule& prog) {
    if (!prog.constant_pool.final_tableau.has_value()) {
        std::cout << "  (no final tableau -- statevector expansion unavailable)\n";
        return;
    }

    auto sv =
        get_statevector(state, prog.constant_pool.gf2_basis,
                        prog.constant_pool.final_tableau.value(), prog.constant_pool.global_weight);

    uint32_t n = static_cast<uint32_t>(prog.constant_pool.final_tableau->num_qubits);
    std::cout << kSeparator << "\n";
    std::cout << "FINAL STATEVECTOR  (2^" << n << " = " << sv.size() << " amplitudes)\n";
    std::cout << kSeparator << "\n";

    for (size_t i = 0; i < sv.size(); ++i) {
        if (std::abs(sv[i]) > 1e-12) {
            std::cout << "  |";
            for (int q = static_cast<int>(n) - 1; q >= 0; --q) {
                std::cout << ((i >> q) & 1);
            }
            std::cout << "> : " << format_complex(sv[i]);
            std::cout << "  (prob=" << std::fixed << std::setprecision(6) << std::norm(sv[i])
                      << ")\n";
        }
    }
    std::cout << "\n";
}

// =============================================================================
// Multi-Shot Sampling Mode
// =============================================================================

static void run_sampling(const CompiledModule& prog, uint32_t shots, uint64_t seed) {
    std::cout << kSeparator << "\n";
    std::cout << "SAMPLING  (" << shots << " shots, seed=" << seed << ")\n";
    std::cout << kSeparator << "\n";

    auto result = sample(prog, shots, seed);

    uint32_t nm = prog.num_measurements;
    uint32_t nd = prog.num_detectors;
    uint32_t no = prog.num_observables;

    uint32_t show = std::min(shots, 20u);
    for (uint32_t s = 0; s < show; ++s) {
        std::cout << "  shot " << std::setw(4) << s << ": meas=";
        for (uint32_t m = 0; m < nm; ++m) {
            std::cout << (int)result.measurements[s * nm + m];
        }
        if (nd > 0) {
            std::cout << "  det=";
            for (uint32_t d = 0; d < nd; ++d) {
                std::cout << (int)result.detectors[s * nd + d];
            }
        }
        if (no > 0) {
            std::cout << "  obs=";
            for (uint32_t o = 0; o < no; ++o) {
                std::cout << (int)result.observables[s * no + o];
            }
        }
        std::cout << "\n";
    }
    if (shots > show) {
        std::cout << "  ... (" << (shots - show) << " more shots)\n";
    }

    if (nm > 0 && shots > 1) {
        std::cout << "\n  Marginal probabilities (P(1)):";
        for (uint32_t m = 0; m < nm; ++m) {
            uint32_t ones = 0;
            for (uint32_t s = 0; s < shots; ++s) {
                ones += result.measurements[s * nm + m];
            }
            std::cout << "\n    meas[" << m << "] = " << std::fixed << std::setprecision(4)
                      << static_cast<double>(ones) / shots;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// =============================================================================
// Main
// =============================================================================

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS] [circuit.stim]\n";
    std::cerr << "\nReads a .stim circuit from file or stdin and traces compilation + execution.\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  -s SEED    RNG seed (default: 0)\n";
    std::cerr << "  --shots N  Run N shots in sampling mode (skips trace)\n";
    std::cerr << "  --no-trace Skip step-by-step trace (show only AST/HIR/bytecode)\n";
    std::cerr << "  --version  Print version and exit\n";
    std::cerr << "  -h         Print this help\n";
}

int main(int argc, char** argv) {
    uint64_t seed = 0;
    uint32_t shots = 0;
    bool do_trace = true;
    const char* filepath = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--shots") == 0 && i + 1 < argc) {
            shots = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            do_trace = false;
        } else if (std::strcmp(argv[i], "--version") == 0) {
            std::cout << "ucctrace" << "\n";
            return 0;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        } else {
            filepath = argv[i];
        }
    }

    std::string input;
    if (filepath) {
        std::ifstream f(filepath);
        if (!f) {
            std::cerr << "Error: cannot open '" << filepath << "'\n";
            return 1;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        input = ss.str();
    } else {
        if (isatty(fileno(stdin))) {
            std::cerr << "Reading from stdin (Ctrl-D to end)...\n";
        }
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        input = ss.str();
    }

    if (input.empty()) {
        std::cerr << "Error: empty input\n";
        return 1;
    }

    try {
        Circuit circuit = parse(input);
        print_ast(circuit);

        HirModule hir = trace(circuit);
        print_hir(hir);

        CompiledModule prog = lower(hir);
        print_bytecode(prog);

        if (shots > 0) {
            run_sampling(prog, shots, seed);
        } else {
            SchrodingerState state(prog.peak_rank, prog.num_measurements, prog.num_detectors,
                                   prog.num_observables, seed);
            if (do_trace) {
                std::vector<TraceEntry> trace_log;
                execute_traced(prog, state, trace_log);
                print_trace(prog, trace_log);
            } else {
                execute(prog, state);
            }
            print_measurements(state, prog.num_measurements);

            if (prog.num_measurements == 0 || prog.constant_pool.final_tableau.has_value()) {
                print_statevector(state, prog);
            }
        }

    } catch (const ParseError& e) {
        std::cerr << "Parse error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
