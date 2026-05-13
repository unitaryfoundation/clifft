// Exact measurement-record probability queries.
//
// probability_of() runs the compiled program once per requested record,
// forcing each measurement to the user-supplied outcome and accumulating
// log(prob_b / total) into state.forced_log_probability under the same
// dust-clamping policy sample_branch() uses. A bytecode rewrite swaps
// each OP_MEAS_* opcode for its OP_MEAS_*_FORCED sibling; the rewrite
// runs on a private shallow copy so the user's CompiledModule is
// untouched.
//
// See svm_forced_kernels.h for the forced-kernel contract and
// svm_kernels.inl for the dispatcher wiring.

#include "clifft/svm/svm.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace clifft {
namespace {

[[nodiscard]] bool is_unsupported_probability_of_opcode(Opcode opcode) {
    // Allowed: all gate / expand / array opcodes, EXP_VAL probes, feedback
    // (OP_APPLY_PAULI), and any sampling-mode measurement opcode (rewritten
    // to its FORCED sibling before execution). Forced opcodes shouldn't
    // appear in user-compiled programs; reject defensively so the rewrite
    // is the only path that produces them.
    switch (opcode) {
        case Opcode::OP_FRAME_CNOT:
        case Opcode::OP_FRAME_CZ:
        case Opcode::OP_FRAME_H:
        case Opcode::OP_FRAME_S:
        case Opcode::OP_FRAME_S_DAG:
        case Opcode::OP_FRAME_SWAP:
        case Opcode::OP_ARRAY_CNOT:
        case Opcode::OP_ARRAY_CZ:
        case Opcode::OP_ARRAY_SWAP:
        case Opcode::OP_ARRAY_MULTI_CNOT:
        case Opcode::OP_ARRAY_MULTI_CZ:
        case Opcode::OP_ARRAY_H:
        case Opcode::OP_ARRAY_S:
        case Opcode::OP_ARRAY_S_DAG:
        case Opcode::OP_ARRAY_T:
        case Opcode::OP_ARRAY_T_DAG:
        case Opcode::OP_ARRAY_ROT:
        case Opcode::OP_ARRAY_U2:
        case Opcode::OP_ARRAY_U4:
        case Opcode::OP_EXPAND:
        case Opcode::OP_EXPAND_T:
        case Opcode::OP_EXPAND_T_DAG:
        case Opcode::OP_EXPAND_ROT:
        case Opcode::OP_EXP_VAL:
        case Opcode::OP_APPLY_PAULI:
        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
            return false;

        case Opcode::OP_MEAS_DORMANT_STATIC_FORCED:
        case Opcode::OP_MEAS_DORMANT_RANDOM_FORCED:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED:
        case Opcode::OP_SWAP_MEAS_INTERFERE_FORCED:
        case Opcode::OP_NOISE:
        case Opcode::OP_NOISE_BLOCK:
        case Opcode::OP_READOUT_NOISE:
        case Opcode::OP_DETECTOR:
        case Opcode::OP_POSTSELECT:
        case Opcode::OP_OBSERVABLE:
        case Opcode::NUM_OPCODES:
            return true;
    }
    throw std::invalid_argument("probability_of() encountered an unknown bytecode opcode");
}

void assert_probability_of_program_is_supported(const CompiledModule& program) {
    for (const auto& instr : program.bytecode) {
        if (is_unsupported_probability_of_opcode(instr.opcode)) {
            throw std::invalid_argument(
                "probability_of() requires pure-state evolution with measurements: noise, "
                "feedback-free detectors, observables, and post-selection are not supported. "
                "Forced-measurement opcodes are reserved for the internal rewrite and must not "
                "appear in user-compiled programs.");
        }
    }
}

// Replace each sampling-mode measurement opcode with its FORCED sibling.
// The bytecode otherwise stays identical; classical_idx, axes, sign flag,
// and FLAG_IDENTITY are preserved.
void rewrite_for_forced_execution(std::vector<Instruction>& bytecode) {
    for (auto& instr : bytecode) {
        switch (instr.opcode) {
            case Opcode::OP_MEAS_DORMANT_STATIC:
                instr.opcode = Opcode::OP_MEAS_DORMANT_STATIC_FORCED;
                break;
            case Opcode::OP_MEAS_DORMANT_RANDOM:
                instr.opcode = Opcode::OP_MEAS_DORMANT_RANDOM_FORCED;
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
                instr.opcode = Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED;
                break;
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
                instr.opcode = Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED;
                break;
            case Opcode::OP_SWAP_MEAS_INTERFERE:
                instr.opcode = Opcode::OP_SWAP_MEAS_INTERFERE_FORCED;
                break;
            default:
                break;
        }
    }
}

}  // namespace

std::vector<double> probability_of(const CompiledModule& program, std::span<const uint8_t> records,
                                   size_t num_records) {
    assert_probability_of_program_is_supported(program);
    if (program.num_measurements == 0) {
        throw std::invalid_argument(
            "probability_of() requires a program with at least one measurement; use "
            "clifft::probabilities() for unitary circuits.");
    }
    // Hidden measurement slots (e.g. from R / reset gates lowered to
    // measure + classical-feedback Pauli) would be indexed past the
    // user-visible portion of the forced record. Supporting them would
    // require marginalizing over the hidden outcomes; out of scope for
    // now, so reject programs that have any.
    if (program.total_meas_slots != program.num_measurements) {
        throw std::invalid_argument(
            "probability_of() does not yet support programs with hidden measurement slots "
            "(e.g. R / reset gates). Compile without resets, or use sample() to marginalize.");
    }
    if (records.size() != num_records * program.num_measurements) {
        throw std::invalid_argument(
            "probability_of() record buffer length must equal "
            "num_records * program.num_measurements");
    }
    // Each record byte represents one measurement outcome and must be 0 or 1.
    // The forced kernels would otherwise XOR the byte into the abstract
    // branch index and produce garbage probabilities for out-of-range values.
    for (uint8_t byte : records) {
        if (byte != 0 && byte != 1) {
            throw std::invalid_argument(
                "probability_of() record bytes must be 0 or 1; each represents one "
                "measurement outcome.");
        }
    }

    // Private shallow copy of the program. Only the bytecode vector is
    // rewritten; the constant pool and metadata are shared by value-copy
    // of POD members. The user's CompiledModule is unmodified.
    CompiledModule patched = program;
    rewrite_for_forced_execution(patched.bytecode);

    SchrodingerState state({.peak_rank = patched.peak_rank,
                            .num_measurements = patched.total_meas_slots,
                            .num_qubits = patched.num_qubits,
                            .num_detectors = patched.num_detectors,
                            .num_observables = patched.num_observables,
                            .num_exp_vals = patched.num_exp_vals,
                            .seed = uint64_t{0}});

    const size_t record_stride = program.num_measurements;
    std::vector<double> log_probs;
    log_probs.reserve(num_records);

    for (size_t i = 0; i < num_records; ++i) {
        state.reset();
        state.forced_record = records.subspan(i * record_stride, record_stride);
        // reset() restored forced_log_probability and forced_reachable to
        // their defaults; no need to clear them again.
        execute(patched, state);
        if (state.forced_reachable) {
            log_probs.push_back(state.forced_log_probability);
        } else {
            log_probs.push_back(-std::numeric_limits<double>::infinity());
        }
    }
    return log_probs;
}

}  // namespace clifft
