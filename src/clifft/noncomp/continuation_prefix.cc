#include "clifft/noncomp/continuation_prefix.h"

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace clifft {

namespace {

[[nodiscard]] bool same_bits(double a, double b) {
    return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
}

[[nodiscard]] bool same_complex(std::complex<double> a, std::complex<double> b) {
    return same_bits(a.real(), b.real()) && same_bits(a.imag(), b.imag());
}

[[nodiscard]] bool same_mask(const PauliMaskArena& a, PauliMaskHandle a_handle,
                             const PauliMaskArena& b, PauliMaskHandle b_handle) {
    if (a_handle == kNoMask || b_handle == kNoMask) {
        return a_handle == b_handle;
    }
    const size_t a_idx = static_cast<size_t>(a_handle);
    const size_t b_idx = static_cast<size_t>(b_handle);
    if (a_idx >= a.size() || b_idx >= b.size() || a.num_words() != b.num_words()) {
        return false;
    }
    const PauliMaskView a_mask = a.at(a_handle);
    const PauliMaskView b_mask = b.at(b_handle);
    return a_mask.x() == b_mask.x() && a_mask.z() == b_mask.z() && a_mask.sign() == b_mask.sign();
}

[[nodiscard]] bool same_u2(const ConstantPool& a, const ConstantPool& b, uint32_t idx) {
    if (idx >= a.fused_u2_nodes.size() || idx >= b.fused_u2_nodes.size()) {
        return false;
    }
    const FusedU2Node& x = a.fused_u2_nodes[idx];
    const FusedU2Node& y = b.fused_u2_nodes[idx];
    for (size_t state = 0; state < 4; ++state) {
        for (size_t cell = 0; cell < 4; ++cell) {
            if (!same_complex(x.matrices[state][cell], y.matrices[state][cell])) {
                return false;
            }
        }
        if (!same_complex(x.gamma_multipliers[state], y.gamma_multipliers[state]) ||
            x.out_states[state] != y.out_states[state]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_u4(const ConstantPool& a, const ConstantPool& b, uint32_t idx) {
    if (idx >= a.fused_u4_nodes.size() || idx >= b.fused_u4_nodes.size()) {
        return false;
    }
    const FusedU4Node& x = a.fused_u4_nodes[idx];
    const FusedU4Node& y = b.fused_u4_nodes[idx];
    for (size_t state = 0; state < 16; ++state) {
        for (size_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < 4; ++col) {
                if (!same_complex(x.entries[state].matrix[row][col],
                                  y.entries[state].matrix[row][col])) {
                    return false;
                }
            }
        }
        if (!same_complex(x.entries[state].gamma_multiplier, y.entries[state].gamma_multiplier) ||
            x.entries[state].out_state != y.entries[state].out_state) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_noise_site(const ConstantPool& a, const ConstantPool& b, uint32_t idx) {
    if (idx >= a.noise_sites.size() || idx >= b.noise_sites.size() ||
        idx >= a.noise_hazards.size() || idx >= b.noise_hazards.size()) {
        return false;
    }
    if (!same_bits(a.noise_hazards[idx], b.noise_hazards[idx])) {
        return false;
    }
    const NoiseSite& x = a.noise_sites[idx];
    const NoiseSite& y = b.noise_sites[idx];
    if (x.channels.size() != y.channels.size()) {
        return false;
    }
    for (size_t i = 0; i < x.channels.size(); ++i) {
        if (!same_bits(x.channels[i].prob, y.channels[i].prob) ||
            !same_mask(a.noise_channel_masks, x.channels[i].mask, b.noise_channel_masks,
                       y.channels[i].mask)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_noise_block(const ConstantPool& a, const ConstantPool& b, uint32_t start,
                                    uint32_t count) {
    const uint64_t end = static_cast<uint64_t>(start) + count;
    if (end > a.noise_sites.size() || end > b.noise_sites.size()) {
        return false;
    }
    for (uint64_t idx = start; idx < end; ++idx) {
        if (!same_noise_site(a, b, static_cast<uint32_t>(idx))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_readout(const ConstantPool& a, const ConstantPool& b, uint32_t idx) {
    if (idx >= a.readout_noise.size() || idx >= b.readout_noise.size()) {
        return false;
    }
    const ReadoutNoiseEntry& x = a.readout_noise[idx];
    const ReadoutNoiseEntry& y = b.readout_noise[idx];
    return x.meas_idx == y.meas_idx && same_bits(x.prob_zero_to_one, y.prob_zero_to_one) &&
           same_bits(x.prob_one_to_zero, y.prob_one_to_zero);
}

[[nodiscard]] bool same_probabilities(const InstrumentProbabilities& a,
                                      const InstrumentProbabilities& b) {
    for (size_t source = 0; source < 2; ++source) {
        if (!same_bits(a.p_fire[source], b.p_fire[source])) {
            return false;
        }
        for (size_t destination = 0; destination < 2; ++destination) {
            if (!same_bits(a.p_computational_dest[source][destination],
                           b.p_computational_dest[source][destination])) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool same_instrument(const ConstantPool& a, const ConstantPool& b, uint32_t idx) {
    if (idx >= a.instrument_sites.size() || idx >= b.instrument_sites.size()) {
        return false;
    }
    const CompiledInstrumentSite& x = a.instrument_sites[idx];
    const CompiledInstrumentSite& y = b.instrument_sites[idx];
    return x.site_id == y.site_id && same_probabilities(x.probabilities, y.probabilities) &&
           same_mask(a.instrument_destination_flip_masks, x.destination_flip_mask,
                     b.instrument_destination_flip_masks, y.destination_flip_mask);
}

[[nodiscard]] bool same_instruction(const Instruction& continuation, const Instruction& executed) {
    if (std::memcmp(&continuation, &executed, sizeof(Instruction)) == 0) {
        return true;
    }
    const std::optional<Opcode> forced = forced_measurement_opcode(continuation.opcode);
    if (!forced.has_value() || *forced != executed.opcode) {
        return false;
    }
    Instruction swapped = continuation;
    swapped.opcode = executed.opcode;
    return std::memcmp(&swapped, &executed, sizeof(Instruction)) == 0;
}

[[nodiscard]] bool same_referenced_constants(const Instruction& instr, const ConstantPool& a,
                                             const ConstantPool& b) {
    switch (instr.opcode) {
        case Opcode::OP_ARRAY_U2:
            return same_u2(a, b, instr.u2.cp_idx);
        case Opcode::OP_ARRAY_U4:
            return same_u4(a, b, instr.u4.cp_idx);
        case Opcode::OP_APPLY_PAULI:
            return same_mask(a.pauli_masks, static_cast<PauliMaskHandle>(instr.pauli.cp_mask_idx),
                             b.pauli_masks, static_cast<PauliMaskHandle>(instr.pauli.cp_mask_idx));
        case Opcode::OP_NOISE:
            return same_noise_site(a, b, instr.pauli.cp_mask_idx);
        case Opcode::OP_NOISE_BLOCK:
            return same_noise_block(a, b, instr.pauli.cp_mask_idx, instr.pauli.condition_idx);
        case Opcode::OP_READOUT_NOISE:
            return same_readout(a, b, instr.pauli.cp_mask_idx);
        case Opcode::OP_DETECTOR:
        case Opcode::OP_POSTSELECT:
            return instr.pauli.cp_mask_idx < a.detector_targets.size() &&
                   instr.pauli.cp_mask_idx < b.detector_targets.size() &&
                   a.detector_targets[instr.pauli.cp_mask_idx] ==
                       b.detector_targets[instr.pauli.cp_mask_idx];
        case Opcode::OP_OBSERVABLE:
            return instr.pauli.cp_mask_idx < a.observable_targets.size() &&
                   instr.pauli.cp_mask_idx < b.observable_targets.size() &&
                   a.observable_targets[instr.pauli.cp_mask_idx] ==
                       b.observable_targets[instr.pauli.cp_mask_idx];
        case Opcode::OP_EXP_VAL:
            return same_mask(
                a.exp_val_masks, static_cast<PauliMaskHandle>(instr.exp_val.cp_exp_val_idx),
                b.exp_val_masks, static_cast<PauliMaskHandle>(instr.exp_val.cp_exp_val_idx));
        case Opcode::OP_INSTRUMENT_ACTIVE:
        case Opcode::OP_INSTRUMENT_DORMANT_STATIC:
        case Opcode::OP_INSTRUMENT_EXPAND:
        case Opcode::OP_INSTRUMENT_DORMANT_NEGLECT:
            return same_instrument(a, b, instr.instrument.cp_site_idx);

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
        case Opcode::OP_EXPAND:
        case Opcode::OP_EXPAND_T:
        case Opcode::OP_EXPAND_T_DAG:
        case Opcode::OP_EXPAND_ROT:
        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
        case Opcode::OP_MEAS_DORMANT_STATIC_FORCED:
        case Opcode::OP_MEAS_DORMANT_RANDOM_FORCED:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED:
        case Opcode::OP_SWAP_MEAS_INTERFERE_FORCED:
            return true;
        case Opcode::NUM_OPCODES:
            return false;
    }
    return false;
}

}  // namespace

void validate_continuation_prefix(const CompiledModule& continuation,
                                  const CompiledModule& executed, uint32_t prefix_end) {
    if (continuation.num_qubits != executed.num_qubits) {
        throw std::logic_error(
            "sample_noncomputational: continuation prefix changed the module's qubit count");
    }
    if (prefix_end > continuation.bytecode.size() || prefix_end > executed.bytecode.size()) {
        throw std::logic_error("sample_noncomputational: continuation prefix length " +
                               std::to_string(prefix_end) +
                               " exceeds the compiled bytecode being compared");
    }
    for (uint32_t i = 0; i < prefix_end; ++i) {
        if (!same_instruction(continuation.bytecode[i], executed.bytecode[i])) {
            throw std::logic_error(
                "sample_noncomputational: continuation bytecode prefix diverged at instruction " +
                std::to_string(i));
        }
        if (!same_referenced_constants(continuation.bytecode[i], continuation.constant_pool,
                                       executed.constant_pool)) {
            throw std::logic_error(
                "sample_noncomputational: continuation constant-pool prefix diverged at "
                "instruction " +
                std::to_string(i));
        }
    }
}

}  // namespace clifft
