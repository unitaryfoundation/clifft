#include "clifft/util/introspection.h"

#include <sstream>

namespace clifft {

std::string opcode_to_str(Opcode op) {
    switch (op) {
        case Opcode::OP_FRAME_CNOT:
            return "OP_FRAME_CNOT";
        case Opcode::OP_FRAME_CZ:
            return "OP_FRAME_CZ";
        case Opcode::OP_FRAME_H:
            return "OP_FRAME_H";
        case Opcode::OP_FRAME_S:
            return "OP_FRAME_S";
        case Opcode::OP_FRAME_S_DAG:
            return "OP_FRAME_S_DAG";
        case Opcode::OP_FRAME_SWAP:
            return "OP_FRAME_SWAP";
        case Opcode::OP_ARRAY_CNOT:
            return "OP_ARRAY_CNOT";
        case Opcode::OP_ARRAY_CZ:
            return "OP_ARRAY_CZ";
        case Opcode::OP_ARRAY_SWAP:
            return "OP_ARRAY_SWAP";
        case Opcode::OP_ARRAY_MULTI_CNOT:
            return "OP_ARRAY_MULTI_CNOT";
        case Opcode::OP_ARRAY_MULTI_CZ:
            return "OP_ARRAY_MULTI_CZ";
        case Opcode::OP_ARRAY_H:
            return "OP_ARRAY_H";
        case Opcode::OP_ARRAY_S:
            return "OP_ARRAY_S";
        case Opcode::OP_ARRAY_S_DAG:
            return "OP_ARRAY_S_DAG";
        case Opcode::OP_EXPAND:
            return "OP_EXPAND";
        case Opcode::OP_ARRAY_T:
            return "OP_ARRAY_T";
        case Opcode::OP_ARRAY_T_DAG:
            return "OP_ARRAY_T_DAG";
        case Opcode::OP_EXPAND_T:
            return "OP_EXPAND_T";
        case Opcode::OP_EXPAND_T_DAG:
            return "OP_EXPAND_T_DAG";
        case Opcode::OP_ARRAY_ROT:
            return "OP_ARRAY_ROT";
        case Opcode::OP_EXPAND_ROT:
            return "OP_EXPAND_ROT";
        case Opcode::OP_ARRAY_U2:
            return "OP_ARRAY_U2";
        case Opcode::OP_ARRAY_U4:
            return "OP_ARRAY_U4";
        case Opcode::OP_MEAS_DORMANT_STATIC:
            return "OP_MEAS_DORMANT_STATIC";
        case Opcode::OP_MEAS_DORMANT_RANDOM:
            return "OP_MEAS_DORMANT_RANDOM";
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            return "OP_MEAS_ACTIVE_DIAGONAL";
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            return "OP_MEAS_ACTIVE_INTERFERE";
        case Opcode::OP_SWAP_MEAS_INTERFERE:
            return "OP_SWAP_MEAS_INTERFERE";
        case Opcode::OP_MEAS_DORMANT_STATIC_FORCED:
            return "OP_MEAS_DORMANT_STATIC_FORCED";
        case Opcode::OP_MEAS_DORMANT_RANDOM_FORCED:
            return "OP_MEAS_DORMANT_RANDOM_FORCED";
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED:
            return "OP_MEAS_ACTIVE_DIAGONAL_FORCED";
        case Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED:
            return "OP_MEAS_ACTIVE_INTERFERE_FORCED";
        case Opcode::OP_SWAP_MEAS_INTERFERE_FORCED:
            return "OP_SWAP_MEAS_INTERFERE_FORCED";
        case Opcode::OP_INSTRUMENT_ACTIVE:
            return "OP_INSTRUMENT_ACTIVE";
        case Opcode::OP_INSTRUMENT_DORMANT_STATIC:
            return "OP_INSTRUMENT_DORMANT_STATIC";
        case Opcode::OP_INSTRUMENT_EXPAND:
            return "OP_INSTRUMENT_EXPAND";
        case Opcode::OP_INSTRUMENT_DORMANT_NEGLECT:
            return "OP_INSTRUMENT_DORMANT_NEGLECT";
        case Opcode::OP_APPLY_PAULI:
            return "OP_APPLY_PAULI";
        case Opcode::OP_NOISE:
            return "OP_NOISE";
        case Opcode::OP_NOISE_BLOCK:
            return "OP_NOISE_BLOCK";
        case Opcode::OP_READOUT_NOISE:
            return "OP_READOUT_NOISE";
        case Opcode::OP_DETECTOR:
            return "OP_DETECTOR";
        case Opcode::OP_POSTSELECT:
            return "OP_POSTSELECT";
        case Opcode::OP_OBSERVABLE:
            return "OP_OBSERVABLE";
        case Opcode::OP_EXP_VAL:
            return "OP_EXP_VAL";
        default:
            return "UNKNOWN";
    }
}

std::string format_instruction(const Instruction& inst) {
    std::ostringstream ss;
    ss << opcode_to_str(inst.opcode) << " ";

    if (inst.opcode == Opcode::OP_ARRAY_ROT || inst.opcode == Opcode::OP_EXPAND_ROT) {
        ss << inst.axis_1 << " z=(" << inst.math.weight_re << ", " << inst.math.weight_im << ")";
    } else if (inst.opcode == Opcode::OP_ARRAY_U2) {
        ss << inst.axis_1 << " cp_idx=" << inst.u2.cp_idx;
    } else if (inst.opcode == Opcode::OP_ARRAY_U4) {
        ss << inst.axis_1 << ", " << inst.axis_2 << " cp_idx=" << inst.u4.cp_idx;
    } else if (inst.opcode == Opcode::OP_ARRAY_MULTI_CNOT) {
        ss << "target=" << inst.axis_1 << " ctrl_mask=0x" << std::hex << inst.multi_gate.mask
           << std::dec;
    } else if (inst.opcode == Opcode::OP_ARRAY_MULTI_CZ) {
        ss << "ctrl=" << inst.axis_1 << " target_mask=0x" << std::hex << inst.multi_gate.mask
           << std::dec;
    } else if (inst.opcode == Opcode::OP_SWAP_MEAS_INTERFERE ||
               inst.opcode == Opcode::OP_SWAP_MEAS_INTERFERE_FORCED) {
        ss << "swap(" << inst.axis_1 << "," << inst.axis_2
           << ") meas_idx=" << inst.classical.classical_idx;
        if (inst.flags & Instruction::FLAG_SIGN)
            ss << " (sign)";
    } else if (inst.opcode == Opcode::OP_INSTRUMENT_ACTIVE ||
               inst.opcode == Opcode::OP_INSTRUMENT_DORMANT_STATIC ||
               inst.opcode == Opcode::OP_INSTRUMENT_EXPAND ||
               inst.opcode == Opcode::OP_INSTRUMENT_DORMANT_NEGLECT) {
        ss << inst.axis_1 << " site=" << inst.instrument.cp_site_idx << " r=("
           << inst.instrument.r_g << ", " << inst.instrument.r_e << ")";
        if (inst.flags & Instruction::FLAG_SIGN)
            ss << " (sign)";
    } else if (is_two_axis_opcode(inst.opcode)) {
        ss << inst.axis_1 << ", " << inst.axis_2;
    } else if (is_one_axis_opcode(inst.opcode)) {
        ss << inst.axis_1;
    } else if (is_meas_opcode(inst.opcode)) {
        ss << inst.axis_1 << " -> rec[" << inst.classical.classical_idx << "]";
        if (inst.flags & Instruction::FLAG_SIGN)
            ss << " (invert)";
        if (inst.flags & Instruction::FLAG_IDENTITY)
            ss << " (identity)";
    } else if (inst.opcode == Opcode::OP_APPLY_PAULI) {
        ss << "cp_mask=" << inst.pauli.cp_mask_idx << " if rec[" << inst.pauli.condition_idx << "]";
    } else if (inst.opcode == Opcode::OP_NOISE) {
        ss << "cp_site=" << inst.pauli.cp_mask_idx;
    } else if (inst.opcode == Opcode::OP_NOISE_BLOCK) {
        ss << "sites=[" << inst.pauli.cp_mask_idx << ".."
           << (inst.pauli.cp_mask_idx + inst.pauli.condition_idx) << ")";
    } else if (inst.opcode == Opcode::OP_READOUT_NOISE) {
        ss << "cp_entry=" << inst.pauli.cp_mask_idx;
    } else if (inst.opcode == Opcode::OP_DETECTOR || inst.opcode == Opcode::OP_POSTSELECT) {
        ss << "cp_targets=" << inst.pauli.cp_mask_idx << " -> det[" << inst.pauli.condition_idx
           << "]";
    } else if (inst.opcode == Opcode::OP_OBSERVABLE) {
        ss << "cp_targets=" << inst.pauli.cp_mask_idx << " -> obs[" << inst.pauli.condition_idx
           << "]";
    } else if (inst.opcode == Opcode::OP_EXP_VAL) {
        ss << "cp=" << inst.exp_val.cp_exp_val_idx << " -> exp[" << inst.exp_val.exp_val_idx << "]";
    }
    return ss.str();
}

}  // namespace clifft
