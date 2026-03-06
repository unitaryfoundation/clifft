#include "ucc/util/introspection.h"

#include <sstream>

namespace ucc {

std::string format_pauli_mask(const HeisenbergOp& op) {
    uint64_t x_bits = static_cast<uint64_t>(op.destab_mask());
    uint64_t z_bits = static_cast<uint64_t>(op.stab_mask());
    bool sign = op.sign();

    if (x_bits == 0 && z_bits == 0)
        return sign ? "-I" : "+I";

    std::string result = sign ? "-" : "+";
    bool first = true;
    for (int i = 0; i < static_cast<int>(kStimWidth); ++i) {
        bool x = (x_bits >> i) & 1;
        bool z = (z_bits >> i) & 1;
        if (x || z) {
            if (!first)
                result += "*";
            if (x && z)
                result += "Y" + std::to_string(i);
            else if (x)
                result += "X" + std::to_string(i);
            else
                result += "Z" + std::to_string(i);
            first = false;
        }
    }
    return result;
}

std::string op_type_to_str(OpType type) {
    switch (type) {
        case OpType::T_GATE:
            return "T_GATE";
        case OpType::CLIFFORD_PHASE:
            return "CLIFFORD_PHASE";
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
        default:
            return "UNKNOWN";
    }
}

std::string format_hir_op(const HeisenbergOp& op) {
    std::ostringstream ss;
    switch (op.op_type()) {
        case OpType::T_GATE:
            ss << (op.is_dagger() ? "T_DAG " : "T ") << format_pauli_mask(op);
            break;
        case OpType::CLIFFORD_PHASE:
            ss << (op.is_dagger() ? "S_DAG " : "S ") << format_pauli_mask(op);
            break;
        case OpType::MEASURE:
            ss << "MEASURE " << format_pauli_mask(op) << " -> rec["
               << static_cast<uint32_t>(op.meas_record_idx()) << "]";
            if (op.is_hidden())
                ss << " (hidden)";
            break;
        case OpType::CONDITIONAL_PAULI:
            ss << "IF rec["
               << (op.use_last_outcome()
                       ? std::string("last")
                       : std::to_string(static_cast<uint32_t>(op.controlling_meas())))
               << "] THEN " << format_pauli_mask(op);
            break;
        case OpType::NOISE:
            ss << "NOISE site=" << static_cast<uint32_t>(op.noise_site_idx());
            break;
        case OpType::READOUT_NOISE:
            ss << "READOUT_NOISE entry=" << static_cast<uint32_t>(op.readout_noise_idx());
            break;
        case OpType::DETECTOR:
            ss << "DETECTOR target_list=" << static_cast<uint32_t>(op.detector_idx());
            break;
        case OpType::OBSERVABLE:
            ss << "OBSERVABLE index=" << static_cast<uint32_t>(op.observable_idx())
               << " target_list=" << op.observable_target_list_idx();
            break;
    }
    return ss.str();
}

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
        case Opcode::OP_ARRAY_H:
            return "OP_ARRAY_H";
        case Opcode::OP_ARRAY_S:
            return "OP_ARRAY_S";
        case Opcode::OP_ARRAY_S_DAG:
            return "OP_ARRAY_S_DAG";
        case Opcode::OP_EXPAND:
            return "OP_EXPAND";
        case Opcode::OP_PHASE_T:
            return "OP_PHASE_T";
        case Opcode::OP_PHASE_T_DAG:
            return "OP_PHASE_T_DAG";
        case Opcode::OP_MEAS_DORMANT_STATIC:
            return "OP_MEAS_DORMANT_STATIC";
        case Opcode::OP_MEAS_DORMANT_RANDOM:
            return "OP_MEAS_DORMANT_RANDOM";
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            return "OP_MEAS_ACTIVE_DIAGONAL";
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            return "OP_MEAS_ACTIVE_INTERFERE";
        case Opcode::OP_APPLY_PAULI:
            return "OP_APPLY_PAULI";
        case Opcode::OP_NOISE:
            return "OP_NOISE";
        case Opcode::OP_READOUT_NOISE:
            return "OP_READOUT_NOISE";
        case Opcode::OP_DETECTOR:
            return "OP_DETECTOR";
        case Opcode::OP_POSTSELECT:
            return "OP_POSTSELECT";
        case Opcode::OP_OBSERVABLE:
            return "OP_OBSERVABLE";
        default:
            return "UNKNOWN";
    }
}

bool is_two_axis_opcode(Opcode op) {
    return op == Opcode::OP_FRAME_CNOT || op == Opcode::OP_FRAME_CZ ||
           op == Opcode::OP_FRAME_SWAP || op == Opcode::OP_ARRAY_CNOT ||
           op == Opcode::OP_ARRAY_CZ || op == Opcode::OP_ARRAY_SWAP;
}

bool is_one_axis_opcode(Opcode op) {
    return op == Opcode::OP_FRAME_H || op == Opcode::OP_FRAME_S || op == Opcode::OP_FRAME_S_DAG ||
           op == Opcode::OP_ARRAY_H || op == Opcode::OP_ARRAY_S || op == Opcode::OP_ARRAY_S_DAG ||
           op == Opcode::OP_EXPAND || op == Opcode::OP_PHASE_T || op == Opcode::OP_PHASE_T_DAG;
}

bool is_meas_opcode(Opcode op) {
    return op == Opcode::OP_MEAS_DORMANT_STATIC || op == Opcode::OP_MEAS_DORMANT_RANDOM ||
           op == Opcode::OP_MEAS_ACTIVE_DIAGONAL || op == Opcode::OP_MEAS_ACTIVE_INTERFERE;
}

std::string format_instruction(const Instruction& inst) {
    std::ostringstream ss;
    ss << opcode_to_str(inst.opcode) << " ";

    if (is_two_axis_opcode(inst.opcode)) {
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
    } else if (inst.opcode == Opcode::OP_READOUT_NOISE) {
        ss << "cp_entry=" << inst.pauli.cp_mask_idx;
    } else if (inst.opcode == Opcode::OP_DETECTOR || inst.opcode == Opcode::OP_POSTSELECT) {
        ss << "cp_targets=" << inst.pauli.cp_mask_idx << " -> det[" << inst.pauli.condition_idx
           << "]";
    } else if (inst.opcode == Opcode::OP_OBSERVABLE) {
        ss << "cp_targets=" << inst.pauli.cp_mask_idx << " -> obs[" << inst.pauli.condition_idx
           << "]";
    }
    return ss.str();
}

}  // namespace ucc
