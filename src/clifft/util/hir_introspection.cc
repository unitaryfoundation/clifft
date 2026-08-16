#include "clifft/util/hir_introspection.h"

#include <sstream>

namespace clifft {

std::string format_pauli_mask(PauliMaskView mask) {
    MaskView x_mask = mask.x();
    MaskView z_mask = mask.z();
    bool sign = mask.sign();

    if (x_mask.is_zero() && z_mask.is_zero())
        return sign ? "-I" : "+I";

    std::string result = sign ? "-" : "+";
    bool first = true;
    const uint32_t bits = x_mask.num_words() * 64;
    for (uint32_t i = 0; i < bits; ++i) {
        bool x = x_mask.bit_get(i);
        bool z = z_mask.bit_get(i);
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
        case OpType::MEASURE:
            return "MEASURE";
        case OpType::CONDITIONAL_PAULI:
            return "CONDITIONAL_PAULI";
        case OpType::NOISE:
            return "NOISE";
        case OpType::READOUT_NOISE:
            return "READOUT_NOISE";
        case OpType::PHASE_ROTATION:
            return "PHASE_ROTATION";
        case OpType::DETECTOR:
            return "DETECTOR";
        case OpType::OBSERVABLE:
            return "OBSERVABLE";
        case OpType::EXP_VAL:
            return "EXP_VAL";
        case OpType::INSTRUMENT:
            return "INSTRUMENT";
        default:
            return "UNKNOWN";
    }
}

std::string format_hir_op(const HeisenbergOp& op, std::optional<PauliMaskView> mask) {
    std::ostringstream ss;
    switch (op.op_type()) {
        case OpType::T_GATE:
            ss << (op.is_dagger() ? "T_DAG " : "T ") << format_pauli_mask(*mask);
            break;
        case OpType::MEASURE:
            ss << "MEASURE " << format_pauli_mask(*mask) << " -> rec["
               << static_cast<uint32_t>(op.meas_record_idx()) << "]";
            if (op.is_hidden())
                ss << " (hidden)";
            break;
        case OpType::CONDITIONAL_PAULI:
            ss << "IF rec[" << static_cast<uint32_t>(op.controlling_meas()) << "] THEN "
               << format_pauli_mask(*mask);
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
        case OpType::PHASE_ROTATION:
            ss << "PHASE_ROTATION " << format_pauli_mask(*mask) << " alpha=" << op.alpha();
            break;
        case OpType::EXP_VAL:
            ss << "EXP_VAL " << format_pauli_mask(*mask) << " -> exp["
               << static_cast<uint32_t>(op.exp_val_idx()) << "]";
            break;
        case OpType::INSTRUMENT:
            ss << "INSTRUMENT " << format_pauli_mask(*mask)
               << " site=" << static_cast<uint32_t>(op.instrument_site_idx());
            break;
        case OpType::NUM_OP_TYPES:
            break;
    }
    return ss.str();
}

}  // namespace clifft
