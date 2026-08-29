#include "clifft/circuit/gate_data.h"

#include <stdexcept>
#include <string>

namespace clifft {

GateType inverse_fixed_clifford_gate(GateType gate) {
    switch (gate) {
        case GateType::S:
            return GateType::S_DAG;
        case GateType::S_DAG:
            return GateType::S;
        case GateType::SQRT_X:
            return GateType::SQRT_X_DAG;
        case GateType::SQRT_X_DAG:
            return GateType::SQRT_X;
        case GateType::SQRT_Y:
            return GateType::SQRT_Y_DAG;
        case GateType::SQRT_Y_DAG:
            return GateType::SQRT_Y;
        case GateType::C_XYZ:
            return GateType::C_ZYX;
        case GateType::C_ZYX:
            return GateType::C_XYZ;
        case GateType::C_NXYZ:
            return GateType::C_ZYNX;
        case GateType::C_NZYX:
            return GateType::C_XYNZ;
        case GateType::C_XNYZ:
            return GateType::C_ZNYX;
        case GateType::C_XYNZ:
            return GateType::C_NZYX;
        case GateType::C_ZNYX:
            return GateType::C_XNYZ;
        case GateType::C_ZYNX:
            return GateType::C_NXYZ;
        case GateType::ISWAP:
            return GateType::ISWAP_DAG;
        case GateType::ISWAP_DAG:
            return GateType::ISWAP;
        case GateType::SQRT_XX:
            return GateType::SQRT_XX_DAG;
        case GateType::SQRT_XX_DAG:
            return GateType::SQRT_XX;
        case GateType::SQRT_YY:
            return GateType::SQRT_YY_DAG;
        case GateType::SQRT_YY_DAG:
            return GateType::SQRT_YY;
        case GateType::SQRT_ZZ:
            return GateType::SQRT_ZZ_DAG;
        case GateType::SQRT_ZZ_DAG:
            return GateType::SQRT_ZZ;
        case GateType::SPP:
            return GateType::SPP_DAG;
        case GateType::SPP_DAG:
            return GateType::SPP;
        case GateType::CXSWAP:
            return GateType::SWAPCX;
        case GateType::SWAPCX:
            return GateType::CXSWAP;
        case GateType::H:
        case GateType::I:
        case GateType::II:
        case GateType::X:
        case GateType::Y:
        case GateType::Z:
        case GateType::H_XY:
        case GateType::H_YZ:
        case GateType::H_NXY:
        case GateType::H_NXZ:
        case GateType::H_NYZ:
        case GateType::CX:
        case GateType::CY:
        case GateType::CZ:
        case GateType::SWAP:
        case GateType::CZSWAP:
        case GateType::XCX:
        case GateType::XCY:
        case GateType::XCZ:
        case GateType::YCX:
        case GateType::YCY:
        case GateType::YCZ:
            return gate;
        default:
            throw std::invalid_argument("Gate does not have an explicit Clifford inverse: " +
                                        std::string(gate_name(gate)));
    }
}

}  // namespace clifft
