#pragma once

// Gate type enumeration and metadata for the circuit AST.
//
// This defines all gates supported by the UCC parser, including:
// - Clifford gates (H, S, X, CX, etc.)
// - Non-Clifford gates (T, T_DAG)
// - Measurements (M, MX, MPP, MR, MRX)
// - Resets (R, RX) - decomposed by parser into M + feedback
// - Annotations (TICK)
// - Classical feedback (uses CX/CZ with rec targets)

#include <cstdint>
#include <string_view>

namespace ucc {

enum class GateType : uint16_t {
    // Single-qubit Clifford gates
    H,
    S,
    S_DAG,
    X,
    Y,
    Z,

    // Additional single-qubit Cliffords
    SQRT_X,
    SQRT_X_DAG,
    SQRT_Y,
    SQRT_Y_DAG,
    H_XY,
    H_YZ,
    H_NXY,  // Stim: H but in the -X,Y plane
    H_NXZ,  // Stim: same as H_XZ but negated
    H_NYZ,  // Stim: H but in the -Y,Z plane
    C_XYZ,  // Period-3 Clifford rotations
    C_ZYX,
    C_NXYZ,
    C_NZYX,
    C_XNYZ,
    C_XYNZ,
    C_ZNYX,
    C_ZYNX,

    // Non-Clifford gates
    T,
    T_DAG,

    // Two-qubit Clifford gates
    CX,
    CY,
    CZ,
    SWAP,
    ISWAP,
    ISWAP_DAG,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    CXSWAP,
    CZSWAP,
    SWAPCX,
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,

    // Measurements
    M,    // Z-basis measurement
    MX,   // X-basis measurement
    MY,   // Y-basis measurement
    MR,   // Measure + reset (Z-basis) - result visible
    MRX,  // Measure + reset (X-basis) - result visible
    MPP,  // Multi-Pauli measurement
    MXX,  // Pair measurement in XX basis (desugars to MPP)
    MYY,  // Pair measurement in YY basis (desugars to MPP)
    MZZ,  // Pair measurement in ZZ basis (desugars to MPP)

    // Resets (no visible measurement)
    R,    // Reset to |0> (Z-basis)
    RX,   // Reset to |+> (X-basis)
    RY,   // Reset to |+i> (Y-basis)
    MRY,  // Measure + reset (Y-basis) - result visible

    // Deterministic padding
    MPAD,  // Classical measurement padding (target is 0 or 1)

    // Identity no-ops (parsed for validation, never emitted to AST)
    I,         // Single-qubit identity
    II,        // Two-qubit identity
    I_ERROR,   // Single-qubit identity error (no-op)
    II_ERROR,  // Two-qubit identity error (no-op)

    // Noise channels
    X_ERROR,      // Single-qubit X error
    Y_ERROR,      // Single-qubit Y error
    Z_ERROR,      // Single-qubit Z error
    DEPOLARIZE1,  // Single-qubit depolarizing channel
    DEPOLARIZE2,  // Two-qubit depolarizing channel

    // Multi-parameter noise channels
    PAULI_CHANNEL_1,  // 3-arg: P(X), P(Y), P(Z)
    PAULI_CHANNEL_2,  // 15-arg: P(IX), P(IY), ..., P(ZZ)

    // Synthetic gates (emitted by parser, not in input syntax)
    READOUT_NOISE,  // Classical bit-flip on measurement result

    // QEC annotations
    DETECTOR,            // Detector declaration
    OBSERVABLE_INCLUDE,  // Observable accumulator

    // Annotations
    TICK,  // Timing layer marker (no-op)

    // Sentinel for unknown/unsupported gates
    UNKNOWN,
};

// Gate arity classification.
enum class GateArity : uint8_t {
    SINGLE,      // Single qubit (H, S, X, T, M, etc.)
    PAIR,        // Two qubits consumed in pairs (CX, CY, CZ)
    MULTI,       // Variable targets (MPP)
    ANNOTATION,  // No qubit targets (TICK)
};

// Get the arity classification for a gate type.
inline constexpr GateArity gate_arity(GateType g) {
    switch (g) {
        case GateType::CX:
        case GateType::CY:
        case GateType::CZ:
        case GateType::SWAP:
        case GateType::ISWAP:
        case GateType::ISWAP_DAG:
        case GateType::SQRT_XX:
        case GateType::SQRT_XX_DAG:
        case GateType::SQRT_YY:
        case GateType::SQRT_YY_DAG:
        case GateType::SQRT_ZZ:
        case GateType::SQRT_ZZ_DAG:
        case GateType::CXSWAP:
        case GateType::CZSWAP:
        case GateType::SWAPCX:
        case GateType::XCX:
        case GateType::XCY:
        case GateType::XCZ:
        case GateType::YCX:
        case GateType::YCY:
        case GateType::YCZ:
        case GateType::MXX:
        case GateType::MYY:
        case GateType::MZZ:
        case GateType::II:
        case GateType::II_ERROR:
        case GateType::DEPOLARIZE2:
        case GateType::PAULI_CHANNEL_2:
            return GateArity::PAIR;
        case GateType::MPP:
            return GateArity::MULTI;
        case GateType::TICK:
        case GateType::DETECTOR:
        case GateType::OBSERVABLE_INCLUDE:
            return GateArity::ANNOTATION;
        default:
            return GateArity::SINGLE;
    }
}

// Check if a gate is a Clifford gate.
inline constexpr bool is_clifford(GateType g) {
    switch (g) {
        case GateType::H:
        case GateType::S:
        case GateType::S_DAG:
        case GateType::X:
        case GateType::Y:
        case GateType::Z:
        case GateType::SQRT_X:
        case GateType::SQRT_X_DAG:
        case GateType::SQRT_Y:
        case GateType::SQRT_Y_DAG:
        case GateType::H_XY:
        case GateType::H_YZ:
        case GateType::H_NXY:
        case GateType::H_NXZ:
        case GateType::H_NYZ:
        case GateType::C_XYZ:
        case GateType::C_ZYX:
        case GateType::C_NXYZ:
        case GateType::C_NZYX:
        case GateType::C_XNYZ:
        case GateType::C_XYNZ:
        case GateType::C_ZNYX:
        case GateType::C_ZYNX:
        case GateType::CX:
        case GateType::CY:
        case GateType::CZ:
        case GateType::SWAP:
        case GateType::ISWAP:
        case GateType::ISWAP_DAG:
        case GateType::SQRT_XX:
        case GateType::SQRT_XX_DAG:
        case GateType::SQRT_YY:
        case GateType::SQRT_YY_DAG:
        case GateType::SQRT_ZZ:
        case GateType::SQRT_ZZ_DAG:
        case GateType::CXSWAP:
        case GateType::CZSWAP:
        case GateType::SWAPCX:
        case GateType::XCX:
        case GateType::XCY:
        case GateType::XCZ:
        case GateType::YCX:
        case GateType::YCY:
        case GateType::YCZ:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a measurement (produces visible outcome).
inline constexpr bool is_measurement(GateType g) {
    switch (g) {
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
        case GateType::MPP:
        case GateType::MXX:
        case GateType::MYY:
        case GateType::MZZ:
        case GateType::MPAD:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a reset (no visible outcome, but collapses state).
inline constexpr bool is_reset(GateType g) {
    switch (g) {
        case GateType::R:
        case GateType::RX:
        case GateType::RY:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a measure-reset (produces visible outcome AND resets).
inline constexpr bool is_measure_reset(GateType g) {
    switch (g) {
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
            return true;
        default:
            return false;
    }
}

// Check if a gate is an identity no-op (parsed but never emitted to AST).
inline constexpr bool is_identity_noop(GateType g) {
    switch (g) {
        case GateType::I:
        case GateType::II:
        case GateType::I_ERROR:
        case GateType::II_ERROR:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a noise channel.
inline constexpr bool is_noise_gate(GateType g) {
    switch (g) {
        case GateType::X_ERROR:
        case GateType::Y_ERROR:
        case GateType::Z_ERROR:
        case GateType::DEPOLARIZE1:
        case GateType::DEPOLARIZE2:
        case GateType::PAULI_CHANNEL_1:
        case GateType::PAULI_CHANNEL_2:
        case GateType::READOUT_NOISE:
            return true;
        default:
            return false;
    }
}

// Convert gate type to string for display/debugging.
inline constexpr std::string_view gate_name(GateType g) {
    switch (g) {
        case GateType::H:
            return "H";
        case GateType::S:
            return "S";
        case GateType::S_DAG:
            return "S_DAG";
        case GateType::X:
            return "X";
        case GateType::Y:
            return "Y";
        case GateType::Z:
            return "Z";
        case GateType::SQRT_X:
            return "SQRT_X";
        case GateType::SQRT_X_DAG:
            return "SQRT_X_DAG";
        case GateType::SQRT_Y:
            return "SQRT_Y";
        case GateType::SQRT_Y_DAG:
            return "SQRT_Y_DAG";
        case GateType::H_XY:
            return "H_XY";
        case GateType::H_YZ:
            return "H_YZ";
        case GateType::H_NXY:
            return "H_NXY";
        case GateType::H_NXZ:
            return "H_NXZ";
        case GateType::H_NYZ:
            return "H_NYZ";
        case GateType::C_XYZ:
            return "C_XYZ";
        case GateType::C_ZYX:
            return "C_ZYX";
        case GateType::C_NXYZ:
            return "C_NXYZ";
        case GateType::C_NZYX:
            return "C_NZYX";
        case GateType::C_XNYZ:
            return "C_XNYZ";
        case GateType::C_XYNZ:
            return "C_XYNZ";
        case GateType::C_ZNYX:
            return "C_ZNYX";
        case GateType::C_ZYNX:
            return "C_ZYNX";
        case GateType::T:
            return "T";
        case GateType::T_DAG:
            return "T_DAG";
        case GateType::CX:
            return "CX";
        case GateType::CY:
            return "CY";
        case GateType::CZ:
            return "CZ";
        case GateType::SWAP:
            return "SWAP";
        case GateType::ISWAP:
            return "ISWAP";
        case GateType::ISWAP_DAG:
            return "ISWAP_DAG";
        case GateType::SQRT_XX:
            return "SQRT_XX";
        case GateType::SQRT_XX_DAG:
            return "SQRT_XX_DAG";
        case GateType::SQRT_YY:
            return "SQRT_YY";
        case GateType::SQRT_YY_DAG:
            return "SQRT_YY_DAG";
        case GateType::SQRT_ZZ:
            return "SQRT_ZZ";
        case GateType::SQRT_ZZ_DAG:
            return "SQRT_ZZ_DAG";
        case GateType::CXSWAP:
            return "CXSWAP";
        case GateType::CZSWAP:
            return "CZSWAP";
        case GateType::SWAPCX:
            return "SWAPCX";
        case GateType::XCX:
            return "XCX";
        case GateType::XCY:
            return "XCY";
        case GateType::XCZ:
            return "XCZ";
        case GateType::YCX:
            return "YCX";
        case GateType::YCY:
            return "YCY";
        case GateType::YCZ:
            return "YCZ";
        case GateType::M:
            return "M";
        case GateType::MX:
            return "MX";
        case GateType::MY:
            return "MY";
        case GateType::MR:
            return "MR";
        case GateType::MRX:
            return "MRX";
        case GateType::MPP:
            return "MPP";
        case GateType::MXX:
            return "MXX";
        case GateType::MYY:
            return "MYY";
        case GateType::MZZ:
            return "MZZ";
        case GateType::R:
            return "R";
        case GateType::RX:
            return "RX";
        case GateType::RY:
            return "RY";
        case GateType::MRY:
            return "MRY";
        case GateType::MPAD:
            return "MPAD";
        case GateType::I:
            return "I";
        case GateType::II:
            return "II";
        case GateType::I_ERROR:
            return "I_ERROR";
        case GateType::II_ERROR:
            return "II_ERROR";
        case GateType::X_ERROR:
            return "X_ERROR";
        case GateType::Y_ERROR:
            return "Y_ERROR";
        case GateType::Z_ERROR:
            return "Z_ERROR";
        case GateType::DEPOLARIZE1:
            return "DEPOLARIZE1";
        case GateType::DEPOLARIZE2:
            return "DEPOLARIZE2";
        case GateType::PAULI_CHANNEL_1:
            return "PAULI_CHANNEL_1";
        case GateType::PAULI_CHANNEL_2:
            return "PAULI_CHANNEL_2";
        case GateType::READOUT_NOISE:
            return "READOUT_NOISE";
        case GateType::DETECTOR:
            return "DETECTOR";
        case GateType::OBSERVABLE_INCLUDE:
            return "OBSERVABLE_INCLUDE";
        case GateType::TICK:
            return "TICK";
        case GateType::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

// Parse gate name string to GateType.
// Returns GateType::UNKNOWN for unrecognized names.
GateType parse_gate_name(std::string_view name);

}  // namespace ucc
