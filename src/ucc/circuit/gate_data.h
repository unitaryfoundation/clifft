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

    // Non-Clifford gates
    T,
    T_DAG,

    // Two-qubit Clifford gates
    CX,
    CY,
    CZ,

    // Measurements
    M,    // Z-basis measurement
    MX,   // X-basis measurement
    MY,   // Y-basis measurement
    MR,   // Measure + reset (Z-basis)
    MRX,  // Measure + reset (X-basis)
    MPP,  // Multi-Pauli measurement

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
            return GateArity::PAIR;
        case GateType::MPP:
            return GateArity::MULTI;
        case GateType::TICK:
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
        case GateType::CX:
        case GateType::CY:
        case GateType::CZ:
            return true;
        default:
            return false;
    }
}

// Check if a gate is a measurement.
inline constexpr bool is_measurement(GateType g) {
    switch (g) {
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
        case GateType::MR:
        case GateType::MRX:
        case GateType::MPP:
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
