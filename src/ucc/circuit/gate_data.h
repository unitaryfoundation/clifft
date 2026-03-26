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

    // Parameterized rotation gates (continuous angles in half-turn units)
    R_X,
    R_Y,
    R_Z,
    U3,
    R_XX,
    R_YY,
    R_ZZ,
    R_PAULI,

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

    // Simulation-only probes
    EXP_VAL,  // Non-destructive expectation value

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

// Centralized gate metadata. One entry per GateType, indexed by enum value.
// Booleans default to false so designated initializers only name true fields.
struct GateTraits {
    GateArity arity;
    bool clifford = false;
    bool measurement = false;
    bool reset = false;
    bool measure_reset = false;
    bool identity_noop = false;
    bool noise = false;
    std::string_view name;
};

namespace detail {

constexpr auto S = GateArity::SINGLE;
constexpr auto P = GateArity::PAIR;
constexpr auto ML = GateArity::MULTI;
constexpr auto A = GateArity::ANNOTATION;

// clang-format off
inline constexpr GateTraits kGateTraitsData[] = {
    // Single-qubit Cliffords
    {.arity = S, .clifford = true, .name = "H"},
    {.arity = S, .clifford = true, .name = "S"},
    {.arity = S, .clifford = true, .name = "S_DAG"},
    {.arity = S, .clifford = true, .name = "X"},
    {.arity = S, .clifford = true, .name = "Y"},
    {.arity = S, .clifford = true, .name = "Z"},
    {.arity = S, .clifford = true, .name = "SQRT_X"},
    {.arity = S, .clifford = true, .name = "SQRT_X_DAG"},
    {.arity = S, .clifford = true, .name = "SQRT_Y"},
    {.arity = S, .clifford = true, .name = "SQRT_Y_DAG"},
    {.arity = S, .clifford = true, .name = "H_XY"},
    {.arity = S, .clifford = true, .name = "H_YZ"},
    {.arity = S, .clifford = true, .name = "H_NXY"},
    {.arity = S, .clifford = true, .name = "H_NXZ"},
    {.arity = S, .clifford = true, .name = "H_NYZ"},
    {.arity = S, .clifford = true, .name = "C_XYZ"},
    {.arity = S, .clifford = true, .name = "C_ZYX"},
    {.arity = S, .clifford = true, .name = "C_NXYZ"},
    {.arity = S, .clifford = true, .name = "C_NZYX"},
    {.arity = S, .clifford = true, .name = "C_XNYZ"},
    {.arity = S, .clifford = true, .name = "C_XYNZ"},
    {.arity = S, .clifford = true, .name = "C_ZNYX"},
    {.arity = S, .clifford = true, .name = "C_ZYNX"},
    // Non-Clifford
    {.arity = S, .name = "T"},
    {.arity = S, .name = "T_DAG"},
    // Parameterized rotations
    {.arity = S, .name = "R_X"},
    {.arity = S, .name = "R_Y"},
    {.arity = S, .name = "R_Z"},
    {.arity = S, .name = "U3"},
    {.arity = P, .name = "R_XX"},
    {.arity = P, .name = "R_YY"},
    {.arity = P, .name = "R_ZZ"},
    {.arity = ML, .name = "R_PAULI"},
    // Two-qubit Cliffords
    {.arity = P, .clifford = true, .name = "CX"},
    {.arity = P, .clifford = true, .name = "CY"},
    {.arity = P, .clifford = true, .name = "CZ"},
    {.arity = P, .clifford = true, .name = "SWAP"},
    {.arity = P, .clifford = true, .name = "ISWAP"},
    {.arity = P, .clifford = true, .name = "ISWAP_DAG"},
    {.arity = P, .clifford = true, .name = "SQRT_XX"},
    {.arity = P, .clifford = true, .name = "SQRT_XX_DAG"},
    {.arity = P, .clifford = true, .name = "SQRT_YY"},
    {.arity = P, .clifford = true, .name = "SQRT_YY_DAG"},
    {.arity = P, .clifford = true, .name = "SQRT_ZZ"},
    {.arity = P, .clifford = true, .name = "SQRT_ZZ_DAG"},
    {.arity = P, .clifford = true, .name = "CXSWAP"},
    {.arity = P, .clifford = true, .name = "CZSWAP"},
    {.arity = P, .clifford = true, .name = "SWAPCX"},
    {.arity = P, .clifford = true, .name = "XCX"},
    {.arity = P, .clifford = true, .name = "XCY"},
    {.arity = P, .clifford = true, .name = "XCZ"},
    {.arity = P, .clifford = true, .name = "YCX"},
    {.arity = P, .clifford = true, .name = "YCY"},
    {.arity = P, .clifford = true, .name = "YCZ"},
    // Measurements
    {.arity = S, .measurement = true, .name = "M"},
    {.arity = S, .measurement = true, .name = "MX"},
    {.arity = S, .measurement = true, .name = "MY"},
    {.arity = S, .measurement = true, .measure_reset = true, .name = "MR"},
    {.arity = S, .measurement = true, .measure_reset = true, .name = "MRX"},
    {.arity = ML, .measurement = true, .name = "MPP"},
    {.arity = P, .measurement = true, .name = "MXX"},
    {.arity = P, .measurement = true, .name = "MYY"},
    {.arity = P, .measurement = true, .name = "MZZ"},
    // Resets
    {.arity = S, .reset = true, .name = "R"},
    {.arity = S, .reset = true, .name = "RX"},
    {.arity = S, .reset = true, .name = "RY"},
    {.arity = S, .measurement = true, .measure_reset = true, .name = "MRY"},
    // Deterministic padding
    {.arity = S, .measurement = true, .name = "MPAD"},
    // Identity no-ops
    {.arity = S, .identity_noop = true, .name = "I"},
    {.arity = P, .identity_noop = true, .name = "II"},
    {.arity = S, .identity_noop = true, .name = "I_ERROR"},
    {.arity = P, .identity_noop = true, .name = "II_ERROR"},
    // Noise channels
    {.arity = S, .noise = true, .name = "X_ERROR"},
    {.arity = S, .noise = true, .name = "Y_ERROR"},
    {.arity = S, .noise = true, .name = "Z_ERROR"},
    {.arity = S, .noise = true, .name = "DEPOLARIZE1"},
    {.arity = P, .noise = true, .name = "DEPOLARIZE2"},
    {.arity = S, .noise = true, .name = "PAULI_CHANNEL_1"},
    {.arity = P, .noise = true, .name = "PAULI_CHANNEL_2"},
    {.arity = S, .noise = true, .name = "READOUT_NOISE"},
    // QEC annotations
    {.arity = A, .name = "DETECTOR"},
    {.arity = A, .name = "OBSERVABLE_INCLUDE"},
    {.arity = A, .name = "TICK"},
    // Simulation-only probes
    {.arity = ML, .name = "EXP_VAL"},
    // Sentinel
    {.arity = S, .name = "UNKNOWN"},
};
// clang-format on

static_assert(sizeof(kGateTraitsData) / sizeof(kGateTraitsData[0]) ==
                  static_cast<size_t>(GateType::UNKNOWN) + 1,
              "kGateTraitsData must have one entry per GateType");

}  // namespace detail

// Lookup gate traits by enum value. O(1) array index.
inline constexpr const GateTraits& gate_traits(GateType g) {
    return detail::kGateTraitsData[static_cast<size_t>(g)];
}

inline constexpr GateArity gate_arity(GateType g) {
    return gate_traits(g).arity;
}
inline constexpr bool is_clifford(GateType g) {
    return gate_traits(g).clifford;
}
inline constexpr bool is_measurement(GateType g) {
    return gate_traits(g).measurement;
}
inline constexpr bool is_reset(GateType g) {
    return gate_traits(g).reset;
}
inline constexpr bool is_measure_reset(GateType g) {
    return gate_traits(g).measure_reset;
}
inline constexpr bool is_identity_noop(GateType g) {
    return gate_traits(g).identity_noop;
}
inline constexpr bool is_noise_gate(GateType g) {
    return gate_traits(g).noise;
}
inline constexpr bool is_exp_val(GateType g) {
    return g == GateType::EXP_VAL;
}
inline constexpr std::string_view gate_name(GateType g) {
    return gate_traits(g).name;
}

// Parse gate name string to GateType.
// Returns GateType::UNKNOWN for unrecognized names.
GateType parse_gate_name(std::string_view name);

}  // namespace ucc
