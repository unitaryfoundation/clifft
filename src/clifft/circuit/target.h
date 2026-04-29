#pragma once

// Target encoding for circuit AST nodes.
//
// Each target in an AstNode is a Target struct wrapping a 32-bit value that can represent:
// - A plain qubit index
// - A measurement record reference (rec[-k])
// - A Pauli-tagged qubit for MPP (e.g., X3, Y5, Z0)
//
// 32-bit Layout:
//   Bits 0-27:  Value (qubit index or record absolute offset)
//   Bit 28:     Is Record flag (rec[-k])
//   Bits 29-30: Pauli Tag (0=None, 1=X, 2=Y, 3=Z)
//   Bit 31:     Is Inverted (for ! measurement flags)
//
// Target is a thin wrapper that prevents accidental mixing of raw integers
// with encoded targets. Use the factory functions to create Targets.

#include <cstdint>

namespace clifft {

struct Target {
    // Raw encoded bits - public for serialization/debugging, but prefer accessors.
    uint32_t bits;

    // Bit layout constants.
    static constexpr uint32_t kValueMask = 0x0FFFFFFF;
    static constexpr uint32_t kRecBit = 1u << 28;
    static constexpr uint32_t kPauliShift = 29;
    static constexpr uint32_t kPauliMask = 3u << kPauliShift;
    static constexpr uint32_t kPauliNone = 0u << kPauliShift;
    static constexpr uint32_t kPauliX = 1u << kPauliShift;
    static constexpr uint32_t kPauliY = 2u << kPauliShift;
    static constexpr uint32_t kPauliZ = 3u << kPauliShift;
    static constexpr uint32_t kInvertBit = 1u << 31;

    // -------------------------------------------------------------------------
    // Factory functions (the canonical way to create Targets)
    // -------------------------------------------------------------------------

    static constexpr Target qubit(uint32_t q) { return Target{q & kValueMask}; }

    /// `offset` is the absolute index into the measurement record.
    static constexpr Target rec(uint32_t offset) { return Target{(offset & kValueMask) | kRecBit}; }

    /// `pauli_flag` must be one of kPauliX, kPauliY, kPauliZ.
    static constexpr Target pauli(uint32_t q, uint32_t pauli_flag) {
        return Target{(q & kValueMask) | pauli_flag};
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    constexpr bool is_rec() const { return (bits & kRecBit) != 0; }

    constexpr bool is_inverted() const { return (bits & kInvertBit) != 0; }

    /// Returns the 28-bit value (qubit index or record offset).
    constexpr uint32_t value() const { return bits & kValueMask; }

    /// Returns one of kPauliNone, kPauliX, kPauliY, kPauliZ.
    constexpr uint32_t pauli() const { return bits & kPauliMask; }

    constexpr bool has_pauli() const { return pauli() != kPauliNone; }

    /// Pauli letter for display ('I' when untagged, else 'X' / 'Y' / 'Z').
    constexpr char pauli_char() const {
        switch (pauli()) {
            case kPauliX:
                return 'X';
            case kPauliY:
                return 'Y';
            case kPauliZ:
                return 'Z';
            default:
                return 'I';
        }
    }

    // -------------------------------------------------------------------------
    // Modifiers (return new Target)
    // -------------------------------------------------------------------------

    constexpr Target inverted() const { return Target{bits | kInvertBit}; }

    // -------------------------------------------------------------------------
    // Comparison operators
    // -------------------------------------------------------------------------

    constexpr bool operator==(Target other) const { return bits == other.bits; }
    constexpr bool operator!=(Target other) const { return bits != other.bits; }
};

}  // namespace clifft
