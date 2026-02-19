#pragma once

// UCC compile-time configuration
// This header defines constants and limits used throughout the codebase.

namespace ucc {

// Maximum number of qubits supported in the fast-path (inline uint64_t masks).
// Circuits exceeding this limit will require the simd_bits<W> path (Phase 3).
constexpr int kMaxInlineQubits = 64;

}  // namespace ucc
