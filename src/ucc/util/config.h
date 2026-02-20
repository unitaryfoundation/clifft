#pragma once

// UCC compile-time configuration
// This header defines constants and limits used throughout the codebase.

#include <cstdint>

namespace ucc {

// Maximum number of qubits supported in the fast-path (inline uint64_t masks).
// Circuits exceeding this limit will require the simd_bits<W> path (Phase 3).
constexpr int kMaxInlineQubits = 64;

// Maximum targets per instruction line (defense against malicious input).
// 1M targets is far beyond any legitimate use case.
constexpr uint32_t kMaxTargetsPerInstruction = 1'000'000;

}  // namespace ucc
