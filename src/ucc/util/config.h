#pragma once

// UCC compile-time configuration
// This header defines constants and limits used throughout the codebase.

#include <cstdint>

namespace ucc {

// Maximum number of qubits supported at compile-time.
// Set via -DUCC_MAX_QUBITS=N at CMake configure time (default 64).
// Determines the width of BitMask<N> used in HIR Pauli masks and the
// SVM Pauli frame. The VM Instruction struct stays 32 bytes regardless.
#ifndef UCC_MAX_QUBITS
#define UCC_MAX_QUBITS 64
#endif

static_assert(UCC_MAX_QUBITS >= 64, "UCC_MAX_QUBITS must be at least 64");
static_assert(UCC_MAX_QUBITS % 64 == 0, "UCC_MAX_QUBITS must be a multiple of 64");

constexpr uint32_t kMaxInlineQubits = UCC_MAX_QUBITS;

// Number of 64-bit words needed to hold kMaxInlineQubits bits.
constexpr uint32_t kMaxInlineWords = kMaxInlineQubits / 64;

// Maximum targets per instruction line (defense against malicious input).
// 1M targets is far beyond any legitimate use case.
constexpr uint32_t kMaxTargetsPerInstruction = 1'000'000;

// Maximum total AST nodes after REPEAT unrolling (defense against OOM).
constexpr size_t kMaxUnrolledOps = 10'000'000;

}  // namespace ucc
