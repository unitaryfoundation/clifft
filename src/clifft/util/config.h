#pragma once

// Clifft compile-time configuration
// This header defines constants and limits used throughout the codebase.

#include <cstdint>

namespace clifft {

// Maximum number of qubits supported at compile-time.
// Set via -DCLIFFT_MAX_QUBITS=N at CMake configure time (default 128).
// Determines the width of BitMask<N> used in HIR Pauli masks and the
// SVM Pauli frame. The VM Instruction struct stays 32 bytes regardless.
#ifndef CLIFFT_MAX_QUBITS
#define CLIFFT_MAX_QUBITS 128
#endif

static_assert(CLIFFT_MAX_QUBITS >= 64, "CLIFFT_MAX_QUBITS must be at least 64");
static_assert(CLIFFT_MAX_QUBITS % 64 == 0, "CLIFFT_MAX_QUBITS must be a multiple of 64");

constexpr uint32_t kMaxInlineQubits = CLIFFT_MAX_QUBITS;

// Number of 64-bit words needed to hold kMaxInlineQubits bits.
constexpr uint32_t kMaxInlineWords = kMaxInlineQubits / 64;

// Maximum targets per instruction line (defense against malicious input).
// 1M targets is far beyond any legitimate use case.
constexpr uint32_t kMaxTargetsPerInstruction = 1'000'000;

// Maximum total AST nodes after REPEAT unrolling (defense against OOM).
constexpr size_t kMaxUnrolledOps = 10'000'000;

}  // namespace clifft
