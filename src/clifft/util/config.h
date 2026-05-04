#pragma once

// Clifft compile-time configuration.

#include <cstddef>
#include <cstdint>

namespace clifft {

// Maximum targets per instruction line (defense against malicious input).
// 1M targets is far beyond any legitimate use case.
constexpr uint32_t kMaxTargetsPerInstruction = 1'000'000;

// Maximum total AST nodes after REPEAT unrolling (defense against OOM).
constexpr size_t kMaxUnrolledOps = 10'000'000;

}  // namespace clifft
