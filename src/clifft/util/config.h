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

// Wide coefficient traversals can amortize OpenMP team startup. Expert
// thread layouts may override this default without rebuilding Clifft.
inline constexpr uint32_t kDefaultIntraShotMinActiveWidth = 18;

}  // namespace clifft
