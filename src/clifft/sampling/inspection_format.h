#pragma once

// Shared formatting primitives for the human-readable sampling-IR inspection
// output produced by SamplingPlan and ExecutablePlan. Keeping these helpers in
// one place gives both inspectors identical number and operand rendering.

#include <cstdint>
#include <string>

namespace clifft::sampling {

// Minimal-digits round-trip formatting: the shortest of the "%.15g", "%.16g",
// and "%.17g" renderings whose strtod parse reproduces the input bit for bit.
// Falls back to "%.17g" when no candidate round-trips, such as for NaN.
[[nodiscard]] std::string format_double_roundtrip(double value);

// Sparse Pauli product notation over ascending bit indices: each set bit
// contributes X<i>, Z<i>, or Y<i> depending on which of x/z hold it, joined by
// '*'. Both masks zero renders as "I".
[[nodiscard]] std::string format_pauli_product(uint64_t x, uint64_t z);

// Active-width prefix for one inspected action: "w<n>" when an action leaves
// the active width unchanged, or "w<before>-><after>" when it changes.
[[nodiscard]] std::string format_width_prefix(uint32_t before, uint32_t after);

}  // namespace clifft::sampling
