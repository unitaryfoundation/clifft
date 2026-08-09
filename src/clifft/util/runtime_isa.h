#pragma once

namespace clifft::internal {

// One process-wide selection shared by every architecture-specific kernel.
// Trap states defer configuration errors until a backend reaches an execution
// or preparation boundary, avoiding exceptions during static initialization.
// Builds without runtime dispatch always select Scalar and ignore
// CLIFFT_FORCE_ISA, matching the portable SVM behavior.
enum class RuntimeIsa {
    Scalar,
    Avx2,
    Avx512,
    TrapAvx2,
    TrapAvx512,
    TrapUnknown,
};

[[nodiscard]] RuntimeIsa runtime_isa();
[[nodiscard]] const char* runtime_isa_name(RuntimeIsa isa) noexcept;

// Returns for executable selections and throws the established
// CLIFFT_FORCE_ISA error for trap selections.
void validate_runtime_isa(RuntimeIsa isa);

}  // namespace clifft::internal
