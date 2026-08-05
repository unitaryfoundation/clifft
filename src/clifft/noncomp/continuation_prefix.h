#pragma once

#include "clifft/backend/backend.h"

#include <cstdint>

namespace clifft {

/// Require a newly compiled continuation to reproduce the executed bytecode
/// prefix and every constant-pool value referenced by it. The one permitted
/// bytecode difference is a sampling measurement that was changed to its
/// forced-outcome variant in the executed module.
///
/// Throws std::logic_error on divergence. This check is part of the release
/// runtime contract because resume() reuses the live VM state produced by the
/// executed module.
void validate_continuation_prefix(const CompiledModule& continuation,
                                  const CompiledModule& executed, uint32_t prefix_end);

}  // namespace clifft
