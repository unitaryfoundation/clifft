#pragma once

// Shared numeric helpers for noncomputational model validation.

#include "clifft/util/numeric.h"

namespace clifft {

// Tolerance for derived-quantity bounds (column sums, the
// initial-state sum, source-independence equality). Raw user-supplied
// matrix entries are checked strictly against [0, 1]; only quantities
// derived from them tolerate this much floating drift.
inline constexpr double kProbTolerance = 1e-12;

}  // namespace clifft
