#pragma once

#include "ucc/frontend/hir.h"

#include <cstdint>
#include <vector>

namespace ucc {

/// Noiseless reference syndrome for error syndrome normalization.
struct ReferenceSyndrome {
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
};

/// Compute the noiseless reference syndrome for an HirModule.
/// Strips all noise from a copy of the HIR, lowers it, and runs one
/// deterministic shot to extract the expected detector/observable parities.
[[nodiscard]] ReferenceSyndrome compute_reference_syndrome(const HirModule& hir);

}  // namespace ucc
