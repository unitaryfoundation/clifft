#pragma once

#include "clifft/frontend/hir.h"

#include <cstdint>
#include <vector>

namespace clifft {

/// Noiseless reference syndrome for error syndrome normalization.
struct ReferenceSyndrome {
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
};

/// Compute the noiseless reference syndrome for an HirModule.
/// Strips stochastic and readout noise from a copy of the HIR and samples one
/// shot at seed 0. This does not check determinism; coherent rotations remain.
/// For coherent errors, pass the intended ideal circuit's HIR instead.
[[nodiscard]] ReferenceSyndrome compute_reference_syndrome(const HirModule& hir);

}  // namespace clifft
