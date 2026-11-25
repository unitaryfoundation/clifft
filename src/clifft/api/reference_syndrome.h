#pragma once

#include "clifft/circuit/circuit.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/global_reconstruction.h"

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
/// shot at seed 0. With a reconstruction source, wide references use a narrower
/// detector plan when available, including when sequential planning overflows.
/// With runtime_frame, a reference whose sequential plan overflows is planned
/// on a runtime Clifford frame before reconstruction is tried.
/// This does not check determinism; coherent rotations remain. For coherent
/// errors, pass the intended ideal circuit's HIR instead.
[[nodiscard]] ReferenceSyndrome compute_reference_syndrome(
    const HirModule& hir, const Circuit* reconstruction_source = nullptr,
    const ReconstructionLimits& limits = {}, bool runtime_frame = false);

}  // namespace clifft
