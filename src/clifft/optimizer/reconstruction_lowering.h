#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/global_reconstruction.h"

namespace clifft {

struct ReconstructionLoweringResult {
    std::optional<HirModule> hir;
    std::string fallback_reason;
};

// The caller supplies references chosen by the original pipeline. A fresh
// reference shot of the reconstructed problem can choose a different convention.
[[nodiscard]] ReconstructionLoweringResult lower_global_reconstruction(
    const GlobalReconstruction& reconstruction, const PostselectionMetadata& postselection = {},
    const ReconstructionLimits& limits = {});

}  // namespace clifft
