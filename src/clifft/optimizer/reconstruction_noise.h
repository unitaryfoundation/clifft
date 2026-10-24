#pragma once

#include "clifft/optimizer/global_reconstruction.h"

namespace clifft {

// Only the joint law of signs seen by operations matters. Coordinates describe
// that image, not the original fault sites. Constants remain separate and signs
// retain operation order; outcomes within each channel remain mutually exclusive.
struct ReconstructionNoise {
    uint32_t num_bits = 0;
    std::vector<ReconstructionSign> signs;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels;
};

struct ReconstructionNoiseResult {
    std::optional<ReconstructionNoise> noise;
    std::string fallback_reason;
};

// Project complete channels into the operation-sign span and convolve channels
// with identical image support. This transformation has no backend dependency.
[[nodiscard]] ReconstructionNoiseResult compress_reconstruction_noise(
    const GlobalReconstruction& reconstruction, const ReconstructionLimits& limits = {});

}  // namespace clifft
