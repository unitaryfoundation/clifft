#pragma once

#include "clifft/noncomp/preparation.h"

namespace clifft::experimental {

// Research executor: online inverse Clifford coordinates, production dense kernels.
// The caller chooses an explicit coefficient-memory budget before sampling.
NonComputationalSample sample_hybrid(const PreparedNonComputationalRequest& request, uint32_t shots,
                                     uint64_t seed, uint32_t capacity);

}  // namespace clifft::experimental
