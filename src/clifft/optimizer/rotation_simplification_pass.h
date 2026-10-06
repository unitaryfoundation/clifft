#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>

namespace clifft {

struct RotationSimplificationOptions {
    // Zero disables the pass. The cap bounds live terms and pairwise commutation work.
    uint32_t max_region_ops = 256;
    uint32_t max_region_passes = 8;
};

// Uses outcome-independent Pauli constraints from the complete circuit's |0>
// input to simplify arbitrary-angle rotations. Run after PhasePolynomialPass,
// before squeezing or scheduling. All nonrotation operations are boundaries.
// Newly absorbed Cliffords update a compile-time frame, so subsequent regions
// see stronger facts without rerunning the circuit's earlier optimization passes.
class RotationSimplificationPass : public HirPass {
  public:
    explicit RotationSimplificationPass(RotationSimplificationOptions options = {});
    void run(HirModule& hir) override;

    size_t regions_examined() const { return regions_examined_; }
    size_t regions_reduced() const { return regions_reduced_; }
    size_t regions_capped() const { return regions_capped_; }
    size_t rotations_removed() const { return rotations_removed_; }
    bool applied() const { return regions_reduced_ != 0; }

  private:
    RotationSimplificationOptions options_;
    size_t regions_examined_ = 0;
    size_t regions_reduced_ = 0;
    size_t regions_capped_ = 0;
    size_t rotations_removed_ = 0;
};

}  // namespace clifft
