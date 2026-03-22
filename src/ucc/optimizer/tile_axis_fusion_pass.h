#pragma once

#include "ucc/optimizer/bytecode_pass.h"

namespace ucc {

/// Fuses consecutive 2-qubit tile sequences into single OP_ARRAY_U4
/// instructions with precomputed 4x4 unitary matrices.
///
/// A "tile run" is a maximal sequence of array-touching operations where
/// every op targets only virtual axes {a, b} for some fixed pair, plus
/// frame ops on those same axes (which are absorbed into the fused
/// matrix). The pass pre-computes 4x4 unitaries for all 16 possible
/// incoming Pauli frame states and stores them in the ConstantPool.
///
/// Tile runs are terminated by:
///   - Any instruction that touches a different axis (including frame ops)
///   - An EXPAND (changes array dimension)
///   - A measurement, noise, or classical op
///   - A frame op that touches one tile axis and one external axis
///
/// A run is only fused if it contains at least 3 array-touching operations.
/// Below that threshold, the bandwidth savings from eliminating sweeps
/// do not compensate for the heavier 4x4 matrix multiply.
///
/// This pass should run BEFORE SingleAxisFusionPass so it operates on
/// raw primitive opcodes. SingleAxisFusionPass then cleans up any
/// remaining isolated 1Q chains.
class TileAxisFusionPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace ucc
