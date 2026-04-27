#pragma once

#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

/// Fuses consecutive single-axis operations (ARRAY_H, ARRAY_S, ARRAY_S_DAG,
/// ARRAY_T, ARRAY_T_DAG, ARRAY_ROT, FRAME_H, FRAME_S, FRAME_S_DAG) on the
/// same virtual axis into a single OP_ARRAY_U2 instruction.
///
/// For each fusible run, the pass pre-computes 2x2 unitary matrices for all
/// 4 possible incoming Pauli frame states (I, X, Z, Y) and stores them in
/// the ConstantPool. The VM then performs a single butterfly array sweep
/// instead of multiple separate passes.
///
/// Fusible runs are terminated by any instruction that:
///   - Operates on a different axis
///   - Is a two-qubit or multi-qubit gate
///   - Is an EXPAND (changes array dimension)
///   - Is a measurement, noise, or classical op
///
/// A run is only fused if it contains at least 3 array-touching operations,
/// or at least 2 array ops when one is a continuous rotation (OP_ARRAY_ROT).
/// Isolated length-2 sequences of lightweight ops (e.g. H+T) are cheaper
/// unfused than a dense 2x2 matrix sweep.
class SingleAxisFusionPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace clifft
