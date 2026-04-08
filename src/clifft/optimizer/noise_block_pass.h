#pragma once

#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

/// Coalesces contiguous OP_NOISE instructions with consecutive site indices
/// into single OP_NOISE_BLOCK instructions.
///
/// The VM's exponential gap-sampling already skips silent noise sites in O(1)
/// RNG time, but without this pass, the dispatch loop still individually
/// fetches and decodes each OP_NOISE instruction. For the d=5 surface code
/// circuit, this collapses 3471 OP_NOISE instructions into 24 OP_NOISE_BLOCK
/// instructions, eliminating ~68% of all bytecode dispatch overhead.
class NoiseBlockPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace clifft
