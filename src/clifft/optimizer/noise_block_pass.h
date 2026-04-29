#pragma once

#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

/// Coalesces contiguous OP_NOISE instructions with consecutive site indices
/// into single OP_NOISE_BLOCK instructions, reducing dispatch overhead in
/// noise-heavy circuits.
class NoiseBlockPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace clifft
