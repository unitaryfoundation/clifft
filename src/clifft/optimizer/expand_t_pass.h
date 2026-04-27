#pragma once

#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

/// Fuses contiguous OP_EXPAND + OP_ARRAY_T (or T_DAG) pairs into single
/// OP_EXPAND_T (or OP_EXPAND_T_DAG) instructions.
///
/// Every T-gate injection in the factored state architecture requires an
/// EXPAND to activate a dormant axis followed by a T-phase rotation on
/// that axis.  The separate instructions cause two array passes (copy +
/// phase multiply).  The fused instruction performs both in one loop:
/// arr[i + half] = arr[i] * exp(+/-i*pi/4).
class ExpandTPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

/// Fuses contiguous OP_EXPAND + OP_ARRAY_ROT pairs into single
/// OP_EXPAND_ROT instructions.
class ExpandRotPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace clifft
