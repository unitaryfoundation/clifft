#pragma once

#include "ucc/optimizer/bytecode_pass.h"

namespace ucc {

/// Fuses contiguous ARRAY_CNOT instructions sharing a target axis into a
/// single OP_ARRAY_MULTI_CNOT, and contiguous ARRAY_CZ instructions sharing
/// a control axis into a single OP_ARRAY_MULTI_CZ.
///
/// These "star-graph" patterns arise from the backend's greedy Pauli
/// compressor folding multi-qubit Pauli strings onto a single pivot.
/// Without fusion, each CNOT/CZ triggers a separate full O(2^k) array pass.
/// The fused instruction processes all targets/controls in one pass using
/// popcount-based parity checks.
class MultiGatePass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace ucc
