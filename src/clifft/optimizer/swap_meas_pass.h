#pragma once

#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

/// Fuses contiguous OP_ARRAY_SWAP + OP_MEAS_ACTIVE_INTERFERE pairs into
/// single OP_SWAP_MEAS_INTERFERE instructions.
///
/// The backend emits a SWAP to route the measurement axis to position k-1
/// (required for contiguous compaction), followed by the interfere
/// measurement.  The fused instruction performs both operations in a
/// single dispatch cycle, eliminating the separate O(2^k) ARRAY_SWAP memory
/// pass and its instruction fetch/decode overhead.
class SwapMeasPass : public BytecodePass {
  public:
    void run(CompiledModule& module) override;
};

}  // namespace clifft
