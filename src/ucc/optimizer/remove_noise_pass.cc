#include "ucc/optimizer/remove_noise_pass.h"

#include <algorithm>

namespace ucc {

void RemoveNoisePass::run(HirModule& hir) {
    hir.ops.erase(std::remove_if(hir.ops.begin(), hir.ops.end(),
                                 [](const HeisenbergOp& op) {
                                     return op.op_type() == OpType::NOISE ||
                                            op.op_type() == OpType::READOUT_NOISE;
                                 }),
                  hir.ops.end());

    hir.noise_sites.clear();
    hir.readout_noise.clear();
    hir.source_map.clear();
}

}  // namespace ucc
