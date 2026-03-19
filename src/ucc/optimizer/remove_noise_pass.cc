#include "ucc/optimizer/remove_noise_pass.h"

namespace ucc {

void RemoveNoisePass::run(HirModule& hir) {
    std::erase_if(hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::NOISE || op.op_type() == OpType::READOUT_NOISE;
    });

    hir.noise_sites.clear();
    hir.readout_noise.clear();
    hir.source_map.clear();
}

}  // namespace ucc
