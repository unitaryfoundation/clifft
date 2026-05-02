#include "clifft/optimizer/remove_noise_pass.h"

namespace clifft {

void RemoveNoisePass::run(HirModule& hir) {
    std::erase_if(hir.ops, [](const HeisenbergOp& op) {
        return op.op_type() == OpType::NOISE || op.op_type() == OpType::READOUT_NOISE;
    });

    hir.noise_sites.clear();
    hir.readout_noise.clear();
    // The noise_channel_masks arena is fixed-capacity and references can't
    // be dropped without dropping the arena itself. Replace with an empty
    // arena so the slots don't sit around as dead weight after removal.
    hir.noise_channel_masks = PauliMaskArena{};
    hir.source_map.clear();
}

}  // namespace clifft
