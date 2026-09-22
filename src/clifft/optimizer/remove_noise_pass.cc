#include "clifft/optimizer/remove_noise_pass.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace clifft {

void RemoveNoisePass::run(HirModule& hir) {
    auto is_noise = [](const HeisenbergOp& op) {
        return op.op_type() == OpType::NOISE || op.op_type() == OpType::READOUT_NOISE;
    };

    if (hir.source_map.size() == hir.ops.size()) {
        std::vector<std::vector<uint32_t>> retained_source_map;
        retained_source_map.reserve(hir.source_map.size());
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            if (!is_noise(hir.ops[i])) {
                retained_source_map.push_back(std::move(hir.source_map[i]));
            }
        }
        hir.source_map = std::move(retained_source_map);
    } else {
        hir.source_map.clear();
    }
    std::erase_if(hir.ops, is_noise);

    hir.noise_sites.clear();
    hir.readout_noise.clear();
    // The noise_channel_masks arena is fixed-capacity and references can't
    // be dropped without dropping the arena itself. Replace with an empty
    // arena so the slots don't sit around as dead weight after removal.
    hir.noise_channel_masks = PauliMaskArena{};
    // With every noise site gone there is nothing left for a logical
    // position to correct for.
    hir.logical_noise_prefix.clear();
}

}  // namespace clifft
