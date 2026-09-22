#include "clifft/optimizer/drop_non_unitary_pass.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace clifft {

static_assert(static_cast<int>(OpType::NUM_OP_TYPES) == 10,
              "Update DropNonUnitaryPass when adding a new HIR OpType");

void DropNonUnitaryPass::run(HirModule& hir) {
    auto is_non_unitary = [](const HeisenbergOp& op) {
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
                return false;
            case OpType::MEASURE:
            case OpType::CONDITIONAL_PAULI:
            case OpType::NOISE:
            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::EXP_VAL:
            case OpType::INSTRUMENT:
                return true;
            case OpType::NUM_OP_TYPES:
                return true;
        }
        return true;
    };

    if (hir.source_map.size() == hir.ops.size()) {
        std::vector<std::vector<uint32_t>> retained_source_map;
        retained_source_map.reserve(hir.source_map.size());
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            if (!is_non_unitary(hir.ops[i])) {
                retained_source_map.push_back(std::move(hir.source_map[i]));
            }
        }
        hir.source_map = std::move(retained_source_map);
    } else {
        hir.source_map.clear();
    }
    std::erase_if(hir.ops, is_non_unitary);

    hir.noise_sites.clear();
    hir.instrument_sites.clear();
    hir.readout_noise.clear();
    hir.detector_targets.clear();
    hir.observable_targets.clear();
    hir.noise_channel_masks = PauliMaskArena{};
    // With every noise site gone there is nothing left for a logical
    // position to correct for.
    hir.logical_noise_prefix.clear();

    hir.num_measurements = 0;
    hir.num_hidden_measurements = 0;
    hir.num_detectors = 0;
    hir.num_observables = 0;
    hir.num_exp_vals = 0;
    hir.neglect_instrument_damping = false;
    hir.forced_traceout_slot.reset();
}

}  // namespace clifft
