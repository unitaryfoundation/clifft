#include "clifft/optimizer/drop_non_unitary_pass.h"

#include <algorithm>

namespace clifft {

static_assert(static_cast<int>(OpType::NUM_OP_TYPES) == 9,
              "Update DropNonUnitaryPass when adding a new HIR OpType");

void DropNonUnitaryPass::run(HirModule& hir) {
    std::erase_if(hir.ops, [](const HeisenbergOp& op) {
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
                return true;
            case OpType::NUM_OP_TYPES:
                return true;
        }
        return true;
    });

    hir.noise_sites.clear();
    hir.readout_noise.clear();
    hir.detector_targets.clear();
    hir.observable_targets.clear();
    hir.noise_channel_masks = PauliMaskArena{};

    hir.num_measurements = 0;
    hir.num_hidden_measurements = 0;
    hir.num_detectors = 0;
    hir.num_observables = 0;
    hir.num_exp_vals = 0;
    hir.source_map.clear();
}

}  // namespace clifft
