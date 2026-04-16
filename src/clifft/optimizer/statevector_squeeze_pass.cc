#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>

namespace clifft {

void StatevectorSqueezePass::run(HirModule& hir) {
    bool has_sm = hir.source_map.size() == hir.ops.size();

    // EXP_VAL acts as a hard barrier via can_swap(), so neither sweep
    // will move operations across an expectation value probe.

    // Sweep 1: Eager Compaction (leftward bubble of MEASUREs)
    for (size_t i = 1; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() != OpType::MEASURE) {
            continue;
        }
        size_t curr = i;
        while (curr > 0 && can_swap(hir.ops[curr - 1], hir.ops[curr], hir)) {
            std::swap(hir.ops[curr - 1], hir.ops[curr]);
            if (has_sm) {
                std::swap(hir.source_map[curr - 1], hir.source_map[curr]);
            }
            --curr;
        }
    }

    // Sweep 2: Lazy Expansion (rightward bubble of non-Clifford gates)
    if (hir.ops.size() >= 2) {
        for (size_t i = hir.ops.size() - 2;; --i) {
            auto t = hir.ops[i].op_type();
            if (t == OpType::T_GATE || t == OpType::PHASE_ROTATION) {
                size_t curr = i;
                while (curr < hir.ops.size() - 1 &&
                       can_swap(hir.ops[curr], hir.ops[curr + 1], hir)) {
                    // Don't uselessly reorder two expanding gates past each other
                    auto nt = hir.ops[curr + 1].op_type();
                    if (nt == OpType::T_GATE || nt == OpType::PHASE_ROTATION) {
                        break;
                    }
                    std::swap(hir.ops[curr], hir.ops[curr + 1]);
                    if (has_sm) {
                        std::swap(hir.source_map[curr], hir.source_map[curr + 1]);
                    }
                    ++curr;
                }
            }
            if (i == 0)
                break;
        }
    }
}

}  // namespace clifft
