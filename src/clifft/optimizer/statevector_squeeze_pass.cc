#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "clifft/optimizer/commutation.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace clifft {

namespace {

bool is_expansion(const HeisenbergOp& op) {
    return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
}

bool can_bypass_to(const HirModule& hir, size_t moving, size_t target) {
    assert(moving < target && target < hir.ops.size());
    assert(!is_expansion(hir.ops[target]));
    if (!can_swap(hir.ops[moving], hir.ops[target], hir)) {
        return false;
    }
    for (size_t next = target; next > moving + 1; --next) {
        if (!can_swap(hir.ops[moving], hir.ops[next - 1], hir)) {
            return false;
        }
    }
    return true;
}

}  // namespace

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

    // Sweep 2: Lazy Expansion (rightward bubble of non-Clifford gates).
    // For example, T0 T1 H1 M1 can become T1 H1 M1 T0: T1 is blocked by
    // H1, but T0 commutes with every operation on qubit 1.
    if (hir.ops.size() >= 2) {
        // Avoid full-mask commutation scans when an expansion-only suffix has
        // no useful destination. Crossings move one cached position left and
        // never change the relative order of non-expansions.
        std::vector<size_t> non_expansions;
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            if (!is_expansion(hir.ops[i])) {
                non_expansions.push_back(i);
            }
        }

        for (size_t i = hir.ops.size() - 2;; --i) {
            if (is_expansion(hir.ops[i])) {
                size_t curr = i;
                while (curr < hir.ops.size() - 1) {
                    // Crossing another expansion is useful only when it lets
                    // this one move past a later non-expanding operation.
                    if (is_expansion(hir.ops[curr + 1])) {
                        auto target_it =
                            std::upper_bound(non_expansions.begin(), non_expansions.end(), curr);
                        if (target_it == non_expansions.end()) {
                            break;
                        }
                        const size_t target = *target_it;
                        if (!can_bypass_to(hir, curr, target)) {
                            break;
                        }
                        std::rotate(hir.ops.begin() + static_cast<std::ptrdiff_t>(curr),
                                    hir.ops.begin() + static_cast<std::ptrdiff_t>(curr + 1),
                                    hir.ops.begin() + static_cast<std::ptrdiff_t>(target + 1));
                        if (has_sm) {
                            std::rotate(
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(curr),
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(curr + 1),
                                hir.source_map.begin() + static_cast<std::ptrdiff_t>(target + 1));
                        }
                        --*target_it;
                        curr = target;
                        continue;
                    }

                    if (!can_swap(hir.ops[curr], hir.ops[curr + 1], hir)) {
                        break;
                    }
                    std::swap(hir.ops[curr], hir.ops[curr + 1]);
                    if (has_sm) {
                        std::swap(hir.source_map[curr], hir.source_map[curr + 1]);
                    }
                    auto crossed =
                        std::upper_bound(non_expansions.begin(), non_expansions.end(), curr);
                    assert(crossed != non_expansions.end() && *crossed == curr + 1);
                    --*crossed;
                    ++curr;
                }
            }
            if (i == 0)
                break;
        }
    }
}

}  // namespace clifft
