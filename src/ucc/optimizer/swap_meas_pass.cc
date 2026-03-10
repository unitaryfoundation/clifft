#include "ucc/optimizer/swap_meas_pass.h"

namespace ucc {

void SwapMeasPass::run(CompiledModule& module) {
    auto& old_bc = module.bytecode;
    auto& old_sm = module.source_map;
    bool has_sm = !old_sm.empty();

    std::vector<Instruction> new_bc;
    SourceMap new_sm;
    new_bc.reserve(old_bc.size());
    if (has_sm)
        new_sm.reserve(old_bc.size(), old_sm.data().size());

    size_t i = 0;
    while (i < old_bc.size()) {
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_ARRAY_SWAP &&
            old_bc[i + 1].opcode == Opcode::OP_MEAS_ACTIVE_INTERFERE &&
            old_bc[i + 1].axis_1 == old_bc[i].axis_2) {
            new_bc.push_back(make_swap_meas_interfere(
                old_bc[i].axis_1, old_bc[i].axis_2, old_bc[i + 1].classical.classical_idx,
                (old_bc[i + 1].flags & Instruction::FLAG_SIGN) != 0));

            if (has_sm)
                new_sm.merge_entries(old_sm, i, i + 2);

            i += 2;
            continue;
        }

        new_bc.push_back(old_bc[i]);
        if (has_sm)
            new_sm.copy_entry(old_sm, i);
        ++i;
    }

    old_bc = std::move(new_bc);
    if (has_sm)
        old_sm = std::move(new_sm);
}

}  // namespace ucc
