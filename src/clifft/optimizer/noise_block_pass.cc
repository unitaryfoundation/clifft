#include "clifft/optimizer/noise_block_pass.h"

namespace clifft {

void NoiseBlockPass::run(CompiledModule& module) {
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
        if (old_bc[i].opcode != Opcode::OP_NOISE) {
            new_bc.push_back(old_bc[i]);
            if (has_sm)
                new_sm.copy_entry(old_sm, i);
            ++i;
            continue;
        }

        uint32_t start_site = old_bc[i].pauli.cp_mask_idx;
        size_t run_start = i;
        size_t run_len = 1;
        while (i + run_len < old_bc.size() && old_bc[i + run_len].opcode == Opcode::OP_NOISE &&
               old_bc[i + run_len].pauli.cp_mask_idx == start_site + run_len) {
            ++run_len;
        }

        if (run_len == 1) {
            new_bc.push_back(old_bc[i]);
        } else {
            new_bc.push_back(make_noise_block(start_site, static_cast<uint32_t>(run_len)));
        }

        if (has_sm)
            new_sm.merge_entries(old_sm, run_start, run_start + run_len);

        i += run_len;
    }

    old_bc = std::move(new_bc);
    if (has_sm)
        old_sm = std::move(new_sm);
}

}  // namespace clifft
