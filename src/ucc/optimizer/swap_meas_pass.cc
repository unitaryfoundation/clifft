#include "ucc/optimizer/swap_meas_pass.h"

namespace ucc {

void SwapMeasPass::run(CompiledModule& module) {
    auto& old_bc = module.bytecode;
    auto& old_src = module.source_map;
    auto& old_kh = module.active_k_history;
    bool has_src = !old_src.empty();
    bool has_kh = !old_kh.empty();

    std::vector<Instruction> new_bc;
    std::vector<std::vector<uint32_t>> new_src;
    std::vector<uint32_t> new_kh;
    new_bc.reserve(old_bc.size());
    if (has_src)
        new_src.reserve(old_bc.size());
    if (has_kh)
        new_kh.reserve(old_bc.size());

    size_t i = 0;
    while (i < old_bc.size()) {
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_ARRAY_SWAP &&
            old_bc[i + 1].opcode == Opcode::OP_MEAS_ACTIVE_INTERFERE &&
            old_bc[i + 1].axis_1 == old_bc[i].axis_2) {
            // The SWAP routes axis_1 -> axis_2, then MEAS targets axis_2 (= k-1).
            new_bc.push_back(make_swap_meas_interfere(
                old_bc[i].axis_1, old_bc[i].axis_2, old_bc[i + 1].classical.classical_idx,
                (old_bc[i + 1].flags & Instruction::FLAG_SIGN) != 0));

            if (has_src) {
                std::vector<uint32_t> merged;
                for (uint32_t line : old_src[i])
                    merged.push_back(line);
                for (uint32_t line : old_src[i + 1])
                    merged.push_back(line);
                new_src.push_back(std::move(merged));
            }
            if (has_kh) {
                new_kh.push_back(old_kh[i + 1]);
            }
            i += 2;
            continue;
        }

        new_bc.push_back(old_bc[i]);
        if (has_src)
            new_src.push_back(old_src[i]);
        if (has_kh)
            new_kh.push_back(old_kh[i]);
        ++i;
    }

    old_bc = std::move(new_bc);
    if (has_src)
        old_src = std::move(new_src);
    if (has_kh)
        old_kh = std::move(new_kh);
}

}  // namespace ucc
