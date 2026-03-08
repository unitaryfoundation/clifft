#include "ucc/optimizer/expand_t_pass.h"

namespace ucc {

void ExpandTPass::run(CompiledModule& module) {
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
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_EXPAND) {
            uint16_t axis = old_bc[i].axis_1;
            Opcode next_op = old_bc[i + 1].opcode;

            if ((next_op == Opcode::OP_PHASE_T || next_op == Opcode::OP_PHASE_T_DAG) &&
                old_bc[i + 1].axis_1 == axis) {
                Opcode fused_op =
                    (next_op == Opcode::OP_PHASE_T) ? Opcode::OP_EXPAND_T : Opcode::OP_EXPAND_T_DAG;
                Instruction fused{};
                fused.opcode = fused_op;
                fused.axis_1 = axis;
                new_bc.push_back(fused);

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
