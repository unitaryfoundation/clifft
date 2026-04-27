#include "clifft/optimizer/expand_t_pass.h"

namespace clifft {

void ExpandTPass::run(CompiledModule& module) {
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
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_EXPAND) {
            uint16_t axis = old_bc[i].axis_1;
            Opcode next_op = old_bc[i + 1].opcode;

            if ((next_op == Opcode::OP_ARRAY_T || next_op == Opcode::OP_ARRAY_T_DAG) &&
                old_bc[i + 1].axis_1 == axis) {
                Opcode fused_op =
                    (next_op == Opcode::OP_ARRAY_T) ? Opcode::OP_EXPAND_T : Opcode::OP_EXPAND_T_DAG;
                Instruction fused{};
                fused.opcode = fused_op;
                fused.axis_1 = axis;
                new_bc.push_back(fused);

                if (has_sm)
                    new_sm.merge_entries(old_sm, i, i + 2);

                i += 2;
                continue;
            }
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

void ExpandRotPass::run(CompiledModule& module) {
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
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_EXPAND) {
            uint16_t axis = old_bc[i].axis_1;

            if (old_bc[i + 1].opcode == Opcode::OP_ARRAY_ROT && old_bc[i + 1].axis_1 == axis) {
                new_bc.push_back(make_expand_rot(axis, old_bc[i + 1].math.weight_re,
                                                 old_bc[i + 1].math.weight_im));
                if (has_sm)
                    new_sm.merge_entries(old_sm, i, i + 2);
                i += 2;
                continue;
            }
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

}  // namespace clifft
