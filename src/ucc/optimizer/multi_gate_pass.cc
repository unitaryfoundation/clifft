#include "ucc/optimizer/multi_gate_pass.h"

namespace ucc {

void MultiGatePass::run(CompiledModule& module) {
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
        if (old_bc[i].opcode == Opcode::OP_ARRAY_CNOT) {
            uint16_t shared_target = old_bc[i].axis_2;
            size_t run_start = i;
            uint64_t ctrl_mask = 0;
            size_t run_len = 0;

            while (i < old_bc.size() && old_bc[i].opcode == Opcode::OP_ARRAY_CNOT &&
                   old_bc[i].axis_2 == shared_target) {
                ctrl_mask |= 1ULL << old_bc[i].axis_1;
                ++run_len;
                ++i;
            }

            if (run_len >= 2) {
                new_bc.push_back(make_array_multi_cnot(shared_target, ctrl_mask));
            } else {
                new_bc.push_back(old_bc[run_start]);
            }

            if (has_sm)
                new_sm.merge_entries(old_sm, run_start, run_start + run_len);
            continue;
        }

        if (old_bc[i].opcode == Opcode::OP_ARRAY_CZ) {
            uint16_t shared_ctrl = old_bc[i].axis_1;
            size_t run_start = i;
            uint64_t target_mask = 0;
            size_t run_len = 0;

            while (i < old_bc.size() && old_bc[i].opcode == Opcode::OP_ARRAY_CZ &&
                   old_bc[i].axis_1 == shared_ctrl) {
                target_mask |= 1ULL << old_bc[i].axis_2;
                ++run_len;
                ++i;
            }

            if (run_len >= 2) {
                new_bc.push_back(make_array_multi_cz(shared_ctrl, target_mask));
            } else {
                new_bc.push_back(old_bc[run_start]);
            }

            if (has_sm)
                new_sm.merge_entries(old_sm, run_start, run_start + run_len);
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
