#include "clifft/optimizer/multi_gate_pass.h"

#include <bit>

namespace clifft {

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
                ctrl_mask ^= 1ULL << old_bc[i].axis_1;
                ++run_len;
                ++i;
            }

            if (ctrl_mask == 0) {
                // All gates cancelled (self-inverse pairs)
            } else if (run_len >= 2 && std::popcount(ctrl_mask) >= 2) {
                new_bc.push_back(make_array_multi_cnot(shared_target, ctrl_mask));
            } else {
                // Single surviving gate: emit as plain CNOT
                uint16_t ctrl_axis = static_cast<uint16_t>(std::countr_zero(ctrl_mask));
                new_bc.push_back(make_array_cnot(ctrl_axis, shared_target));
            }

            if (ctrl_mask != 0 && has_sm)
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
                target_mask ^= 1ULL << old_bc[i].axis_2;
                ++run_len;
                ++i;
            }

            if (target_mask == 0) {
                // All gates cancelled
            } else if (run_len >= 2 && std::popcount(target_mask) >= 2) {
                new_bc.push_back(make_array_multi_cz(shared_ctrl, target_mask));
            } else {
                uint16_t tgt_axis = static_cast<uint16_t>(std::countr_zero(target_mask));
                new_bc.push_back(make_array_cz(shared_ctrl, tgt_axis));
            }

            if (target_mask != 0 && has_sm)
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

}  // namespace clifft
