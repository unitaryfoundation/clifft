#include "ucc/optimizer/multi_gate_pass.h"

namespace ucc {

void MultiGatePass::run(CompiledModule& module) {
    auto& old_bc = module.bytecode;
    auto& old_src_data = module.source_map_data;
    auto& old_src_off = module.source_map_offsets;
    auto& old_kh = module.active_k_history;
    bool has_src = !old_src_off.empty();
    bool has_kh = !old_kh.empty();

    std::vector<Instruction> new_bc;
    std::vector<uint32_t> new_src_data;
    std::vector<uint32_t> new_src_off;
    std::vector<uint32_t> new_kh;
    new_bc.reserve(old_bc.size());
    if (has_src) {
        new_src_data.reserve(old_src_data.size());
        new_src_off.push_back(0);
    }
    if (has_kh)
        new_kh.reserve(old_bc.size());

    auto merge_src_range = [&](size_t run_start, size_t run_len) {
        if (!has_src)
            return;
        uint32_t b = old_src_off[run_start], e = old_src_off[run_start + run_len];
        new_src_data.insert(new_src_data.end(), old_src_data.begin() + b, old_src_data.begin() + e);
        new_src_off.push_back(static_cast<uint32_t>(new_src_data.size()));
    };

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

            merge_src_range(run_start, run_len);
            if (has_kh)
                new_kh.push_back(old_kh[run_start + run_len - 1]);
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

            merge_src_range(run_start, run_len);
            if (has_kh)
                new_kh.push_back(old_kh[run_start + run_len - 1]);
            continue;
        }

        new_bc.push_back(old_bc[i]);
        merge_src_range(i, 1);
        if (has_kh)
            new_kh.push_back(old_kh[i]);
        ++i;
    }

    old_bc = std::move(new_bc);
    if (has_src) {
        old_src_data = std::move(new_src_data);
        old_src_off = std::move(new_src_off);
    }
    if (has_kh)
        old_kh = std::move(new_kh);
}

}  // namespace ucc
