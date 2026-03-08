#include "ucc/optimizer/multi_gate_pass.h"

namespace ucc {

void MultiGatePass::run(CompiledModule& module) {
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
        // --- ARRAY_CNOT star: contiguous CNOTs sharing a target axis ---
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

            if (has_src) {
                std::vector<uint32_t> merged;
                for (size_t j = run_start; j < run_start + run_len; ++j)
                    for (uint32_t line : old_src[j])
                        merged.push_back(line);
                new_src.push_back(std::move(merged));
            }
            if (has_kh) {
                new_kh.push_back(old_kh[run_start + run_len - 1]);
            }
            continue;
        }

        // --- ARRAY_CZ star: contiguous CZs sharing a first axis ---
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

            if (has_src) {
                std::vector<uint32_t> merged;
                for (size_t j = run_start; j < run_start + run_len; ++j)
                    for (uint32_t line : old_src[j])
                        merged.push_back(line);
                new_src.push_back(std::move(merged));
            }
            if (has_kh) {
                new_kh.push_back(old_kh[run_start + run_len - 1]);
            }
            continue;
        }

        // --- Pass through all other instructions ---
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
