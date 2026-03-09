#include "ucc/optimizer/swap_meas_pass.h"

namespace ucc {

void SwapMeasPass::run(CompiledModule& module) {
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

    size_t i = 0;
    while (i < old_bc.size()) {
        if (i + 1 < old_bc.size() && old_bc[i].opcode == Opcode::OP_ARRAY_SWAP &&
            old_bc[i + 1].opcode == Opcode::OP_MEAS_ACTIVE_INTERFERE &&
            old_bc[i + 1].axis_1 == old_bc[i].axis_2) {
            new_bc.push_back(make_swap_meas_interfere(
                old_bc[i].axis_1, old_bc[i].axis_2, old_bc[i + 1].classical.classical_idx,
                (old_bc[i + 1].flags & Instruction::FLAG_SIGN) != 0));

            if (has_src) {
                uint32_t b0 = old_src_off[i], e0 = old_src_off[i + 1];
                uint32_t b1 = old_src_off[i + 1], e1 = old_src_off[i + 2];
                new_src_data.insert(new_src_data.end(), old_src_data.begin() + b0,
                                    old_src_data.begin() + e0);
                new_src_data.insert(new_src_data.end(), old_src_data.begin() + b1,
                                    old_src_data.begin() + e1);
                new_src_off.push_back(static_cast<uint32_t>(new_src_data.size()));
            }
            if (has_kh) {
                new_kh.push_back(old_kh[i + 1]);
            }
            i += 2;
            continue;
        }

        new_bc.push_back(old_bc[i]);
        if (has_src) {
            uint32_t b = old_src_off[i], e = old_src_off[i + 1];
            new_src_data.insert(new_src_data.end(), old_src_data.begin() + b,
                                old_src_data.begin() + e);
            new_src_off.push_back(static_cast<uint32_t>(new_src_data.size()));
        }
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
