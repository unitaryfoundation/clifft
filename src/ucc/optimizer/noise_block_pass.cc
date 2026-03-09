#include "ucc/optimizer/noise_block_pass.h"

namespace ucc {

void NoiseBlockPass::run(CompiledModule& module) {
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
        if (old_bc[i].opcode != Opcode::OP_NOISE) {
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

        if (has_src) {
            uint32_t b = old_src_off[run_start], e = old_src_off[run_start + run_len];
            new_src_data.insert(new_src_data.end(), old_src_data.begin() + b,
                                old_src_data.begin() + e);
            new_src_off.push_back(static_cast<uint32_t>(new_src_data.size()));
        }
        if (has_kh) {
            new_kh.push_back(old_kh[run_start + run_len - 1]);
        }

        i += run_len;
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
