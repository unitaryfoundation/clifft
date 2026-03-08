#include "ucc/optimizer/noise_block_pass.h"

namespace ucc {

void NoiseBlockPass::run(CompiledModule& module) {
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
        if (old_bc[i].opcode != Opcode::OP_NOISE) {
            new_bc.push_back(old_bc[i]);
            if (has_src)
                new_src.push_back(old_src[i]);
            if (has_kh)
                new_kh.push_back(old_kh[i]);
            ++i;
            continue;
        }

        // Start of a noise run. Scan forward for contiguous site indices.
        uint32_t start_site = old_bc[i].pauli.cp_mask_idx;
        size_t run_start = i;
        size_t run_len = 1;
        while (i + run_len < old_bc.size() && old_bc[i + run_len].opcode == Opcode::OP_NOISE &&
               old_bc[i + run_len].pauli.cp_mask_idx == start_site + run_len) {
            ++run_len;
        }

        if (run_len == 1) {
            // Single noise instruction -- keep as-is (no block overhead).
            new_bc.push_back(old_bc[i]);
            if (has_src)
                new_src.push_back(old_src[i]);
            if (has_kh)
                new_kh.push_back(old_kh[i]);
        } else {
            // Coalesce into a single NOISE_BLOCK.
            new_bc.push_back(make_noise_block(start_site, static_cast<uint32_t>(run_len)));
            if (has_src) {
                // Merge source lines from all coalesced instructions.
                std::vector<uint32_t> merged;
                for (size_t j = run_start; j < run_start + run_len; ++j) {
                    for (uint32_t line : old_src[j]) {
                        merged.push_back(line);
                    }
                }
                new_src.push_back(std::move(merged));
            }
            if (has_kh) {
                // active_k doesn't change across noise ops; use the last.
                new_kh.push_back(old_kh[run_start + run_len - 1]);
            }
        }

        i += run_len;
    }

    old_bc = std::move(new_bc);
    if (has_src)
        old_src = std::move(new_src);
    if (has_kh)
        old_kh = std::move(new_kh);
}

}  // namespace ucc
