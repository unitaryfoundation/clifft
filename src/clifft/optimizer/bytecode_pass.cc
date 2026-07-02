#include "clifft/optimizer/bytecode_pass.h"

#include <utility>
#include <vector>

namespace clifft {

void BytecodePassManager::add_pass(std::unique_ptr<BytecodePass> pass) {
    passes_.push_back(std::move(pass));
}

void BytecodePassManager::run(CompiledModule& module) {
    for (auto& pass : passes_) {
        pass->run(module);
    }
}

void BytecodePassManager::run_segmented(CompiledModule& module,
                                        const std::function<bool(const Instruction&)>& is_fence) {
    // Swap the full instruction stream out and hand each fence-free segment
    // to run() as the module's bytecode. The constant pool and counts stay
    // put, so instructions keep referencing the pool verbatim and per-segment
    // pool appends (fusion nodes) accumulate correctly.
    std::vector<Instruction> full = std::move(module.bytecode);
    module.bytecode.clear();

    // The source map is parallel to the bytecode when present; slice it
    // alongside. A non-parallel map is treated as absent and restored
    // untouched.
    const bool has_map = module.source_map.size() == full.size() && !full.empty();
    SourceMap full_map;
    SourceMap stale_map;
    if (has_map) {
        full_map = std::move(module.source_map);
    } else {
        stale_map = std::move(module.source_map);
    }
    module.source_map = SourceMap{};

    std::vector<Instruction> out;
    out.reserve(full.size());
    SourceMap out_map;
    if (has_map) {
        out_map.reserve(full.size(), full_map.data().size());
    }

    size_t i = 0;
    while (i < full.size()) {
        if (is_fence(full[i])) {
            out.push_back(full[i]);
            if (has_map) {
                out_map.copy_entry(full_map, i);
            }
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < full.size() && !is_fence(full[j])) {
            ++j;
        }

        module.bytecode.assign(full.begin() + i, full.begin() + j);
        if (has_map) {
            SourceMap segment_map;
            segment_map.reserve(j - i);
            for (size_t k = i; k < j; ++k) {
                segment_map.copy_entry(full_map, k);
            }
            module.source_map = std::move(segment_map);
        }
        run(module);
        out.insert(out.end(), module.bytecode.begin(), module.bytecode.end());
        if (has_map) {
            for (size_t k = 0; k < module.source_map.size(); ++k) {
                out_map.copy_entry(module.source_map, k);
            }
        }
        module.bytecode.clear();
        module.source_map = SourceMap{};
        i = j;
    }

    module.bytecode = std::move(out);
    module.source_map = has_map ? std::move(out_map) : std::move(stale_map);
}

}  // namespace clifft
