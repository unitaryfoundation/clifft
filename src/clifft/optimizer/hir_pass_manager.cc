#include "clifft/optimizer/hir_pass_manager.h"

#include <iterator>
#include <utility>
#include <vector>

namespace clifft {

void HirPassManager::add_pass(std::unique_ptr<HirPass> pass) {
    passes_.push_back(std::move(pass));
}

void HirPassManager::run(HirModule& hir) {
    for (auto& pass : passes_) {
        pass->run(hir);
    }
}

void HirPassManager::run_segmented(HirModule& hir,
                                   const std::function<bool(const HeisenbergOp&)>& is_fence) {
    // Swap the full op stream out and hand each fence-free segment to run()
    // as the module's op vector. Everything else on the module (arenas, side
    // tables, counters) stays put, so ops keep referencing it verbatim and a
    // pass sees a normal module whose circuit happens to end at the fence.
    std::vector<HeisenbergOp> full = std::move(hir.ops);
    hir.ops.clear();

    // The source map is parallel to ops when present; slice it alongside.
    // A non-parallel map is treated as absent (as lower() does) and restored
    // untouched.
    const bool has_map = hir.source_map.size() == full.size() && !full.empty();
    std::vector<std::vector<uint32_t>> full_map;
    std::vector<std::vector<uint32_t>> stale_map;
    if (has_map) {
        full_map = std::move(hir.source_map);
    } else {
        stale_map = std::move(hir.source_map);
    }
    hir.source_map.clear();

    std::vector<HeisenbergOp> out;
    out.reserve(full.size());
    std::vector<std::vector<uint32_t>> out_map;
    if (has_map) {
        out_map.reserve(full.size());
    }

    size_t i = 0;
    while (i < full.size()) {
        if (is_fence(full[i])) {
            out.push_back(full[i]);
            if (has_map) {
                out_map.push_back(std::move(full_map[i]));
            }
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < full.size() && !is_fence(full[j])) {
            ++j;
        }

        hir.ops.assign(full.begin() + i, full.begin() + j);
        if (has_map) {
            hir.source_map.assign(std::make_move_iterator(full_map.begin() + i),
                                  std::make_move_iterator(full_map.begin() + j));
        }
        run(hir);
        out.insert(out.end(), hir.ops.begin(), hir.ops.end());
        if (has_map) {
            out_map.insert(out_map.end(), std::make_move_iterator(hir.source_map.begin()),
                           std::make_move_iterator(hir.source_map.end()));
        }
        hir.ops.clear();
        hir.source_map.clear();
        i = j;
    }

    hir.ops = std::move(out);
    hir.source_map = has_map ? std::move(out_map) : std::move(stale_map);
}

}  // namespace clifft
