#include "clifft/optimizer/hir_pass_manager.h"

#include <algorithm>

namespace clifft {

void HirPassManager::add_pass(std::unique_ptr<HirPass> pass) {
    passes_.push_back(std::move(pass));
}

void HirPassManager::run(HirModule& hir) {
    for (auto& pass : passes_) {
        pass->run(hir);
    }
}

bool HirPassManager::preserves_sampling_distribution() const {
    return std::ranges::all_of(
        passes_, [](const auto& pass) { return pass->preserves_sampling_distribution(); });
}

}  // namespace clifft
