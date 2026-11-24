#include "clifft/optimizer/hir_pass_manager.h"

#include "clifft/optimizer/active_width_schedule_pass.h"

#include <algorithm>

namespace clifft {

void HirPassManager::add_pass(std::unique_ptr<HirPass> pass) {
    owned_.push_back(std::move(pass));
    add_pass(*owned_.back());
}

void HirPassManager::add_pass(HirPass& pass) {
    // Appending a scheduler replaces an adjacent scheduler so callers adding an
    // explicit pass to the default pipeline do not run scheduling twice.
    if (!passes_.empty() && dynamic_cast<ActiveWidthSchedulePass*>(&pass) &&
        dynamic_cast<ActiveWidthSchedulePass*>(passes_.back())) {
        passes_.back() = &pass;
        return;
    }
    passes_.push_back(&pass);
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
