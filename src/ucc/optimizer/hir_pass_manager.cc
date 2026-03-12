#include "ucc/optimizer/hir_pass_manager.h"

namespace ucc {

void HirPassManager::add_pass(std::unique_ptr<HirPass> pass) {
    passes_.push_back(std::move(pass));
}

void HirPassManager::run(HirModule& hir) {
    for (auto& pass : passes_) {
        pass->run(hir);
    }
}

}  // namespace ucc
