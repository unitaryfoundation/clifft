#include "ucc/optimizer/pass_manager.h"

namespace ucc {

void PassManager::add_pass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
}

void PassManager::run(HirModule& hir) {
    for (auto& pass : passes_) {
        pass->run(hir);
    }
}

}  // namespace ucc
