#include "clifft/optimizer/bytecode_pass.h"

namespace clifft {

void BytecodePassManager::add_pass(std::unique_ptr<BytecodePass> pass) {
    passes_.push_back(std::move(pass));
}

void BytecodePassManager::run(CompiledModule& module) {
    for (auto& pass : passes_) {
        pass->run(module);
    }
}

}  // namespace clifft
