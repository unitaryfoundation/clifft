#include "clifft/optimizer/bytecode_pass.h"

#include "clifft/backend/backend.h"

namespace clifft {

void BytecodePassManager::add_pass(std::unique_ptr<BytecodePass> pass) {
    passes_.push_back(std::move(pass));
}

void BytecodePassManager::run(CompiledModule& module) {
    for (auto& pass : passes_) {
        pass->run(module);
    }
    // Passes fuse and delete instructions, which invalidates the
    // instrument offset table recorded at lowering.
    rebuild_instrument_offsets(module);
}

}  // namespace clifft
