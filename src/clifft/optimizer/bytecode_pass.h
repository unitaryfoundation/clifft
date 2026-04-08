#pragma once

#include "clifft/backend/backend.h"

#include <memory>
#include <vector>

namespace clifft {

/// Abstract base class for bytecode optimization passes.
///
/// Each pass receives a mutable CompiledModule and may rewrite, reorder,
/// or remove instructions. Passes operate on the finalized bytecode after
/// the Back-End has lowered the HIR.
class BytecodePass {
  public:
    virtual void run(CompiledModule& module) = 0;
    virtual ~BytecodePass() = default;
};

/// Runs a sequence of bytecode optimization passes over a CompiledModule.
///
/// Passes execute in the order they were added. Each pass receives
/// the module mutated by all prior passes.
class BytecodePassManager {
  public:
    BytecodePassManager() = default;
    BytecodePassManager(BytecodePassManager&&) = default;
    BytecodePassManager& operator=(BytecodePassManager&&) = default;
    BytecodePassManager(const BytecodePassManager&) = delete;
    BytecodePassManager& operator=(const BytecodePassManager&) = delete;

    void add_pass(std::unique_ptr<BytecodePass> pass);
    void run(CompiledModule& module);

  private:
    std::vector<std::unique_ptr<BytecodePass>> passes_;
};

}  // namespace clifft
