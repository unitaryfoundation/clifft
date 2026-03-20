#pragma once

#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/hir_pass.h"
#include "ucc/optimizer/hir_pass_manager.h"

#include <memory>
#include <string>
#include <string_view>

namespace ucc {

/// Create an HIR pass by name. Throws std::invalid_argument if unknown.
std::unique_ptr<HirPass> make_hir_pass(std::string_view name);

/// Create a bytecode pass by name. Throws std::invalid_argument if unknown.
std::unique_ptr<BytecodePass> make_bytecode_pass(std::string_view name);

/// Build an HirPassManager with all default-enabled HIR passes.
HirPassManager default_hir_pass_manager();

/// Build a BytecodePassManager with all default-enabled bytecode passes.
BytecodePassManager default_bytecode_pass_manager();

/// Serialize the pass registry to a JSON string (no nlohmann dependency).
/// Format: [{"name":"...","kind":"hir"|"bytecode","default":true|false}, ...]
std::string pass_registry_json();

}  // namespace ucc
