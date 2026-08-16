#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"

#include <memory>
#include <string>
#include <string_view>

namespace clifft {

/// Create an HIR pass by name. Throws std::invalid_argument if unknown.
std::unique_ptr<HirPass> make_hir_pass(std::string_view name);

/// Build an HirPassManager with all default-enabled HIR passes.
HirPassManager default_hir_pass_manager();

/// Serialize the pass registry to a JSON string.
/// Format: [{"name":"...","kind":"hir","default":true|false}, ...]
std::string pass_registry_json();

}  // namespace clifft
