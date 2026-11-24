#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"

#include <memory>
#include <string>
#include <string_view>

namespace clifft {

/// Create an HIR pass by name. Throws std::invalid_argument if unknown.
std::unique_ptr<HirPass> make_hir_pass(std::string_view name);

/// Build an HirPassManager with default-enabled HIR passes.
/// Set include_schedule=false for pipelines whose HIR is already scheduled.
HirPassManager default_hir_pass_manager(bool include_schedule = true);

/// Serialize the pass registry to a JSON string.
/// Format: [{"name":"...","kind":"hir","default":true|false}, ...]
std::string pass_registry_json();

}  // namespace clifft
