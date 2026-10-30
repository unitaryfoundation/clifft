#pragma once

// The whole ZX lowering of a traced HIR module: build the single-copy
// diagram, reduce it, read it as a scheduled Pauli-based computation and emit
// it as HIR. ZxSpacetimePass adds its width test on top.

#include "clifft/frontend/hir.h"

#include <cstdint>
#include <optional>
#include <string>

namespace clifft::zx {

struct RewriteOptions {
    // Upper bound on the qubits of the diagram, where every reset continues
    // on a fresh qubit, and on the rewritten module, sign lines included.
    uint32_t max_qubits = 4096;
    // Hadamard-edge toggles allowed while reducing (see Graph::work). Dense
    // circuits make each local complementation quadratic in the degree.
    uint64_t max_reduce_work = uint64_t{1} << 23;
};

// The rewritten module, or nullopt with a reason when the circuit is outside
// what the lowering covers or exceeds a limit. A circuit without measurement
// records is rejected: its only output is the state, which the rewrite does
// not keep.
[[nodiscard]] std::optional<HirModule> rewrite_hir(const HirModule& hir,
                                                   const RewriteOptions& options,
                                                   std::string* reason);

}  // namespace clifft::zx
