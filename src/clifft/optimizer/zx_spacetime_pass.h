#pragma once

// Rewrites the HIR through single-copy ZX calculus when that narrows the
// sampled active state.
//
// The circuit becomes a ZX diagram over measurement outcomes and noise bits,
// is reduced, and is read back as Pauli rotations and measurements on fresh
// lines (src/clifft/zx/). Unlike the standard pipeline, which can only
// reorder the traced operations, this changes the spacetime structure: a
// qubit's state is never carried past the point where the diagram no longer
// needs it.
//
// The rewrite preserves the distribution of measurement records, detectors
// and observables, but not the quantum state: the result has no final
// tableau, so final-state queries are unavailable. It keeps every noise
// location and its probability. It needs the traced HIR: run it before any
// other pass.
//
// The pass keeps the rewrite only if, after the default passes, the planned
// peak active width of the rewritten module is strictly below the input's.
// Otherwise, or when the circuit is outside what the lowering covers, it
// leaves the module unchanged and records why. Comparing planned widths makes
// the optimizer depend on the sampling planner, unlike other passes: the
// planner's coordinate changes can land below the structural trace, so a
// structural comparison accepts rewrites that gain nothing.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"
#include "clifft/zx/rewrite.h"

#include <cstdint>
#include <string>

namespace clifft {

// Must both be positive.
using ZxSpacetimeOptions = zx::RewriteOptions;

class ZxSpacetimePass : public HirPass {
  public:
    explicit ZxSpacetimePass(ZxSpacetimeOptions options = {});

    void run(HirModule& hir) override;

    // Statistics from the last run() call: peak active widths after the
    // default passes, planned, or structural when either module exceeds the
    // dense limit. result_peak is zero when no rewrite was built.
    [[nodiscard]] bool applied() const { return applied_; }
    [[nodiscard]] uint32_t input_peak() const { return input_peak_; }
    [[nodiscard]] uint32_t result_peak() const { return result_peak_; }
    // Empty when the rewrite was applied.
    [[nodiscard]] const std::string& reason() const { return reason_; }

  private:
    ZxSpacetimeOptions options_;
    bool applied_ = false;
    uint32_t input_peak_ = 0;
    uint32_t result_peak_ = 0;
    std::string reason_;
};

}  // namespace clifft
